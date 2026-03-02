//! OctoFlow CLI library — shared modules for octoflow and octo-media binaries.

/// OctoFlow version string — single source of truth for REPL banner, --version, etc.
pub const VERSION: &str = "1.3.0";

pub mod chat;
pub mod runtime;
pub use runtime as compiler;  // backward compat — existing code uses crate::compiler
pub mod loader;
pub mod repl;
pub mod templates;
pub mod mcp;
pub mod update;

// ── Reorganized module directories (v1.2) ──
pub mod io;
pub mod analysis;
pub mod platform;

// ── Re-exports for backward compatibility ──
// These allow existing code using `crate::http_io`, `crate::preflight`, etc.
// to continue working during the transition.
pub use io::http as http_io;
pub use io::image as image_io;
pub use io::json as json_io;
pub use io::net as net_io;
pub use io::octo as octo_io;
pub use io::regex as regex_io;

pub use analysis::preflight;
pub use analysis::lint;
pub use analysis::range_tracker;

#[cfg(target_os = "windows")]
pub use platform::win32 as win32_sys;
#[cfg(target_os = "windows")]
pub use platform::text_render;
#[cfg(target_os = "windows")]
pub use platform::nvml as nvml_io;
#[cfg(target_os = "windows")]
pub use platform::cpu as cpu_io;
pub use platform::window as window_io;
pub use platform::audio as audio_io;

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

/// Permission scope for file/network/exec access.
#[derive(Debug, Clone)]
pub enum PermScope {
    /// Access denied (default).
    Deny,
    /// All access allowed (bare --allow-read).
    AllowAll,
    /// Access allowed only within specified paths (--allow-read=./data).
    AllowScoped(Vec<PathBuf>),
}

impl Default for PermScope {
    fn default() -> Self { PermScope::Deny }
}

impl PermScope {
    /// Check if this scope permits access (ignoring path — for net/exec).
    pub fn is_allowed(&self) -> bool {
        !matches!(self, PermScope::Deny)
    }

    /// Check if this scope permits access to the given path.
    /// For Deny: always false. For AllowAll: always true.
    /// For AllowScoped: true if path is under any of the allowed directories.
    pub fn allows_path(&self, path: &str) -> bool {
        match self {
            PermScope::Deny => false,
            PermScope::AllowAll => true,
            PermScope::AllowScoped(dirs) => {
                let target = match std::fs::canonicalize(path) {
                    Ok(p) => p,
                    Err(_) => {
                        // If file doesn't exist yet, canonicalize the parent
                        let p = PathBuf::from(path);
                        if let Some(parent) = p.parent() {
                            match std::fs::canonicalize(parent) {
                                Ok(canon_parent) => canon_parent.join(p.file_name().unwrap_or_default()),
                                Err(_) => return false,
                            }
                        } else {
                            return false;
                        }
                    }
                };
                dirs.iter().any(|allowed| {
                    let canon = match std::fs::canonicalize(allowed) {
                        Ok(p) => p,
                        Err(_) => allowed.clone(),
                    };
                    target.starts_with(&canon)
                })
            }
        }
    }
}

/// Input limits — prevents resource exhaustion from oversized files.
pub const MAX_IMAGE_FILE_BYTES: u64 = 100 * 1024 * 1024; // 100 MB
pub const MAX_IMAGE_DIMENSION: u32 = 16384;               // 16384 x 16384
pub const MAX_CSV_FILE_BYTES: u64 = 50 * 1024 * 1024;     // 50 MB
pub const MAX_CSV_VALUES: usize = 10_000_000;              // 10M values
pub const MAX_OCTO_FILE_BYTES: u64 = 500 * 1024 * 1024;   // 500 MB (.octo binary)

/// A runtime value — float, integer, string, or map (structured record).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Float(f32),
    Int(i64),
    Str(String),
    Map(HashMap<String, Value>),
    None,
}

impl Value {
    /// Extract as f32, auto-coercing Int → f32.
    pub fn as_float(&self) -> Result<f32, CliError> {
        match self {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f32),
            Value::Str(s) => Err(CliError::Compile(format!("expected number, got string \"{}\"", s))),
            Value::Map(_) => Err(CliError::Compile("expected number, got map".into())),
            Value::None => Err(CliError::Compile("expected number, got none".into())),
        }
    }

    /// Extract as i64, auto-coercing Float → i64 (truncates).
    pub fn as_int(&self) -> Result<i64, CliError> {
        match self {
            Value::Int(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            Value::Str(s) => Err(CliError::Compile(format!("expected integer, got string \"{}\"", s))),
            Value::Map(_) => Err(CliError::Compile("expected integer, got map".into())),
            Value::None => Err(CliError::Compile("expected integer, got none".into())),
        }
    }

    pub fn as_str(&self) -> Result<&str, CliError> {
        match self {
            Value::Str(s) => Ok(s),
            Value::Float(f) => Err(CliError::Compile(format!("expected string, got number {}", f))),
            Value::Int(i) => Err(CliError::Compile(format!("expected string, got integer {}", i))),
            Value::Map(_) => Err(CliError::Compile("expected string, got map".into())),
            Value::None => Err(CliError::Compile("expected string, got none".into())),
        }
    }

    pub fn is_float(&self) -> bool { matches!(self, Value::Float(_)) }
    pub fn is_int(&self) -> bool { matches!(self, Value::Int(_)) }
    /// True if value is Float or Int.
    pub fn is_numeric(&self) -> bool { matches!(self, Value::Float(_) | Value::Int(_)) }
    pub fn is_str(&self) -> bool { matches!(self, Value::Str(_)) }
    pub fn is_map(&self) -> bool { matches!(self, Value::Map(_)) }
    pub fn is_none(&self) -> bool { matches!(self, Value::None) }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Float(n) => write!(f, "{}", n),
            Value::Int(i) => write!(f, "{}", i),
            Value::Str(s) => write!(f, "{}", s),
            Value::Map(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let items: Vec<String> = keys.iter().map(|k| format!("{}={}", k, map[*k])).collect();
                write!(f, "{{{}}}", items.join(", "))
            }
            Value::None => write!(f, "none"),
        }
    }
}

/// CLI overrides for program parameterization (`--set`, `-i`, `-o`).
#[derive(Debug, Default, Clone)]
pub struct Overrides {
    /// Scalar value overrides from `--set name=value`.
    pub scalars: HashMap<String, Value>,
    /// Input path override from `-i path`.
    pub input_path: Option<String>,
    /// Output path override from `-o path`.
    pub output_path: Option<String>,
    /// Allow file read operations (--allow-read or --allow-read=path).
    pub allow_read: PermScope,
    /// Allow file write operations (--allow-write or --allow-write=path).
    pub allow_write: PermScope,
    /// Allow network access (--allow-net).
    pub allow_net: PermScope,
    /// Allow command execution (--allow-exec).
    pub allow_exec: PermScope,
    /// Allow FFI calls to native shared libraries (--allow-ffi).
    pub allow_ffi: bool,
    /// Override the per-while-loop iteration limit (--max-iters N, default 10_000).
    pub max_iters: Option<usize>,
    /// Verbose inference logging (--verbose).
    pub verbose: bool,
    /// GPU memory quota in bytes (--gpu-max-mb N).
    pub gpu_max_bytes: Option<u64>,
}

/// CLI error type.
#[derive(Debug)]
pub enum CliError {
    Io(String),
    Csv(String),
    Parse(String),
    Compile(String),
    Gpu(String),
    Security(String),
    UndefinedStream(String),
    UndefinedScalar(String),
    UnknownOperation(String),
    /// Runtime error (FFI, dynamic dispatch, etc.)
    Runtime(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Io(msg) => write!(f, "I/O error: {}", msg),
            CliError::Csv(msg) => write!(f, "CSV error: {}", msg),
            CliError::Parse(msg) => write!(f, "Parse error: {}", msg),
            CliError::Compile(msg) => write!(f, "Compile error: {}", msg),
            CliError::Gpu(msg) => write!(f, "GPU error: {}", msg),
            CliError::Security(msg) => write!(f, "security: {}", msg),
            CliError::UndefinedStream(name) => write!(f, "undefined stream: {}", name),
            CliError::UndefinedScalar(name) => write!(f, "undefined scalar: {}", name),
            CliError::UnknownOperation(name) => write!(f, "unknown operation: {}", name),
            CliError::Runtime(msg) => write!(f, "runtime error: {}", msg),
        }
    }
}

impl CliError {
    /// Enrich an error with the source line number for LLM-friendly diagnostics.
    pub fn with_line(self, line: usize) -> Self {
        if line == 0 { return self; }
        let suffix = format!(" (line {})", line);
        match self {
            CliError::Compile(msg) if msg.contains(&suffix) => CliError::Compile(msg),
            CliError::Compile(msg) => CliError::Compile(format!("{}{}", msg, suffix)),
            CliError::Gpu(msg) if msg.contains(&suffix) => CliError::Gpu(msg),
            CliError::Gpu(msg) => CliError::Gpu(format!("{}{}", msg, suffix)),
            CliError::Io(msg) => CliError::Io(format!("{}{}", msg, suffix)),
            CliError::Csv(msg) => CliError::Csv(format!("{}{}", msg, suffix)),
            CliError::Parse(msg) => CliError::Parse(msg), // already has position info
            CliError::Security(msg) => CliError::Security(format!("{}{}", msg, suffix)),
            CliError::Runtime(msg) if msg.contains(&suffix) => CliError::Runtime(msg),
            CliError::Runtime(msg) => CliError::Runtime(format!("{}{}", msg, suffix)),
            CliError::UndefinedStream(n) => CliError::Compile(format!("undefined stream '{}'{}", n, suffix)),
            CliError::UndefinedScalar(n) => CliError::Compile(format!("undefined scalar '{}'{}", n, suffix)),
            CliError::UnknownOperation(n) => CliError::Compile(format!("unknown operation '{}'{}", n, suffix)),
        }
    }
}

impl std::error::Error for CliError {}

/// Extract the line number from an error message that contains `(line N)`.
pub fn extract_error_line(err: &CliError) -> Option<usize> {
    let msg = format!("{}", err);
    if let Some(pos) = msg.rfind("(line ") {
        let rest = &msg[pos + 6..];
        if let Some(end) = rest.find(')') {
            return rest[..end].parse::<usize>().ok();
        }
    }
    None
}

/// Format an error with source line context for human-readable output.
///
/// If the error contains a line number and source is provided, shows:
/// ```text
///   error: Compile error: undefined scalar 'foo' (line 5)
///     5 | let x = foo + 1
///       ^^^^^^^^^^^^^^^^^
/// ```
pub fn format_error_with_source(err: &CliError, source: &str) -> String {
    let base = format!("error: {}", err);

    if let Some(line_num) = extract_error_line(err) {
        if line_num > 0 {
            let lines: Vec<&str> = source.lines().collect();
            if line_num <= lines.len() {
                let src_line = lines[line_num - 1];
                let line_prefix = format!("  {} | ", line_num);
                let underline = " ".repeat(line_prefix.len()) + &"^".repeat(src_line.len().max(1));
                return format!("{}\n{}{}\n{}", base, line_prefix, src_line, underline);
            }
        }
    }

    base
}

/// Assign an error code based on error variant and message content.
///
/// Code ranges (canonical, per ERROR_CATALOG.md v1.0):
/// - E001-E009: I/O errors (E001 generic, E002 CSV, E003 not found, E004 write, E005 exec, E006 seek, E007 octo format)
/// - E010-E019: Parse/syntax errors (E010 generic, E011 unexpected token, E012 expected, E013 unterminated string, E014 invalid number, E015 unexpected char, E016 expected end, E017 expected statement)
/// - E020-E029: Compile errors (E020 generic, E021 undefined var, E022 undefined stream, E023 unknown op, E024 type mismatch, E025 arity, E026 immutability, E027 scope, E028 module not found)
/// - E030-E039: Runtime errors (E030 generic, E031 recursion, E032 iteration limit, E033 index, E034 allocation, E035 FFI, E036 handle, E037 video/frame)
/// - E040-E049: GPU errors (E040 generic, E041 memory quota, E042 tensor)
/// - E050-E059: Security errors (E050 generic, E051 read denied, E052 write denied, E053 network denied, E054 exec denied, E055 FFI denied)
pub fn error_code(err: &CliError) -> &'static str {
    match err {
        CliError::Io(m) => {
            if m.contains("No such file") || m.contains("not found") { "E003" }
            else if m.contains("write") || m.contains("append") || m.contains("save_data") { "E004" }
            else if m.contains("exec") { "E005" }
            else if m.contains("seek") || m.contains("ensure_tensor") || m.contains("ensure_gpu_buffer") { "E006" }
            else if m.contains("unsupported version") || m.contains("invalid UTF-8") || m.contains("invalid column type") || m.contains("raw column data") { "E007" }
            else { "E001" }
        }
        CliError::Csv(_) => "E002",
        CliError::Parse(m) => {
            if m.contains("unterminated string") { "E013" }
            else if m.contains("invalid number") || m.contains("hex") { "E014" }
            else if m.contains("unexpected character") { "E015" }
            else if m.contains("expected 'end'") { "E016" }
            else if m.contains("expected statement") || m.contains("expected newline") { "E017" }
            else if m.contains("unexpected") { "E011" }
            else if m.contains("expected") { "E012" }
            else { "E010" }
        }
        CliError::Compile(m) => {
            if m.contains("Cannot assign") || m.contains("Cannot push") || m.contains("Cannot insert") { "E026" }
            else if m.contains("only be used inside") { "E027" }
            else if (m.contains("Module") || m.contains("module")) && m.contains("not found") { "E028" }
            else if m.contains("undefined scalar") || m.contains("undefined variable") || m.contains("Undefined scalar") || m.contains("Undefined variable") { "E021" }
            else if m.contains("undefined stream") || m.contains("undefined array") || m.contains("Undefined array") { "E022" }
            else if m.contains("unknown operation") || m.contains("unknown function") { "E023" }
            else if m.contains("expected number") || m.contains("expected string") || m.contains("must be a string") || m.contains("must be a number") || m.contains("must be numeric") { "E024" }
            else if m.contains("argument") || m.contains("arity") || m.contains("requires") { "E025" }
            else { "E020" }
        }
        CliError::Gpu(m) => {
            if m.contains("quota") || m.contains("GPU_MAX") { "E041" }
            else if m.contains("tensor") || m.contains("ensure_gpu_buffer") { "E042" }
            else { "E040" }
        }
        CliError::Security(m) => {
            if m.contains("read") { "E051" }
            else if m.contains("write") { "E052" }
            else if m.contains("network") || m.contains("net") { "E053" }
            else if m.contains("exec") || m.contains("command") { "E054" }
            else if m.contains("FFI") || m.contains("ffi") { "E055" }
            else { "E050" }
        }
        CliError::UndefinedStream(_) => "E022",
        CliError::UndefinedScalar(_) => "E021",
        CliError::UnknownOperation(_) => "E023",
        CliError::Runtime(m) => {
            if m.contains("recursion") { "E031" }
            else if m.contains("iteration") || m.contains("max_iters") || m.contains("exceeded") { "E032" }
            else if m.contains("index") { "E033" }
            else if m.contains("mem_alloc") || m.contains("allocation") { "E034" }
            else if m.contains("FFI") { "E035" }
            else if m.contains("handle") || m.contains("mem_free") { "E036" }
            else if m.contains("video") || m.contains("frame") { "E037" }
            else { "E030" }
        }
    }
}

/// Strip the `(line N)` suffix from an error message to get the clean message.
pub fn clean_error_message(err: &CliError) -> String {
    let msg = match err {
        CliError::Io(m) | CliError::Csv(m) | CliError::Parse(m) | CliError::Compile(m)
        | CliError::Gpu(m) | CliError::Security(m) | CliError::Runtime(m) => m.clone(),
        CliError::UndefinedStream(n) => format!("undefined stream '{}'", n),
        CliError::UndefinedScalar(n) => format!("undefined scalar '{}'", n),
        CliError::UnknownOperation(n) => format!("unknown operation '{}'", n),
    };
    // Strip trailing (line N)
    if let Some(pos) = msg.rfind(" (line ") {
        let rest = &msg[pos + 7..];
        if rest.ends_with(')') && rest[..rest.len() - 1].parse::<usize>().is_ok() {
            return msg[..pos].to_string();
        }
    }
    msg
}

/// Extract a fix suggestion from an error message.
///
/// Mines "did you mean: X?" patterns and generates suggestions based on error type.
pub fn error_suggestion(err: &CliError) -> Option<String> {
    let msg = format!("{}", err);

    // Extract existing "did you mean" suggestions
    if let Some(pos) = msg.find("did you mean: ") {
        let rest = &msg[pos + 14..];
        let end = rest.find('?').unwrap_or(rest.len());
        return Some(format!("Did you mean '{}'?", &rest[..end]));
    }
    if let Some(pos) = msg.find("Did you mean") {
        let rest = &msg[pos..];
        let end = rest.find('?').map(|p| p + 1).unwrap_or(rest.len());
        return Some(rest[..end].to_string());
    }

    // Generate suggestions based on error type (per ERROR_CATALOG.md fix patterns)
    match err {
        CliError::UndefinedScalar(name) => {
            Some(format!("Add 'let {} = 0.0' before first use", name))
        }
        CliError::UndefinedStream(name) => {
            Some(format!("Add 'let {} = []' before first use", name))
        }
        CliError::Compile(m) if m.contains("undefined scalar") || m.contains("Undefined scalar") => {
            let var = m.strip_prefix("undefined scalar '")
                .or_else(|| m.strip_prefix("Undefined scalar '"))
                .and_then(|s| s.split('\'').next())
                .unwrap_or("x");
            Some(format!("Add 'let {} = 0.0' before first use", var))
        }
        CliError::Compile(m) if m.contains("undefined stream") || m.contains("Undefined array") => {
            let var = m.strip_prefix("undefined stream '")
                .or_else(|| m.strip_prefix("Undefined array '"))
                .and_then(|s| s.split('\'').next())
                .unwrap_or("arr");
            Some(format!("Add 'let {} = []' before first use", var))
        }
        CliError::Compile(m) if m.contains("Cannot assign") || m.contains("Cannot push") || m.contains("Cannot insert") => {
            Some("Variable is immutable. Change 'let x = ...' to 'let mut x = ...'".to_string())
        }
        CliError::Compile(m) if m.contains("only be used inside") => {
            Some("Move this statement inside the appropriate block (loop, function, etc.)".to_string())
        }
        CliError::Compile(m) if m.contains("not found") && (m.contains("Module") || m.contains("module")) => {
            Some("Check module name and ensure the .flow file exists in stdlib/ or locally".to_string())
        }
        CliError::UnknownOperation(name) => {
            Some(format!("'{}' is not a known function or operation", name))
        }
        CliError::Parse(m) if m.contains("unterminated string") => {
            Some("Add closing quote '\"' to the string literal".to_string())
        }
        CliError::Parse(m) if m.contains("expected 'end'") => {
            Some("Add 'end' to close the block".to_string())
        }
        CliError::Parse(m) if m.contains("unexpected end") => {
            Some("Check for missing 'end' keywords or unclosed blocks".to_string())
        }
        CliError::Parse(m) if m.contains("unexpected character") => {
            Some("Remove or replace the unexpected character".to_string())
        }
        CliError::Io(m) if m.contains("No such file") || m.contains("not found") => {
            Some("Check file path with file_exists(path) before reading. Wrap in try(): let r = try(read_file(path))".to_string())
        }
        CliError::Security(m) if m.contains("read") => {
            Some("Run with --allow-read flag (or --allow-read=./path for scoped access)".to_string())
        }
        CliError::Security(m) if m.contains("write") => {
            Some("Run with --allow-write flag (or --allow-write=./path for scoped access)".to_string())
        }
        CliError::Security(m) if m.contains("network") || m.contains("net") => {
            Some("Run with --allow-net flag for network access".to_string())
        }
        CliError::Security(m) if m.contains("exec") || m.contains("command") => {
            Some("Run with --allow-exec flag for command execution".to_string())
        }
        CliError::Security(m) if m.contains("FFI") || m.contains("ffi") => {
            Some("Run with --allow-ffi flag for FFI calls".to_string())
        }
        CliError::Security(_) => {
            Some("Use --allow-read, --allow-write, or --allow-net flags".to_string())
        }
        CliError::Gpu(m) if m.contains("quota") || m.contains("GPU_MAX") => {
            Some("Reduce array sizes or increase --gpu-max-mb limit".to_string())
        }
        CliError::Gpu(m) if m.contains("tensor") => {
            Some("Verify tensor file exists and has supported data type (f32)".to_string())
        }
        CliError::Runtime(m) if m.contains("recursion") => {
            Some("Add a base case to stop recursion, or reduce nesting depth".to_string())
        }
        CliError::Runtime(m) if m.contains("iteration") || m.contains("max_iters") => {
            Some("Add a break condition to the loop, or increase --max-iters limit".to_string())
        }
        CliError::Runtime(m) if m.contains("index") => {
            Some("Check array length with len(arr) before indexing".to_string())
        }
        _ => None,
    }
}

/// Format an error as structured JSON for machine-readable output.
///
/// Output format:
/// ```json
/// {"code":"E021","message":"undefined scalar 'foo'","line":5,"suggestion":"Add 'let foo = 0.0'","context":"  let x = foo + 1"}
/// ```
///
/// When `source` is provided, the context field contains the source line at the error location.
pub fn format_error_json(err: &CliError) -> String {
    format_error_json_full(err, None)
}

/// Format an error as structured JSON with optional source context.
pub fn format_error_json_full(err: &CliError, source: Option<&str>) -> String {
    let code = error_code(err);
    let message = clean_error_message(err);
    let line = extract_error_line(err);
    let suggestion = error_suggestion(err);

    // Get source context line
    let context = line.and_then(|n| {
        source.and_then(|src| {
            let lines: Vec<&str> = src.lines().collect();
            if n > 0 && n <= lines.len() {
                Some(lines[n - 1].to_string())
            } else {
                None
            }
        })
    });

    let escaped_msg = json_escape_str(&message);

    let mut out = format!("{{\"code\":\"{}\",\"message\":\"{}\"", code, escaped_msg);

    if let Some(n) = line {
        out.push_str(&format!(",\"line\":{}", n));
    }

    if let Some(ref sug) = suggestion {
        out.push_str(&format!(",\"suggestion\":\"{}\"", json_escape_str(sug)));
    }

    if let Some(ref ctx) = context {
        out.push_str(&format!(",\"context\":\"{}\"", json_escape_str(ctx)));
    }

    out.push('}');
    out
}

/// Format a preflight report as JSON array of structured errors.
pub fn format_preflight_json(report: &analysis::preflight::PreflightReport, source: &str) -> String {
    let mut items = Vec::new();
    for check in &report.checks {
        for error in &check.errors {
            let context = {
                let lines: Vec<&str> = source.lines().collect();
                if error.line > 0 && error.line <= lines.len() {
                    Some(lines[error.line - 1])
                } else {
                    None
                }
            };
            let escaped_msg = json_escape_str(&error.message);
            let escaped_sug = json_escape_str(&error.suggestion);
            let mut item = format!(
                "{{\"code\":\"E020\",\"message\":\"{}\",\"line\":{}",
                escaped_msg, error.line
            );
            if !error.suggestion.is_empty() {
                item.push_str(&format!(",\"suggestion\":\"{}\"", escaped_sug));
            }
            if let Some(ctx) = context {
                item.push_str(&format!(",\"context\":\"{}\"", json_escape_str(ctx)));
            }
            item.push('}');
            items.push(item);
        }
    }
    format!("[{}]", items.join(","))
}

/// Escape a string for embedding in JSON (without surrounding quotes).
fn json_escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Compute Levenshtein (edit) distance between two strings.
/// Used by both preflight and runtime for "did you mean?" suggestions.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];

    for i in 0..=a.len() { dp[i][0] = i; }
    for j in 0..=b.len() { dp[0][j] = j; }

    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[a.len()][b.len()]
}

/// Suggest the closest match from a list of candidates (max edit distance 2).
pub fn suggest_closest(typo: &str, candidates: &[&str]) -> Option<String> {
    candidates.iter()
        .filter_map(|&c| {
            let d = levenshtein(typo, c);
            if d > 0 && d <= 2 { Some((d, c.to_string())) } else { None }
        })
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- error_code tests ---

    #[test]
    fn test_error_code_io() {
        assert_eq!(error_code(&CliError::Io("fail".into())), "E001");
        assert_eq!(error_code(&CliError::Io("No such file or directory".into())), "E003");
        assert_eq!(error_code(&CliError::Io("path not found".into())), "E003");
        assert_eq!(error_code(&CliError::Io("write_file failed".into())), "E004");
        assert_eq!(error_code(&CliError::Io("exec failed".into())), "E005");
        assert_eq!(error_code(&CliError::Io("ensure_tensor_cached: seek error".into())), "E006");
        assert_eq!(error_code(&CliError::Io("unsupported version".into())), "E007");
    }

    #[test]
    fn test_error_code_parse() {
        assert_eq!(error_code(&CliError::Parse("unexpected token".into())), "E011");
        assert_eq!(error_code(&CliError::Parse("expected 'end'".into())), "E016");
        assert_eq!(error_code(&CliError::Parse("unterminated string literal".into())), "E013");
        assert_eq!(error_code(&CliError::Parse("invalid number format".into())), "E014");
        assert_eq!(error_code(&CliError::Parse("unexpected character '#'".into())), "E015");
        assert_eq!(error_code(&CliError::Parse("expected statement".into())), "E017");
        assert_eq!(error_code(&CliError::Parse("expected newline".into())), "E017");
        assert_eq!(error_code(&CliError::Parse("expected something else".into())), "E012");
        assert_eq!(error_code(&CliError::Parse("some error".into())), "E010");
    }

    #[test]
    fn test_error_code_compile_granular() {
        assert_eq!(error_code(&CliError::UndefinedScalar("x".into())), "E021");
        assert_eq!(error_code(&CliError::Compile("undefined scalar 'x' (line 5)".into())), "E021");
        assert_eq!(error_code(&CliError::Compile("Undefined variable 'x'".into())), "E021");
        assert_eq!(error_code(&CliError::UndefinedStream("arr".into())), "E022");
        assert_eq!(error_code(&CliError::Compile("Undefined array 'data'".into())), "E022");
        assert_eq!(error_code(&CliError::UnknownOperation("foo".into())), "E023");
        assert_eq!(error_code(&CliError::Compile("expected number, got string".into())), "E024");
        assert_eq!(error_code(&CliError::Compile("must be a string".into())), "E024");
        assert_eq!(error_code(&CliError::Compile("wrong number of arguments".into())), "E025");
        assert_eq!(error_code(&CliError::Compile("requires 2 args".into())), "E025");
        assert_eq!(error_code(&CliError::Compile("Cannot assign to immutable variable".into())), "E026");
        assert_eq!(error_code(&CliError::Compile("Cannot push to immutable".into())), "E026");
        assert_eq!(error_code(&CliError::Compile("break can only be used inside a loop".into())), "E027");
        assert_eq!(error_code(&CliError::Compile("Module 'foo' not found".into())), "E028");
        assert_eq!(error_code(&CliError::Compile("some other compile error".into())), "E020");
    }

    #[test]
    fn test_error_code_runtime_granular() {
        assert_eq!(error_code(&CliError::Runtime("recursion depth exceeded".into())), "E031");
        assert_eq!(error_code(&CliError::Runtime("iteration limit exceeded".into())), "E032");
        assert_eq!(error_code(&CliError::Runtime("max_iters reached".into())), "E032");
        assert_eq!(error_code(&CliError::Runtime("index out of bounds".into())), "E033");
        assert_eq!(error_code(&CliError::Runtime("mem_alloc failed".into())), "E034");
        assert_eq!(error_code(&CliError::Runtime("allocation error".into())), "E034");
        assert_eq!(error_code(&CliError::Runtime("FFI call failed".into())), "E035");
        assert_eq!(error_code(&CliError::Runtime("invalid handle".into())), "E036");
        assert_eq!(error_code(&CliError::Runtime("mem_free invalid".into())), "E036");
        assert_eq!(error_code(&CliError::Runtime("video decode error".into())), "E037");
        assert_eq!(error_code(&CliError::Runtime("frame buffer overflow".into())), "E037");
        assert_eq!(error_code(&CliError::Runtime("some failure".into())), "E030");
    }

    #[test]
    fn test_error_code_gpu_granular() {
        assert_eq!(error_code(&CliError::Gpu("out of memory".into())), "E040");
        assert_eq!(error_code(&CliError::Gpu("GPU quota exceeded".into())), "E041");
        assert_eq!(error_code(&CliError::Gpu("GPU_MAX_BYTES limit".into())), "E041");
        assert_eq!(error_code(&CliError::Gpu("tensor not found".into())), "E042");
        assert_eq!(error_code(&CliError::Gpu("ensure_gpu_buffer failed".into())), "E042");
    }

    #[test]
    fn test_error_code_security_granular() {
        assert_eq!(error_code(&CliError::Security("denied".into())), "E050");
        assert_eq!(error_code(&CliError::Security("file read denied".into())), "E051");
        assert_eq!(error_code(&CliError::Security("file write denied".into())), "E052");
        assert_eq!(error_code(&CliError::Security("network access denied".into())), "E053");
        assert_eq!(error_code(&CliError::Security("exec denied".into())), "E054");
        assert_eq!(error_code(&CliError::Security("command execution denied".into())), "E054");
        assert_eq!(error_code(&CliError::Security("FFI not allowed".into())), "E055");
    }

    // --- clean_error_message tests ---

    #[test]
    fn test_clean_message_strips_line() {
        let err = CliError::Compile("undefined scalar 'x' (line 5)".into());
        assert_eq!(clean_error_message(&err), "undefined scalar 'x'");
    }

    #[test]
    fn test_clean_message_no_line() {
        let err = CliError::Compile("type mismatch".into());
        assert_eq!(clean_error_message(&err), "type mismatch");
    }

    #[test]
    fn test_clean_message_undefined_stream_variant() {
        let err = CliError::UndefinedStream("data".into());
        assert_eq!(clean_error_message(&err), "undefined stream 'data'");
    }

    // --- error_suggestion tests ---

    #[test]
    fn test_suggestion_undefined_scalar() {
        let err = CliError::UndefinedScalar("total".into());
        let sug = error_suggestion(&err);
        assert!(sug.is_some());
        assert!(sug.unwrap().contains("let total"));
    }

    #[test]
    fn test_suggestion_security_read() {
        let err = CliError::Security("file read denied".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("--allow-read"));
    }

    #[test]
    fn test_suggestion_security_write() {
        let err = CliError::Security("file write denied".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("--allow-write"));
    }

    #[test]
    fn test_suggestion_security_net() {
        let err = CliError::Security("network access denied".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("--allow-net"));
    }

    #[test]
    fn test_suggestion_security_exec() {
        let err = CliError::Security("exec denied".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("--allow-exec"));
    }

    #[test]
    fn test_suggestion_security_ffi() {
        let err = CliError::Security("FFI not allowed".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("--allow-ffi"));
    }

    #[test]
    fn test_suggestion_parse_missing_end() {
        let err = CliError::Parse("expected 'end'".into());
        let sug = error_suggestion(&err);
        assert!(sug.is_some());
        assert!(sug.unwrap().contains("end"));
    }

    #[test]
    fn test_suggestion_parse_unterminated_string() {
        let err = CliError::Parse("unterminated string literal".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("closing quote"));
    }

    #[test]
    fn test_suggestion_immutability() {
        let err = CliError::Compile("Cannot assign to immutable variable 'x'".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("let mut"));
    }

    #[test]
    fn test_suggestion_module_not_found() {
        let err = CliError::Compile("Module 'foo' not found".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("module name"));
    }

    #[test]
    fn test_suggestion_io_not_found() {
        let err = CliError::Io("No such file or directory: data.csv".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("file_exists"));
    }

    #[test]
    fn test_suggestion_gpu_quota() {
        let err = CliError::Gpu("GPU quota exceeded: 512MB".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("--gpu-max-mb"));
    }

    #[test]
    fn test_suggestion_recursion() {
        let err = CliError::Runtime("recursion depth exceeded".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("base case"));
    }

    #[test]
    fn test_suggestion_index_bounds() {
        let err = CliError::Runtime("index out of bounds".into());
        let sug = error_suggestion(&err).unwrap();
        assert!(sug.contains("len("));
    }

    // --- format_error_json tests ---

    #[test]
    fn test_format_error_json_basic() {
        let err = CliError::Compile("undefined scalar 'x' (line 5)".into());
        let json = format_error_json(&err);
        assert!(json.contains("\"code\":\"E021\""));
        assert!(json.contains("\"message\":\"undefined scalar 'x'\""));
        assert!(json.contains("\"line\":5"));
    }

    #[test]
    fn test_format_error_json_full_with_source() {
        let err = CliError::Compile("undefined scalar 'x' (line 2)".into());
        let source = "let y = 10\nprint(x)\nlet z = 20";
        let json = format_error_json_full(&err, Some(source));
        assert!(json.contains("\"code\":\"E021\""));
        assert!(json.contains("\"line\":2"));
        assert!(json.contains("\"context\":\"print(x)\""));
    }

    #[test]
    fn test_format_error_json_no_line() {
        let err = CliError::Io("disk full".into());
        let json = format_error_json(&err);
        assert!(json.contains("\"code\":\"E001\""));
        assert!(json.contains("\"message\":\"disk full\""));
        assert!(!json.contains("\"line\""));
    }

    #[test]
    fn test_format_error_json_with_suggestion() {
        let err = CliError::UndefinedScalar("total".into());
        let json = format_error_json(&err);
        assert!(json.contains("\"suggestion\""));
        assert!(json.contains("let total"));
    }

    #[test]
    fn test_format_error_json_escapes_quotes() {
        let err = CliError::Compile("expected \"end\" keyword".into());
        let json = format_error_json(&err);
        assert!(json.contains("\\\"end\\\""));
    }
}
