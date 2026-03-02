//! Pre-flight validation — check programs BEFORE execution.
//!
//! Runs 4 checks in a single AST pass:
//! 1. Operation existence
//! 2. Symbol resolution
//! 3. Argument validation
//! 4. Numeric safety warnings

use std::cell::RefCell;
use std::collections::HashSet;

use octoflow_parser::ast::{Arg, Expr, PrintSegment, Program, ScalarExpr, StageCall, Statement};

thread_local! {
    /// Transitive import guard — canonical paths of already-collected modules.
    static PREFLIGHT_IMPORTED: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

// ── Known operations ────────────────────────────────────────────────

/// Map ops that take exactly 1 argument.
const MAP_OPS_1: &[&str] = &[
    "multiply", "add", "subtract", "divide", "mod", "pow", "min", "max",
    "warm", "cool",
];

/// Map ops that take 0 arguments.
const MAP_OPS_0: &[&str] = &[
    "abs", "sqrt", "negate", "exp", "log", "floor", "ceil", "round", "sin", "cos",
];

/// Map ops that take exactly 2 arguments.
const MAP_OPS_2: &[&str] = &["clamp", "tint"];

/// Temporal ops that take 1 argument.
const TEMPORAL_OPS: &[&str] = &["ema", "decay"];

/// Scan ops that take 0 arguments.
const SCAN_OPS: &[&str] = &["prefix_sum"];

/// Reduce ops (used in `let x = op(stream)`).
const REDUCE_OPS: &[&str] = &["min", "max", "sum", "count"];

/// Scalar functions that take 0 arguments.
const SCALAR_FNS_0: &[&str] = &["random", "time", "os_name", "now", "now_ms", "read_line", "term_supports_graphics", "term_clear", "window_close", "window_alive", "window_poll", "window_event_key", "window_event_x", "window_event_y", "window_width", "window_height", "gguf_prefetch_complete", "vm_gpu_usage", "gdi_text_begin", "cpu_util", "cpu_count", "nvml_init", "nvml_gpu_util", "nvml_mem_util", "nvml_temperature", "nvml_vram_used", "nvml_vram_total", "nvml_power", "nvml_gpu_name", "nvml_clock_gpu", "gguf_tokens_per_sec", "loom_pool_size", "loom_vram_used", "loom_vm_count", "loom_cpu_count", "loom_elapsed_us", "loom_dispatch_time", "loom_pool_info"];

/// Scalar functions that take 1 argument.
const SCALAR_FNS_1: &[&str] = &["abs", "sqrt", "exp", "log", "ln", "sin", "file_mtime", "cos", "floor", "ceil", "round", "str", "float", "int", "trim", "to_upper", "to_lower", "type_of", "try", "env", "mean", "median", "stddev", "variance", "dirname", "basename", "canonicalize_path", "is_file", "is_dir", "is_symlink", "base64_encode", "base64_decode", "hex_encode", "hex_decode", "timestamp", "timestamp_from_unix", "ord", "chr", "float_to_bits", "bits_to_float", "mem_alloc", "mem_free", "mem_size", "mem_from_str", "read_line", "sleep", "term_move_up", "window_title", "print_raw", "to_str", "print_bytes", "gguf_cache_file", "rt_staging_alloc", "rt_staging_ready", "rt_staging_wait", "rt_staging_free", "vm_layer_resident", "vm_shutdown", "vm_build", "vm_execute", "vm_execute_async", "vm_poll", "vm_wait", "gdi_text_w", "gdi_text_h", "gdi_text_off", "gdi_text_height", "vm_free_prog", "loom_build", "loom_run", "loom_launch", "loom_poll", "loom_free", "panic", "clone", "chat_emit_token", "is_none", "is_nan", "is_inf", "loom_shutdown", "loom_prefetch", "loom_wait", "loom_status", "loom_mail_poll", "loom_mail_depth", "loom_park", "loom_unpark", "loom_auto_release", "loom_max_vms", "loom_vram_budget", "loom_threads", "loom_async_read", "loom_await", "loom_vm_info", "octopress_init", "octopress_analyze", "octopress_decode", "octopress_load", "octopress_info", "octopress_stream_open", "octopress_stream_next", "octopress_stream_info", "octopress_stream_reset", "octopress_stream_close", "tokenize"];

/// Scalar functions that take 2 arguments.
const SCALAR_FNS_2: &[&str] = &["pow", "contains", "starts_with", "ends_with", "index_of", "char_at", "repeat", "quantile", "correlation", "add_seconds", "add_minutes", "add_hours", "add_days", "diff_seconds", "diff_days", "diff_hours", "format_datetime", "regex_match", "is_match", "regex_find", "float_byte", "mem_get_u32", "mem_get_f32", "mem_get_ptr", "mem_get_u8", "mem_get_u64", "bit_and", "bit_or", "bit_test", "bit_shl", "bit_shr", "bit_xor", "mem_to_str", "rt_staging_upload", "vm_layer_estimate", "vm_set_heap", "vm_poll_status", "gdi_text_add", "gdi_text_width", "vm_present", "cosine_similarity", "loom_present", "loom_set_heap", "loom_pace", "loom_mailbox", "octopress_encode", "octopress_save", "octopress_gpu_encode", "extend", "array_new"];

/// Scalar functions that take 3 arguments.
const SCALAR_FNS_3: &[&str] = &["clamp", "substr", "replace", "regex_replace", "mem_set_u32", "mem_set_f32", "mem_set_ptr", "mem_set_u8", "mem_set_u64", "mem_to_str_at", "gguf_evict_layer", "gguf_evict_layer_ram", "gguf_prefetch_layer", "vm_boot", "vm_write_globals", "vm_load_weights", "vm_write_metrics", "vm_write_control", "vm_write_control_live", "vm_write_control_u32", "vm_read_metrics", "vm_read_control", "vm_read_globals", "loom_boot", "loom_write", "loom_read_globals", "loom_read_metrics", "loom_read_control", "loom_write_metrics", "loom_write_control", "loom_set_globals", "loom_mail_recv", "loom_auto_spawn", "array_extract"];

/// Additional builtins known at preflight (not in SCALAR_FNS_* because they have
/// special arity handling or are array/map/IO ops). Used for "did you mean?" suggestions.
const EXTRA_KNOWN_FNS: &[&str] = &[
    // Array/map/IO
    "len", "pop", "push", "map_keys", "map_values", "map_get", "map_has", "map_remove", "map_set",
    "map", "first", "last", "min_val", "max_val", "reverse", "sort_array", "unique",
    "join", "find", "range_array", "split", "reduce", "slice", "filter", "map_each", "sort_by",
    "read_file", "read", "write_file", "write_csv", "write_bytes", "append_file", "file_exists", "file_size",
    "is_directory", "file_ext", "file_name", "file_dir", "path_join", "read_lines", "read_bytes", "list_dir",
    "repeat", "clock", "exit",
    // Reduce ops (also work as functions on arrays: sum([1,2,3]))
    "sum", "min", "max", "count",
    // HTTP/Net
    "http_get", "http_post", "http_put", "http_delete", "http_listen", "http_accept", "http_accept_nonblock",
    "http_method", "http_path", "http_query", "http_body", "http_header", "http_respond", "http_respond_json",
    "http_respond_html", "http_respond_image", "http_respond_with_headers",
    "tcp_connect", "tcp_send", "tcp_recv", "tcp_listen", "tcp_close", "tcp_accept",
    "udp_socket", "udp_send_to", "udp_recv_from", "socket_close",
    // JSON
    "json_parse", "json_parse_array", "json_stringify", "load_data", "read_csv",
    // GPU array ops
    "gpu_fill", "gpu_range", "gpu_add", "gpu_sub", "gpu_mul", "gpu_div", "gpu_scale", "gpu_clamp",
    "gpu_pow", "gpu_where", "gpu_cumsum", "gpu_reverse", "gpu_random", "gpu_load_csv", "gpu_load_binary",
    "gpu_ema", "gpu_concat", "gpu_gather", "gpu_scatter", "gpu_run", "mat_transpose", "normalize",
    "gpu_topk", "gpu_topk_indices",
    "gpu_sum", "gpu_min", "gpu_max", "gpu_mean", "gpu_product", "gpu_variance", "gpu_stddev", "dot", "norm",
    "gpu_save_csv", "gpu_save_binary", "gpu_matmul", "mat_mul", "gpu_compute",
    // LLM/GGUF
    "gguf_matvec", "gguf_infer_layer", "gguf_load_tensor", "gguf_tokenize", "gguf_load_vocab",
    "rt_load_file_to_buffer",
    // Window/Terminal
    "window_open", "window_draw", "term_image", "term_image_at", "vm_present",
    // Control
    "format", "assert", "panic", "clone",
];

/// Ops that trigger numeric safety warnings.
const NUMERIC_WARN_OPS: &[&str] = &["divide", "sqrt", "exp", "log", "pow"];

fn all_builtin_ops() -> HashSet<&'static str> {
    let mut s = HashSet::new();
    for &op in MAP_OPS_0.iter()
        .chain(MAP_OPS_1.iter())
        .chain(MAP_OPS_2.iter())
        .chain(TEMPORAL_OPS.iter())
        .chain(SCAN_OPS.iter())
    {
        s.insert(op);
    }
    s
}

fn expected_args(op: &str) -> Option<(usize, usize)> {
    if MAP_OPS_0.contains(&op) || SCAN_OPS.contains(&op) {
        Some((0, 0))
    } else if MAP_OPS_1.contains(&op) || TEMPORAL_OPS.contains(&op) {
        Some((1, 1))
    } else if MAP_OPS_2.contains(&op) {
        Some((2, 2))
    } else {
        None // user-defined fn — skip arg check here
    }
}

// ── Report types ────────────────────────────────────────────────────

pub struct PreflightReport {
    pub checks: Vec<CheckResult>,
    pub passed: bool,
}

pub struct CheckResult {
    pub name: &'static str,
    pub passed: bool,
    pub errors: Vec<PreflightError>,
    pub warnings: Vec<PreflightWarning>,
}

pub struct PreflightError {
    pub message: String,
    pub suggestion: String,
    pub line: usize,
}

pub struct PreflightWarning {
    pub message: String,
    pub suggestion: String,
    pub line: usize,
}

impl std::fmt::Display for PreflightReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for check in &self.checks {
            let has_warnings = !check.warnings.is_empty();
            if check.passed && !has_warnings {
                let detail = match check.name {
                    "Operations" => "All operations recognized",
                    "Symbols" => "All streams and scalars defined",
                    "Arguments" => "All argument counts valid",
                    _ => "OK",
                };
                writeln!(f, "  [PASS] {:<12} {}", check.name, detail)?;
            } else if !check.passed {
                writeln!(
                    f,
                    "  [FAIL] {:<12} {} error{}:",
                    check.name,
                    check.errors.len(),
                    if check.errors.len() == 1 { "" } else { "s" }
                )?;
                for err in &check.errors {
                    if err.line > 0 {
                        writeln!(f, "           - line {}: {}", err.line, err.message)?;
                    } else {
                        writeln!(f, "           - {}", err.message)?;
                    }
                    if !err.suggestion.is_empty() {
                        writeln!(f, "             {}", err.suggestion)?;
                    }
                }
            } else {
                // passed but has warnings
                writeln!(
                    f,
                    "  [WARN] {:<12} {} warning{}:",
                    check.name,
                    check.warnings.len(),
                    if check.warnings.len() == 1 { "" } else { "s" }
                )?;
                for w in &check.warnings {
                    writeln!(f, "           - {}", w.message)?;
                }
                let suggestions: Vec<&str> = check.warnings.iter()
                    .map(|w| w.suggestion.as_str())
                    .filter(|s| !s.is_empty())
                    .collect();
                if !suggestions.is_empty() {
                    writeln!(f, "         Suggestions:")?;
                    for s in &suggestions {
                        writeln!(f, "           - {}", s)?;
                    }
                }
            }
        }

        let total_errors: usize = self.checks.iter().map(|c| c.errors.len()).sum();
        let total_warnings: usize = self.checks.iter().map(|c| c.warnings.len()).sum();

        writeln!(f)?;
        if self.passed && total_warnings == 0 {
            write!(f, "  STATUS: READY")?;
        } else if self.passed {
            write!(
                f,
                "  STATUS: READY ({} warning{})",
                total_warnings,
                if total_warnings == 1 { "" } else { "s" }
            )?;
        } else {
            write!(
                f,
                "  STATUS: BLOCKED -- fix {} error{} before running",
                total_errors,
                if total_errors == 1 { "" } else { "s" }
            )?;
        }
        Ok(())
    }
}

// ── Core validation ─────────────────────────────────────────────────

/// Validate a parsed program. Returns a report with all errors/warnings.
pub fn validate(program: &Program, base_dir: &str) -> PreflightReport {
    // Clear transitive import guard for fresh validation
    PREFLIGHT_IMPORTED.with(|s| s.borrow_mut().clear());
    let builtins = all_builtin_ops();

    let mut op_errors: Vec<PreflightError> = Vec::new();
    let mut sym_errors: Vec<PreflightError> = Vec::new();
    let mut arg_errors: Vec<PreflightError> = Vec::new();
    let mut num_warnings: Vec<PreflightWarning> = Vec::new();

    // Collect all defined names (built up as we walk statements in order)
    let mut defined_streams: HashSet<String> = HashSet::new();
    let mut defined_scalars: HashSet<String> = HashSet::new();
    let mut defined_fns: HashSet<String> = HashSet::new();
    let mut defined_structs: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    let mut defined_arrays: HashSet<String> = HashSet::new();
    let mut defined_maps: HashSet<String> = HashSet::new();
    let mut mutable_scalars: HashSet<String> = HashSet::new();

    // First pass: collect fn, scalar fn, and struct names so forward references work within the file
    for (stmt, _span) in &program.statements {
        if let Statement::FnDecl { name, .. } = stmt {
            defined_fns.insert(name.clone());
        }
        if let Statement::ScalarFnDecl { name, .. } = stmt {
            defined_fns.insert(name.clone());
        }
        if let Statement::StructDef { name, fields } = stmt {
            defined_structs.insert(name.clone(), fields.clone());
        }
    }

    // Process use declarations first to collect imported function names
    for (stmt, _span) in &program.statements {
        if let Statement::UseDecl { module } = stmt {
            collect_module_fns(base_dir, module, &mut defined_fns, &mut defined_structs,
                               &mut defined_scalars, &mut defined_arrays, &mut defined_maps,
                               &mut mutable_scalars);
        }
    }

    // Walk statements in order
    for (stmt, span) in &program.statements {
        let line = span.line;
        match stmt {
            Statement::FnDecl { name: _, params, body } => {
                // Check operations inside the fn body
                let fn_params: HashSet<String> = params.iter().cloned().collect();
                for stage in body {
                    check_stage(
                        stage, &builtins, &defined_fns, &fn_params,
                        &defined_streams, &defined_scalars,
                        &mut op_errors, &mut sym_errors, &mut arg_errors,
                        &mut num_warnings, line,
                    );
                }
            }
            Statement::UseDecl { module } => {
                // Check module file exists (local first, then stdlib fallback)
                let module_path = resolve_path(base_dir, &format!("{}.flow", module));
                if !std::path::Path::new(&module_path).exists() {
                    // Try stdlib directory relative to executable
                    let found_in_stdlib = resolve_stdlib_path(&format!("{}.flow", module)).is_some();
                    if !found_in_stdlib {
                        sym_errors.push(PreflightError {
                            message: format!("Module '{}' not found (looked for '{}')", module, module_path),
                            suggestion: String::new(),
                            line,
                        });
                    }
                }
            }
            Statement::StreamDecl { name, expr } => {
                check_expr(
                    expr, &builtins, &defined_fns,
                    &defined_streams, &defined_scalars,
                    &mut op_errors, &mut sym_errors, &mut arg_errors,
                    &mut num_warnings, line,
                );
                defined_streams.insert(name.clone());
            }
            Statement::StructDef { .. } => {
                // Already collected in first pass
            }
            Statement::ArrayDecl { name, elements, mutable } => {
                for elem in elements {
                    check_scalar_expr(
                        elem,
                        &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                        &mut sym_errors, line,
                    );
                }
                defined_arrays.insert(name.clone());
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::LetDecl { name, value, mutable } => {
                // Check for vec or struct constructor: let v = vec3(...) or let e = Entity(...)
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    // Built-in vec constructors
                    if let Some(dim) = match fn_name.as_str() {
                        "vec2" => Some(2usize),
                        "vec3" => Some(3usize),
                        "vec4" => Some(4usize),
                        _ => None,
                    } {
                        if args.len() != dim {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires {} components, got {}",
                                    line, fn_name, dim, args.len()),
                                suggestion: String::new(),
                                line,
                            });
                        }
                        // Validate component expressions
                        for arg in args {
                            check_scalar_expr(
                                arg,
                                &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                                &mut sym_errors, line,
                            );
                        }
                        // Register component scalars: name.x, name.y, etc.
                        let components = ["x", "y", "z", "w"];
                        for i in 0..args.len().min(4) {
                            defined_scalars.insert(format!("{}.{}", name, components[i]));
                        }
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // User-defined struct constructors
                    if let Some(fields) = defined_structs.get(fn_name) {
                        let fields = fields.clone();
                        if args.len() != fields.len() {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires {} fields, got {}",
                                    line, fn_name, fields.len(), args.len()),
                                suggestion: String::new(),
                                line,
                            });
                        }
                        // Validate component expressions
                        for arg in args {
                            check_scalar_expr(
                                arg,
                                &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                                &mut sym_errors, line,
                            );
                        }
                        // Register field scalars: name.field1, name.field2, etc.
                        for field in &fields {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // try() error handling — register .value, .ok, .error fields
                    if fn_name == "try" {
                        if args.len() != 1 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: try() requires exactly 1 argument, got {}", line, args.len()),
                                suggestion: "Usage: let r = try(expr)".to_string(),
                                line,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(
                                arg,
                                &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                                &mut sym_errors, line,
                            );
                        }
                        for field in &["value", "ok", "error"] {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        continue;
                    }
                    // HTTP client functions — register .status, .body, .ok, .error fields
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        let expected = match fn_name.as_str() {
                            "http_get" | "http_delete" => 1, _ => 2,
                        };
                        if args.len() != expected {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires {} argument(s), got {}", line, fn_name, expected, args.len()),
                                suggestion: if expected == 1 {
                                    format!("Usage: let r = {}(url)", fn_name)
                                } else {
                                    format!("Usage: let r = {}(url, body)", fn_name)
                                },
                                line,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(
                                arg,
                                &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                                &mut sym_errors, line,
                            );
                        }
                        for field in &["status", "body", "ok", "error"] {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        continue;
                    }
                    // read_image(path) — register .width, .height, .r, .g, .b fields
                    if fn_name == "read_image" {
                        for arg in args {
                            check_scalar_expr(
                                arg,
                                &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                                &mut sym_errors, line,
                            );
                        }
                        for field in &["width", "height", "r", "g", "b"] {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        continue;
                    }
                    // Command execution: exec(cmd, ...args) — variadic
                    if fn_name == "exec" {
                        if args.is_empty() {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: exec() requires at least 1 argument (command), got 0", line),
                                suggestion: "Usage: let r = exec(\"command\", \"arg1\", \"arg2\", ...)".to_string(),
                                line,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(
                                arg,
                                &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                                &mut sym_errors, line,
                            );
                        }
                        for field in &["status", "output", "ok", "error"] {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        continue;
                    }
                    if fn_name == "video_open" {
                        if args.len() != 1 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: video_open() requires 1 argument (byte array), got {}", line, args.len()),
                                suggestion: "Usage: let vid = video_open(byte_array)".to_string(),
                                line,
                            });
                        }
                        defined_scalars.insert(name.clone());
                        for field in &["width", "height", "frames", "fps"] {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        continue;
                    }
                    if fn_name == "video_open_file" {
                        if args.len() != 1 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: video_open_file() requires 1 argument (path string), got {}", line, args.len()),
                                suggestion: "Usage: let vid = video_open_file(\"video.mp4\")".to_string(),
                                line,
                            });
                        }
                        defined_scalars.insert(name.clone());
                        for field in &["width", "height", "frames", "fps"] {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        continue;
                    }
                    if fn_name == "video_frame" {
                        if args.len() != 2 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: video_frame() requires 2 arguments (handle, index), got {}", line, args.len()),
                                suggestion: "Usage: let f = video_frame(vid, 0)".to_string(),
                                line,
                            });
                        }
                        for ch in &["r", "g", "b"] {
                            defined_arrays.insert(format!("{}.{}", name, ch));
                        }
                        continue;
                    }
                    // Path operations: join_path(parts...) — variadic (Phase 41)
                    if fn_name == "join_path" {
                        if args.is_empty() {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: join_path() requires at least 1 argument, got 0", line),
                                suggestion: "Usage: let path = join_path(\"/tmp\", \"file.txt\")".to_string(),
                                line,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(
                                arg,
                                &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                                &mut sym_errors, line,
                            );
                        }
                        defined_scalars.insert(name.to_string());
                        continue;
                    }
                    // JSON functions — json_parse registers as map, json_parse_array registers as array
                    if fn_name == "json_parse" || fn_name == "load_data" {
                        if args.len() != 1 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires 1 argument, got {}", line, fn_name, args.len()),
                                suggestion: format!("Usage: let data = {}(str)", fn_name),
                                line,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                        }
                        defined_maps.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // gpu_info() → returns map
                    if fn_name == "gpu_info" {
                        defined_maps.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    if fn_name == "json_parse_array" || fn_name == "read_csv" {
                        if args.len() != 1 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires 1 argument, got {}", line, fn_name, args.len()),
                                suggestion: format!("Usage: let arr = {}(arg)", fn_name),
                                line,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                        }
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // Array-returning string/file functions: split, read_lines, list_dir, regex_split, regex_find_all
                    if matches!(fn_name.as_str(), "split" | "regex_split" | "regex_find_all") {
                        if args.len() != 2 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires 2 arguments, got {}", line, fn_name, args.len()),
                                suggestion: format!("Usage: let arr = {}(str, delim)", fn_name),
                                line,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                        }
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    if matches!(fn_name.as_str(), "read_lines" | "read_bytes" | "read_f32_le" | "list_dir" | "walk_dir" | "capture_groups" | "gpu_compute" | "gdi_text_atlas") {
                        for arg in args {
                            check_scalar_expr(arg, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                        }
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // slice(arr, start, end) → array; range_array(start, end) → array
                    if fn_name == "slice" && args.len() == 3 {
                        for arg in args { check_scalar_expr(arg, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line); }
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    if fn_name == "range_array" && args.len() == 2 {
                        for arg in args { check_scalar_expr(arg, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line); }
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    if fn_name == "mat_mul" {
                        if args.len() != 5 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: mat_mul() requires 5 arguments (a, b, m, n, k), got {}", line, args.len()),
                                suggestion: "Usage: let c = mat_mul(a, b, m, n, k)".into(),
                                line,
                            });
                        }
                        for arg in args { check_scalar_expr(arg, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line); }
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    if fn_name == "gpu_matmul" {
                        if args.len() != 5 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: gpu_matmul() requires 5 arguments (a, b, m, n, k) where A is m×k, B is k×n, got {}", line, args.len()),
                                suggestion: "Usage: let c = gpu_matmul(a, b, m, n, k)".into(),
                                line,
                            });
                        }
                        for arg in args { check_scalar_expr(arg, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line); }
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // GPU array-returning builtins (Phase 75a)
                    if matches!(fn_name.as_str(),
                        "gpu_abs" | "gpu_sqrt" | "gpu_exp" | "gpu_log" | "gpu_negate" |
                        "gpu_floor" | "gpu_ceil" | "gpu_round" | "gpu_sin" | "gpu_cos" |
                        "gpu_scale" | "gpu_clamp" | "gpu_pow" |
                        "gpu_add" | "gpu_sub" | "gpu_mul" | "gpu_div" |
                        "gpu_where" | "gpu_cumsum" | "mat_transpose" | "normalize" |
                        "gpu_fill" | "gpu_range" | "gpu_reverse" |
                        "gpu_load_csv" | "gpu_load_binary" |
                        "gpu_random" | "gpu_matmul" | "gpu_ema" |
                        "gpu_concat" | "gpu_gather" | "gpu_scatter" |
                        "gpu_topk" | "gpu_topk_indices" |
                        "gpu_run" | "gguf_load_tensor" | "gguf_matvec" | "gguf_infer_layer" | "gguf_load_vocab" | "gguf_tokenize" |
                        "rt_load_file_to_buffer" | "vm_read_register" | "loom_read" |
                        "vm_read_metrics" | "vm_read_control" | "vm_read_globals" |
                        "gdi_text_atlas"
                    ) {
                        for arg in args { check_scalar_expr(arg, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line); }
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // Higher-order array functions: filter/map_each/sort_by → result is an array
                    if matches!(fn_name.as_str(), "filter" | "map_each" | "sort_by") && args.len() == 2 {
                        // First arg is array name (skip scalar validation), second is lambda
                        if let ScalarExpr::Ref(arr_name) = &args[0] {
                            if !defined_arrays.contains(arr_name) {
                                sym_errors.push(PreflightError {
                                    message: format!("{}() first argument '{}' is not a defined array", fn_name, arr_name),
                                    suggestion: String::new(),
                                    line,
                                });
                            }
                        }
                        check_scalar_expr(&args[1], &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                }

                check_scalar_expr(
                    value,
                    &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                    &mut sym_errors, line,
                );
                defined_scalars.insert(name.clone());
                // User-fn calls may return arrays — register in both sets
                if let ScalarExpr::FnCall { name: fn_name, .. } = value {
                    if defined_fns.contains(fn_name) {
                        defined_arrays.insert(name.clone());
                    }
                }
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::Assign { name, value } => {
                // Validate that the target is declared as mutable
                if !mutable_scalars.contains(name) {
                    if defined_scalars.contains(name) {
                        sym_errors.push(PreflightError {
                            message: format!("Cannot assign to '{}': not declared as mutable", name),
                            suggestion: "Use 'let mut' when declaring this variable".to_string(),
                            line,
                        });
                    } else {
                        let suggestion = suggest(name, &defined_scalars);
                        sym_errors.push(PreflightError {
                            message: format!("Undefined variable '{}' (used in assignment)", name),
                            suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                            line,
                        });
                    }
                }
                check_scalar_expr(
                    value,
                    &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                    &mut sym_errors, line,
                );
            }
            Statement::ArrayAssign { array, index, value } => {
                if !defined_arrays.contains(array) {
                    sym_errors.push(PreflightError {
                        message: format!("Undefined array '{}' (used in array assignment)", array),
                        suggestion: String::new(),
                        line,
                    });
                } else if !mutable_scalars.contains(array) {
                    sym_errors.push(PreflightError {
                        message: format!("Cannot assign to '{}': not declared as mutable", array),
                        suggestion: "Use 'let mut' when declaring this array".to_string(),
                        line,
                    });
                }
                check_scalar_expr(index, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                check_scalar_expr(value, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
            }
            Statement::ArrayPush { array, value } => {
                if !defined_arrays.contains(array) {
                    sym_errors.push(PreflightError {
                        message: format!("Undefined array '{}' (used in push)", array),
                        suggestion: String::new(),
                        line,
                    });
                } else if !mutable_scalars.contains(array) {
                    mutable_scalars.insert(array.clone());  // Auto-promote to mutable on push
                }
                check_scalar_expr(value, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
            }
            Statement::MapDecl { name, mutable } => {
                defined_maps.insert(name.clone());
                defined_scalars.insert(name.clone());
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::MapInsert { map, key, value } => {
                if !defined_maps.contains(map) {
                    sym_errors.push(PreflightError {
                        message: format!("Undefined map '{}' (used in map_set)", map),
                        suggestion: String::new(),
                        line,
                    });
                } else if !mutable_scalars.contains(map) {
                    sym_errors.push(PreflightError {
                        message: format!("Cannot insert into '{}': not declared as mutable", map),
                        suggestion: "Use 'let mut' when declaring this map".to_string(),
                        line,
                    });
                }
                check_scalar_expr(key, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                check_scalar_expr(value, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
            }
            Statement::Emit { expr, path: _ } => {
                if let Expr::Ref { name } = expr {
                    if !defined_streams.contains(name) {
                        let suggestion = suggest(name, &defined_streams);
                        sym_errors.push(PreflightError {
                            message: format!("Undefined stream '{}' (used in emit)", name),
                            suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                            line,
                        });
                    }
                }
            }
            Statement::Print { segments } => {
                for seg in segments {
                    if let PrintSegment::Scalar { name, .. } = seg {
                        // Handle array indexing: "arr[0]" → check if "arr" is a defined array
                        let base = if let Some(bracket) = name.find('[') { &name[..bracket] } else { name.as_str() };
                        if !defined_scalars.contains(name) && !defined_arrays.contains(name) && !defined_maps.contains(name)
                            && !defined_arrays.contains(base) && !defined_maps.contains(base)
                        {
                            let suggestion = suggest(name, &defined_scalars);
                            sym_errors.push(PreflightError {
                                message: format!("Undefined scalar '{}' (used in print)", name),
                                suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                                line,
                            });
                        }
                    }
                }
            }
            Statement::WhileLoop { condition, body } => {
                check_scalar_expr(
                    condition,
                    &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                    &mut sym_errors, line,
                );
                validate_loop_body(body, &defined_streams, &mut defined_scalars, &mut mutable_scalars, &defined_structs, &mut defined_arrays, &defined_fns, &mut sym_errors);
            }
            Statement::ForLoop { var, start, end, body } => {
                check_scalar_expr(
                    start,
                    &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                    &mut sym_errors, line,
                );
                check_scalar_expr(
                    end,
                    &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                    &mut sym_errors, line,
                );
                defined_scalars.insert(var.clone());
                validate_loop_body(body, &defined_streams, &mut defined_scalars, &mut mutable_scalars, &defined_structs, &mut defined_arrays, &defined_fns, &mut sym_errors);
                defined_scalars.remove(var);
            }
            Statement::ForEachLoop { var, iterable, body } => {
                if !defined_arrays.contains(iterable) {
                    sym_errors.push(PreflightError {
                        message: format!("undefined array '{}' in for-each loop", iterable),
                        suggestion: String::new(),
                        line,
                    });
                }
                defined_scalars.insert(var.clone());
                validate_loop_body(body, &defined_streams, &mut defined_scalars, &mut mutable_scalars, &defined_structs, &mut defined_arrays, &defined_fns, &mut sym_errors);
                defined_scalars.remove(var);
            }
            Statement::IfBlock { condition, body, elif_branches, else_body } => {
                check_scalar_expr(
                    condition,
                    &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                    &mut sym_errors, line,
                );
                validate_if_body(body, &defined_streams, &mut defined_scalars, &mut mutable_scalars, &defined_structs, &mut defined_arrays, &defined_fns, &mut sym_errors);
                for (elif_cond, elif_body) in elif_branches {
                    check_scalar_expr(
                        elif_cond,
                        &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                        &mut sym_errors, line,
                    );
                    validate_if_body(elif_body, &defined_streams, &mut defined_scalars, &mut mutable_scalars, &defined_structs, &mut defined_arrays, &defined_fns, &mut sym_errors);
                }
                validate_if_body(else_body, &defined_streams, &mut defined_scalars, &mut mutable_scalars, &defined_structs, &mut defined_arrays, &defined_fns, &mut sym_errors);
            }
            Statement::ScalarFnDecl { name: _, params, body } => {
                // Register params as local scalars AND arrays for body validation
                // (params may be arrays at call time — we allow both)
                let mut local_scalars = defined_scalars.clone();
                let mut local_mutable = mutable_scalars.clone();
                let mut local_arrays = defined_arrays.clone();
                for p in params {
                    local_scalars.insert(p.clone());
                    local_arrays.insert(p.clone());
                    local_mutable.insert(p.clone());
                }
                validate_loop_body(body, &defined_streams, &mut local_scalars, &mut local_mutable, &defined_structs, &mut local_arrays, &defined_fns, &mut sym_errors);
            }
            Statement::Return { value } => {
                // Return at top level is invalid
                check_scalar_expr(
                    value,
                    &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                    &mut sym_errors, line,
                );
                sym_errors.push(PreflightError {
                    message: "'return' can only be used inside a function".to_string(),
                    suggestion: "Move 'return' inside a fn block".to_string(),
                    line,
                });
            }
            Statement::Break => {
                sym_errors.push(PreflightError {
                    message: "'break' can only be used inside a loop".to_string(),
                    suggestion: "Move 'break' inside a while or for loop".to_string(),
                    line,
                });
            }
            Statement::Continue => {
                sym_errors.push(PreflightError {
                    message: "'continue' can only be used inside a loop".to_string(),
                    suggestion: "Move 'continue' inside a while or for loop".to_string(),
                    line,
                });
            }
            Statement::WriteFile { path, content } | Statement::AppendFile { path, content } => {
                check_scalar_expr(path, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                check_scalar_expr(content, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
            }
            Statement::SaveData { path, map_name } => {
                check_scalar_expr(path, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                if !defined_maps.contains(map_name) {
                    sym_errors.push(PreflightError {
                        message: format!("line {}: save_data() — undefined map '{}'", line, map_name),
                        suggestion: "Declare a map with: let mut m = map()".to_string(),
                        line,
                    });
                }
            }
            Statement::WriteCsv { path, array_name } => {
                check_scalar_expr(path, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                if !defined_arrays.contains(array_name) {
                    sym_errors.push(PreflightError {
                        message: format!("line {}: write_csv() — undefined array '{}'", line, array_name),
                        suggestion: "Declare an array first, e.g.: let data = read_csv(\"input.csv\")".to_string(),
                        line,
                    });
                }
            }
            Statement::WriteBytes { path, array_name } => {
                check_scalar_expr(path, &defined_streams, &defined_scalars, &defined_fns, &defined_arrays, &mut sym_errors, line);
                if !defined_arrays.contains(array_name) {
                    sym_errors.push(PreflightError {
                        message: format!("line {}: write_bytes() — undefined array '{}'", line, array_name),
                        suggestion: "Declare an array first with byte values (0-255)".to_string(),
                        line,
                    });
                }
            }
            Statement::ExternBlock { functions, .. } => {
                // Register extern fn names so calls to them don't trigger "undefined function" errors.
                for f in functions {
                    defined_fns.insert(f.name.clone());
                }
            }
            Statement::ExprStmt { expr } => {
                check_scalar_expr(
                    expr,
                    &defined_streams, &defined_scalars, &defined_fns, &defined_arrays,
                    &mut sym_errors, line,
                );
            }
        }
    }

    // Build check results
    let ops_check = CheckResult {
        name: "Operations",
        passed: op_errors.is_empty(),
        errors: op_errors,
        warnings: Vec::new(),
    };
    let sym_check = CheckResult {
        name: "Symbols",
        passed: sym_errors.is_empty(),
        errors: sym_errors,
        warnings: Vec::new(),
    };
    let arg_check = CheckResult {
        name: "Arguments",
        passed: arg_errors.is_empty(),
        errors: arg_errors,
        warnings: Vec::new(),
    };
    let num_check = CheckResult {
        name: "Numeric",
        passed: true, // warnings never block
        errors: Vec::new(),
        warnings: num_warnings,
    };

    let passed = ops_check.passed && sym_check.passed && arg_check.passed;

    PreflightReport {
        checks: vec![ops_check, sym_check, arg_check, num_check],
        passed,
    }
}

// ── Loop body validation (shared by WhileLoop + ForLoop) ────────────

fn validate_loop_body(
    body: &[(Statement, octoflow_parser::ast::Span)],
    defined_streams: &HashSet<String>,
    defined_scalars: &mut HashSet<String>,
    mutable_scalars: &mut HashSet<String>,
    defined_structs: &std::collections::HashMap<String, Vec<String>>,
    defined_arrays: &mut HashSet<String>,
    defined_fns: &HashSet<String>,
    sym_errors: &mut Vec<PreflightError>,
) {
    for (body_stmt, body_span) in body {
        let bline = body_span.line;
        match body_stmt {
            Statement::LetDecl { name: bname, value, mutable } => {
                // Vec constructors
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    if let Some(dim) = match fn_name.as_str() {
                        "vec2" => Some(2usize), "vec3" => Some(3usize), "vec4" => Some(4usize), _ => None,
                    } {
                        if args.len() != dim {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires {} components, got {}", bline, fn_name, dim, args.len()),
                                suggestion: String::new(), line: bline,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                        }
                        let components = ["x", "y", "z", "w"];
                        for i in 0..args.len().min(4) {
                            defined_scalars.insert(format!("{}.{}", bname, components[i]));
                        }
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    // Struct constructors
                    if let Some(fields) = defined_structs.get(fn_name) {
                        let fields = fields.clone();
                        if args.len() != fields.len() {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires {} fields, got {}", bline, fn_name, fields.len(), args.len()),
                                suggestion: String::new(), line: bline,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                        }
                        for field in &fields {
                            defined_scalars.insert(format!("{}.{}", bname, field));
                        }
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    // try() error handling
                    if fn_name == "try" {
                        if args.len() != 1 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: try() requires exactly 1 argument, got {}", bline, args.len()),
                                suggestion: "Usage: let r = try(expr)".to_string(), line: bline,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                        }
                        for field in &["value", "ok", "error"] {
                            defined_scalars.insert(format!("{}.{}", bname, field));
                        }
                        continue;
                    }
                    // HTTP client functions
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        let expected = match fn_name.as_str() {
                            "http_get" | "http_delete" => 1, _ => 2,
                        };
                        if args.len() != expected {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires {} argument(s), got {}", bline, fn_name, expected, args.len()),
                                suggestion: if expected == 1 {
                                    format!("Usage: let r = {}(url)", fn_name)
                                } else {
                                    format!("Usage: let r = {}(url, body)", fn_name)
                                },
                                line: bline,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                        }
                        for field in &["status", "body", "ok", "error"] {
                            defined_scalars.insert(format!("{}.{}", bname, field));
                        }
                        continue;
                    }
                    // Command execution: exec(cmd, ...args) — variadic
                    if fn_name == "exec" {
                        if args.is_empty() {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: exec() requires at least 1 argument (command), got 0", bline),
                                suggestion: "Usage: let r = exec(\"command\", \"arg1\", \"arg2\", ...)".to_string(),
                                line: bline,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                        }
                        for field in &["status", "output", "ok", "error"] {
                            defined_scalars.insert(format!("{}.{}", bname, field));
                        }
                        continue;
                    }
                    if fn_name == "video_open" {
                        if args.len() != 1 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: video_open() requires 1 argument (byte array), got {}", bline, args.len()),
                                suggestion: "Usage: let vid = video_open(byte_array)".to_string(),
                                line: bline,
                            });
                        }
                        defined_scalars.insert(bname.clone());
                        for field in &["width", "height", "frames", "fps"] {
                            defined_scalars.insert(format!("{}.{}", bname, field));
                        }
                        continue;
                    }
                    if fn_name == "video_open_file" {
                        if args.len() != 1 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: video_open_file() requires 1 argument (path string), got {}", bline, args.len()),
                                suggestion: "Usage: let vid = video_open_file(\"video.mp4\")".to_string(),
                                line: bline,
                            });
                        }
                        defined_scalars.insert(bname.clone());
                        for field in &["width", "height", "frames", "fps"] {
                            defined_scalars.insert(format!("{}.{}", bname, field));
                        }
                        continue;
                    }
                    if fn_name == "video_frame" {
                        if args.len() != 2 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: video_frame() requires 2 arguments (handle, index), got {}", bline, args.len()),
                                suggestion: "Usage: let f = video_frame(vid, 0)".to_string(),
                                line: bline,
                            });
                        }
                        for ch in &["r", "g", "b"] {
                            defined_arrays.insert(format!("{}.{}", bname, ch));
                        }
                        continue;
                    }
                    // JSON/data functions in loop body
                    if fn_name == "json_parse" || fn_name == "load_data" {
                        if args.len() != 1 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires 1 argument, got {}", bline, fn_name, args.len()),
                                suggestion: format!("Usage: let data = {}(str)", fn_name),
                                line: bline,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                        }
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    // gpu_info() in loop body
                    if fn_name == "gpu_info" {
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    if fn_name == "json_parse_array" || fn_name == "read_csv" {
                        if args.len() != 1 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires 1 argument, got {}", bline, fn_name, args.len()),
                                suggestion: format!("Usage: let arr = {}(arg)", fn_name),
                                line: bline,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                        }
                        defined_arrays.insert(bname.clone());
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    // Array-returning string/file functions in loop body
                    if matches!(fn_name.as_str(), "split" | "regex_split" | "regex_find_all") {
                        if args.len() != 2 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: {}() requires 2 arguments, got {}", bline, fn_name, args.len()),
                                suggestion: format!("Usage: let arr = {}(str, delim)", fn_name),
                                line: bline,
                            });
                        }
                        for arg in args {
                            check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                        }
                        defined_arrays.insert(bname.clone());
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    if matches!(fn_name.as_str(), "read_lines" | "read_bytes" | "read_f32_le" | "list_dir" | "walk_dir" | "capture_groups" | "gpu_compute" | "gdi_text_atlas") {
                        for arg in args {
                            check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                        }
                        defined_arrays.insert(bname.clone());
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    if fn_name == "slice" && args.len() == 3 {
                        for arg in args { check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline); }
                        defined_arrays.insert(bname.clone());
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    if fn_name == "range_array" && args.len() == 2 {
                        for arg in args { check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline); }
                        defined_arrays.insert(bname.clone());
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    if fn_name == "mat_mul" {
                        for arg in args { check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline); }
                        defined_arrays.insert(bname.clone());
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    if fn_name == "gpu_matmul" {
                        if args.len() != 5 {
                            sym_errors.push(PreflightError {
                                message: format!("line {}: gpu_matmul() requires 5 arguments (a, b, m, n, k) where A is m×k, B is k×n, got {}", bline, args.len()),
                                suggestion: "Usage: let c = gpu_matmul(a, b, m, n, k)".into(),
                                line: bline,
                            });
                        }
                        for arg in args { check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline); }
                        defined_arrays.insert(bname.clone());
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    // GPU array-returning builtins (Phase 75a) in loop body
                    if matches!(fn_name.as_str(),
                        "gpu_abs" | "gpu_sqrt" | "gpu_exp" | "gpu_log" | "gpu_negate" |
                        "gpu_floor" | "gpu_ceil" | "gpu_round" | "gpu_sin" | "gpu_cos" |
                        "gpu_scale" | "gpu_clamp" | "gpu_pow" |
                        "gpu_add" | "gpu_sub" | "gpu_mul" | "gpu_div" |
                        "gpu_where" | "gpu_cumsum" | "mat_transpose" | "normalize" |
                        "gpu_fill" | "gpu_range" | "gpu_reverse" |
                        "gpu_load_csv" | "gpu_load_binary" |
                        "gpu_random" | "gpu_matmul" | "gpu_ema" |
                        "gpu_concat" | "gpu_gather" | "gpu_scatter" |
                        "gpu_topk" | "gpu_topk_indices" |
                        "gpu_run" | "gguf_load_tensor" | "gguf_matvec" | "gguf_infer_layer" | "gguf_load_vocab" | "gguf_tokenize" |
                        "rt_load_file_to_buffer" | "vm_read_register" | "loom_read" |
                        "vm_read_metrics" | "vm_read_control" | "vm_read_globals" |
                        "gdi_text_atlas"
                    ) {
                        for arg in args { check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline); }
                        defined_arrays.insert(bname.clone());
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    // Higher-order array functions in loop body
                    if matches!(fn_name.as_str(), "filter" | "map_each" | "sort_by") && args.len() == 2 {
                        if let ScalarExpr::Ref(arr_name) = &args[0] {
                            if !defined_arrays.contains(arr_name) {
                                sym_errors.push(PreflightError {
                                    message: format!("{}() first argument '{}' is not a defined array", fn_name, arr_name),
                                    suggestion: String::new(),
                                    line: bline,
                                });
                            }
                        }
                        check_scalar_expr(&args[1], defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                        defined_arrays.insert(bname.clone());
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                }
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                defined_scalars.insert(bname.clone());
                // User-fn calls may return arrays — register in both sets
                if let ScalarExpr::FnCall { name: fn_name, .. } = value {
                    if defined_fns.contains(fn_name) {
                        defined_arrays.insert(bname.clone());
                    }
                }
                if *mutable { mutable_scalars.insert(bname.clone()); }
            }
            Statement::Assign { name: aname, value } => {
                if !mutable_scalars.contains(aname) {
                    if defined_scalars.contains(aname) {
                        sym_errors.push(PreflightError {
                            message: format!("Cannot assign to '{}': not declared as mutable", aname),
                            suggestion: "Use 'let mut' when declaring this variable".to_string(),
                            line: bline,
                        });
                    } else {
                        let suggestion = suggest(aname, defined_scalars);
                        sym_errors.push(PreflightError {
                            message: format!("Undefined variable '{}' (used in assignment)", aname),
                            suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                            line: bline,
                        });
                    }
                }
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::ArrayAssign { array, index, value } => {
                if !defined_arrays.contains(array.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("Undefined array '{}' (used in array assignment)", array),
                        suggestion: String::new(),
                        line: bline,
                    });
                } else if !mutable_scalars.contains(array.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("Cannot assign to '{}': not declared as mutable", array),
                        suggestion: "Use 'let mut' when declaring this array".to_string(),
                        line: bline,
                    });
                }
                check_scalar_expr(index, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::ArrayPush { array, value } => {
                if !defined_arrays.contains(array.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("Undefined array '{}' (used in push)", array),
                        suggestion: String::new(),
                        line: bline,
                    });
                } else if !mutable_scalars.contains(array.as_str()) {
                    mutable_scalars.insert(array.clone());  // Auto-promote to mutable on push
                }
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::Print { segments: segs } => {
                for seg in segs {
                    if let PrintSegment::Scalar { name: sname, .. } = seg {
                        let base = if let Some(bracket) = sname.find('[') { &sname[..bracket] } else { sname.as_str() };
                        if !defined_scalars.contains(sname) && !defined_arrays.contains(sname)
                            && !defined_arrays.contains(base)
                        {
                            let suggestion = suggest(sname, defined_scalars);
                            sym_errors.push(PreflightError {
                                message: format!("Undefined scalar '{}' (used in print)", sname),
                                suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                                line: bline,
                            });
                        }
                    }
                }
            }
            Statement::ArrayDecl { name: aname, elements, mutable } => {
                for elem in elements {
                    check_scalar_expr(elem, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                }
                defined_arrays.insert(aname.clone());
                if *mutable { mutable_scalars.insert(aname.clone()); }
            }
            Statement::WhileLoop { condition, body: inner_body } => {
                check_scalar_expr(condition, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                validate_loop_body(inner_body, defined_streams, defined_scalars, mutable_scalars, defined_structs, defined_arrays, defined_fns, sym_errors);
            }
            Statement::ForLoop { var, start, end, body: inner_body } => {
                check_scalar_expr(start, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                check_scalar_expr(end, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                defined_scalars.insert(var.clone());
                validate_loop_body(inner_body, defined_streams, defined_scalars, mutable_scalars, defined_structs, defined_arrays, defined_fns, sym_errors);
                defined_scalars.remove(var);
            }
            Statement::ForEachLoop { var, iterable, body: inner_body } => {
                if !defined_arrays.contains(iterable) {
                    sym_errors.push(PreflightError {
                        message: format!("undefined array '{}' in for-each loop", iterable),
                        suggestion: String::new(),
                        line: bline,
                    });
                }
                defined_scalars.insert(var.clone());
                validate_loop_body(inner_body, defined_streams, defined_scalars, mutable_scalars, defined_structs, defined_arrays, defined_fns, sym_errors);
                defined_scalars.remove(var);
            }
            Statement::IfBlock { condition, body: if_body, elif_branches, else_body } => {
                check_scalar_expr(condition, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                validate_loop_body(if_body, defined_streams, defined_scalars, mutable_scalars, defined_structs, defined_arrays, defined_fns, sym_errors);
                for (elif_cond, elif_body) in elif_branches {
                    check_scalar_expr(elif_cond, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                    validate_loop_body(elif_body, defined_streams, defined_scalars, mutable_scalars, defined_structs, defined_arrays, defined_fns, sym_errors);
                }
                validate_loop_body(else_body, defined_streams, defined_scalars, mutable_scalars, defined_structs, defined_arrays, defined_fns, sym_errors);
            }
            Statement::Return { value } => {
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                // Return is valid inside fn bodies (which reuse loop body validation)
            }
            Statement::MapDecl { name, mutable } => {
                defined_scalars.insert(name.clone());
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::MapInsert { map, key, value } => {
                if !mutable_scalars.contains(map.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("Cannot insert into '{}': not declared as mutable", map),
                        suggestion: "Use 'let mut' when declaring this map".to_string(),
                        line: bline,
                    });
                }
                check_scalar_expr(key, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::WriteFile { path, content } | Statement::AppendFile { path, content } => {
                check_scalar_expr(path, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                check_scalar_expr(content, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::SaveData { path, .. } => {
                check_scalar_expr(path, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::WriteCsv { path, array_name } => {
                check_scalar_expr(path, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                if !defined_arrays.contains(array_name.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("line {}: write_csv() — undefined array '{}'", bline, array_name),
                        suggestion: "Declare an array first, e.g.: let data = read_csv(\"input.csv\")".to_string(),
                        line: bline,
                    });
                }
            }
            Statement::WriteBytes { path, array_name } => {
                check_scalar_expr(path, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                if !defined_arrays.contains(array_name.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("line {}: write_bytes() — undefined array '{}'", bline, array_name),
                        suggestion: "Declare an array first with byte values (0-255)".to_string(),
                        line: bline,
                    });
                }
            }
            Statement::Break | Statement::Continue => {
                // Valid inside loops — no validation needed
            }
            Statement::ExprStmt { expr } => {
                check_scalar_expr(expr, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            _ => {
                sym_errors.push(PreflightError {
                    message: format!("Only let, assignment, print, array, map, file I/O, while, for, if, break, continue, and return are allowed inside loops/functions"),
                    suggestion: String::new(),
                    line: bline,
                });
            }
        }
    }
}

/// Validate an if-block body (top-level context — same as loop body but no break/continue allowed).
fn validate_if_body(
    body: &[(Statement, octoflow_parser::ast::Span)],
    defined_streams: &HashSet<String>,
    defined_scalars: &mut HashSet<String>,
    mutable_scalars: &mut HashSet<String>,
    defined_structs: &std::collections::HashMap<String, Vec<String>>,
    defined_arrays: &mut HashSet<String>,
    defined_fns: &HashSet<String>,
    sym_errors: &mut Vec<PreflightError>,
) {
    for (body_stmt, body_span) in body {
        let bline = body_span.line;
        match body_stmt {
            Statement::LetDecl { name, value, mutable } => {
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    if matches!(fn_name.as_str(), "vec2" | "vec3" | "vec4") {
                        let components = ["x", "y", "z", "w"];
                        for (i, _) in args.iter().enumerate() {
                            defined_scalars.insert(format!("{}.{}", name, components[i]));
                        }
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    if let Some(fields) = defined_structs.get(fn_name) {
                        for field in fields {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // try() error handling
                    if fn_name == "try" {
                        for field in &["value", "ok", "error"] {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        continue;
                    }
                    // HTTP client functions
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        for field in &["status", "body", "ok", "error"] {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        continue;
                    }
                    // Command execution: exec(cmd, ...args)
                    if fn_name == "exec" {
                        for field in &["status", "output", "ok", "error"] {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        continue;
                    }
                    if fn_name == "video_open" || fn_name == "video_open_file" {
                        defined_scalars.insert(name.clone());
                        for field in &["width", "height", "frames", "fps"] {
                            defined_scalars.insert(format!("{}.{}", name, field));
                        }
                        continue;
                    }
                    if fn_name == "video_frame" {
                        for ch in &["r", "g", "b"] {
                            defined_arrays.insert(format!("{}.{}", name, ch));
                        }
                        continue;
                    }
                    // JSON/data/gpu_info functions in if-body
                    if fn_name == "json_parse" || fn_name == "load_data" || fn_name == "gpu_info" {
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    if fn_name == "json_parse_array" || fn_name == "read_csv" {
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // Array-returning string/file functions in if-body
                    if matches!(fn_name.as_str(), "split" | "regex_split" | "regex_find_all" | "read_lines" | "read_bytes" | "read_f32_le" | "list_dir" | "walk_dir" | "capture_groups" | "gpu_compute" | "mat_mul" |
                        "gpu_abs" | "gpu_sqrt" | "gpu_exp" | "gpu_log" | "gpu_negate" |
                        "gpu_floor" | "gpu_ceil" | "gpu_round" | "gpu_sin" | "gpu_cos" |
                        "gpu_scale" | "gpu_clamp" | "gpu_pow" |
                        "gpu_add" | "gpu_sub" | "gpu_mul" | "gpu_div" |
                        "gpu_where" | "gpu_cumsum" | "mat_transpose" | "normalize" |
                        "gpu_fill" | "gpu_range" | "gpu_reverse" |
                        "gpu_load_csv" | "gpu_load_binary" |
                        "gpu_random" | "gpu_matmul" | "gpu_ema" |
                        "gpu_concat" | "gpu_gather" | "gpu_scatter" |
                        "gpu_topk" | "gpu_topk_indices" |
                        "gpu_run" | "gguf_load_tensor" | "gguf_matvec" | "gguf_infer_layer" | "gguf_load_vocab" | "gguf_tokenize" |
                        "rt_load_file_to_buffer" | "vm_read_register" | "loom_read" |
                        "vm_read_metrics" | "vm_read_control" | "vm_read_globals" |
                        "gdi_text_atlas") {
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // Higher-order array functions in if-body
                    if matches!(fn_name.as_str(), "filter" | "map_each" | "sort_by") {
                        defined_arrays.insert(name.clone());
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                }
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                defined_scalars.insert(name.clone());
                // User-fn calls may return arrays — register in both sets
                if let ScalarExpr::FnCall { name: fn_name, .. } = value {
                    if defined_fns.contains(fn_name) {
                        defined_arrays.insert(name.clone());
                    }
                }
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::Assign { name, value } => {
                if !mutable_scalars.contains(name) {
                    sym_errors.push(PreflightError {
                        message: format!("Cannot assign to '{}': not declared as mutable", name),
                        suggestion: "Declare with 'let mut' to allow reassignment".to_string(),
                        line: bline,
                    });
                }
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::ArrayAssign { array, index, value } => {
                if !defined_arrays.contains(array.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("Undefined array '{}' (used in array assignment)", array),
                        suggestion: String::new(),
                        line: bline,
                    });
                } else if !mutable_scalars.contains(array.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("Cannot assign to '{}': not declared as mutable", array),
                        suggestion: "Use 'let mut' when declaring this array".to_string(),
                        line: bline,
                    });
                }
                check_scalar_expr(index, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::ArrayPush { array, value } => {
                if !defined_arrays.contains(array.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("Undefined array '{}' (used in push)", array),
                        suggestion: String::new(),
                        line: bline,
                    });
                } else if !mutable_scalars.contains(array.as_str()) {
                    mutable_scalars.insert(array.clone());  // Auto-promote to mutable on push
                }
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::Print { segments } => {
                for seg in segments {
                    if let PrintSegment::Scalar { name: sname, .. } = seg {
                        let base = if let Some(bracket) = sname.find('[') { &sname[..bracket] } else { sname.as_str() };
                        if !defined_scalars.contains(sname) && !defined_arrays.contains(sname)
                            && !defined_arrays.contains(base)
                        {
                            let suggestion = suggest(sname, defined_scalars);
                            sym_errors.push(PreflightError {
                                message: format!("Undefined scalar '{}' (used in print)", sname),
                                suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                                line: bline,
                            });
                        }
                    }
                }
            }
            Statement::ArrayDecl { name, elements, mutable } => {
                for elem in elements {
                    check_scalar_expr(elem, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                }
                defined_arrays.insert(name.clone());
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::IfBlock { condition, body: inner_body, elif_branches, else_body } => {
                check_scalar_expr(condition, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                validate_if_body(inner_body, defined_streams, defined_scalars, mutable_scalars, defined_structs, defined_arrays, defined_fns, sym_errors);
                for (elif_cond, elif_body) in elif_branches {
                    check_scalar_expr(elif_cond, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                    validate_if_body(elif_body, defined_streams, defined_scalars, mutable_scalars, defined_structs, defined_arrays, defined_fns, sym_errors);
                }
                validate_if_body(else_body, defined_streams, defined_scalars, mutable_scalars, defined_structs, defined_arrays, defined_fns, sym_errors);
            }
            Statement::MapDecl { name, mutable } => {
                defined_scalars.insert(name.clone());
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::MapInsert { map, key, value } => {
                if !mutable_scalars.contains(map.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("Cannot insert into '{}': not declared as mutable", map),
                        suggestion: "Use 'let mut' when declaring this map".to_string(),
                        line: bline,
                    });
                }
                check_scalar_expr(key, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                check_scalar_expr(value, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::WriteFile { path, content } | Statement::AppendFile { path, content } => {
                check_scalar_expr(path, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                check_scalar_expr(content, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::SaveData { path, .. } => {
                check_scalar_expr(path, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            Statement::WriteCsv { path, array_name } => {
                check_scalar_expr(path, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                if !defined_arrays.contains(array_name.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("line {}: write_csv() — undefined array '{}'", bline, array_name),
                        suggestion: "Declare an array first, e.g.: let data = read_csv(\"input.csv\")".to_string(),
                        line: bline,
                    });
                }
            }
            Statement::WriteBytes { path, array_name } => {
                check_scalar_expr(path, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
                if !defined_arrays.contains(array_name.as_str()) {
                    sym_errors.push(PreflightError {
                        message: format!("line {}: write_bytes() — undefined array '{}'", bline, array_name),
                        suggestion: "Declare an array first with byte values (0-255)".to_string(),
                        line: bline,
                    });
                }
            }
            Statement::ExternBlock { .. } => {
                // ExternBlock is valid inside if-blocks; fn names registered at top-level pass
            }
            Statement::ExprStmt { expr } => {
                check_scalar_expr(expr, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, bline);
            }
            _ => {
                sym_errors.push(PreflightError {
                    message: "Only let, assignment, print, array, map, file I/O, and if blocks are allowed inside if blocks".to_string(),
                    suggestion: String::new(),
                    line: bline,
                });
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn check_expr(
    expr: &Expr,
    builtins: &HashSet<&str>,
    defined_fns: &HashSet<String>,
    defined_streams: &HashSet<String>,
    defined_scalars: &HashSet<String>,
    op_errors: &mut Vec<PreflightError>,
    sym_errors: &mut Vec<PreflightError>,
    arg_errors: &mut Vec<PreflightError>,
    num_warnings: &mut Vec<PreflightWarning>,
    line: usize,
) {
    match expr {
        Expr::Tap { .. } | Expr::RandomStream { .. } | Expr::Cache { .. } => {}
        Expr::Ref { name } => {
            if !defined_streams.contains(name) {
                let suggestion = suggest(name, &defined_streams);
                sym_errors.push(PreflightError {
                    message: format!("Undefined stream '{}'", name),
                    suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                    line,
                });
            }
        }
        Expr::Pipe { input, stages } => {
            check_expr(
                input, builtins, defined_fns,
                defined_streams, defined_scalars,
                op_errors, sym_errors, arg_errors, num_warnings, line,
            );
            let empty_params = HashSet::new();
            for stage in stages {
                check_stage(
                    stage, builtins, defined_fns, &empty_params,
                    defined_streams, defined_scalars,
                    op_errors, sym_errors, arg_errors, num_warnings, line,
                );
            }
        }
    }
}

fn check_stage(
    stage: &StageCall,
    builtins: &HashSet<&str>,
    defined_fns: &HashSet<String>,
    fn_params: &HashSet<String>,
    _defined_streams: &HashSet<String>,
    defined_scalars: &HashSet<String>,
    op_errors: &mut Vec<PreflightError>,
    sym_errors: &mut Vec<PreflightError>,
    arg_errors: &mut Vec<PreflightError>,
    num_warnings: &mut Vec<PreflightWarning>,
    line: usize,
) {
    let op = &stage.operation;

    // For dotted names (module.func), check the full name against defined_fns
    let is_dotted = op.contains('.');
    let is_builtin = builtins.contains(op.as_str());
    let is_fn = defined_fns.contains(op);

    // Check 1: Operation existence
    if !is_builtin && !is_fn {
        let all_names: Vec<&str> = builtins.iter().copied()
            .chain(defined_fns.iter().map(|s| s.as_str()))
            .collect();
        let suggestion = suggest_str(op, &all_names);
        op_errors.push(PreflightError {
            message: format!("Unknown operation '{}'", op),
            suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
            line,
        });
    }

    // Check 2: Arg refs must be defined scalars or fn params
    for arg in &stage.args {
        if let Arg::Ref(name) = arg {
            if !defined_scalars.contains(name) && !fn_params.contains(name)
                && !name.starts_with("ir_") // IR builder module-level state (stdlib/compiler/ir.flow)
            {
                let all_names: HashSet<String> = defined_scalars.union(fn_params).cloned().collect();
                let suggestion = suggest(name, &all_names);
                sym_errors.push(PreflightError {
                    message: format!("Undefined scalar '{}'", name),
                    suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                    line,
                });
            }
        }
    }

    // Check 3: Argument count (only for builtins)
    if is_builtin {
        if let Some((min, max)) = expected_args(op) {
            let got = stage.args.len();
            if got < min || got > max {
                let expected = if min == max {
                    format!("{}", min)
                } else {
                    format!("{}-{}", min, max)
                };
                arg_errors.push(PreflightError {
                    message: format!(
                        "{}() requires {} argument{}, got {}",
                        op, expected,
                        if max == 1 { "" } else { "s" },
                        got
                    ),
                    suggestion: String::new(),
                    line,
                });
            }
        }
    }

    // Check 4: Numeric safety warnings (only for builtins that need them)
    if !is_dotted && NUMERIC_WARN_OPS.contains(&op.as_str()) {
        match op.as_str() {
            "divide" => num_warnings.push(PreflightWarning {
                message: format!("divide(): potential division by zero"),
                suggestion: "Ensure denominator is never zero".to_string(),
                line,
            }),
            "sqrt" => num_warnings.push(PreflightWarning {
                message: format!("sqrt(): input could be negative"),
                suggestion: "Consider adding abs() before sqrt()".to_string(),
                line,
            }),
            "exp" => num_warnings.push(PreflightWarning {
                message: format!("exp(): can overflow float32 for inputs > 88"),
                suggestion: "Consider clamping input first".to_string(),
                line,
            }),
            "log" => num_warnings.push(PreflightWarning {
                message: format!("log(): undefined for zero or negative values"),
                suggestion: "Consider clamping to positive values".to_string(),
                line,
            }),
            "pow" => {
                // Warn if literal exponent > 10
                if let Some(Arg::Literal(n)) = stage.args.first() {
                    if *n > 10.0 {
                        num_warnings.push(PreflightWarning {
                            message: format!("pow({}): large exponent may cause overflow", n),
                            suggestion: "Consider clamping input or using a smaller exponent".to_string(),
                            line,
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

fn check_scalar_expr(
    expr: &ScalarExpr,
    defined_streams: &HashSet<String>,
    defined_scalars: &HashSet<String>,
    defined_fns: &HashSet<String>,
    defined_arrays: &HashSet<String>,
    sym_errors: &mut Vec<PreflightError>,
    line: usize,
) {
    match expr {
        ScalarExpr::Reduce { op, stream } => {
            if !REDUCE_OPS.contains(&op.as_str()) {
                sym_errors.push(PreflightError {
                    message: format!("Unknown reduce operation '{}'", op),
                    suggestion: format!("Valid reduce ops: {}", REDUCE_OPS.join(", ")),
                    line,
                });
            }
            if !defined_streams.contains(stream) {
                let suggestion = suggest(stream, &defined_streams);
                sym_errors.push(PreflightError {
                    message: format!("Undefined stream '{}' (used in {}())", stream, op),
                    suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                    line,
                });
            }
        }
        ScalarExpr::BinOp { left, right, .. } => {
            check_scalar_expr(left, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
            check_scalar_expr(right, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
        }
        ScalarExpr::Compare { left, right, .. } => {
            check_scalar_expr(left, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
            check_scalar_expr(right, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
        }
        ScalarExpr::And { left, right } | ScalarExpr::Or { left, right } => {
            check_scalar_expr(left, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
            check_scalar_expr(right, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
        }
        ScalarExpr::If { condition, then_expr, else_expr } => {
            check_scalar_expr(condition, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
            check_scalar_expr(then_expr, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
            check_scalar_expr(else_expr, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
        }
        ScalarExpr::Ref(name) => {
            // Accept both scalars and arrays (arrays can be passed to functions)
            if !defined_scalars.contains(name) && !defined_arrays.contains(name)
                && !name.starts_with("ir_") // IR builder module-level state (stdlib/compiler/ir.flow)
            {
                let suggestion = suggest(name, &defined_scalars);
                sym_errors.push(PreflightError {
                    message: format!("Undefined scalar '{}'", name),
                    suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                    line,
                });
            }
        }
        ScalarExpr::Literal(_) | ScalarExpr::IntLiteral(_) | ScalarExpr::Bool(_) | ScalarExpr::StringLiteral(_) | ScalarExpr::NoneLiteral => {}
        ScalarExpr::FnCall { name, args } => {
            // Vec constructors: validated at LetDecl level, skip here
            let expected_arity = if matches!(name.as_str(), "vec2" | "vec3" | "vec4") {
                let dim = match name.as_str() {
                    "vec2" => 2, "vec3" => 3, "vec4" => 4, _ => 0,
                };
                Some(dim)
            } else if SCALAR_FNS_0.contains(&name.as_str()) {
                Some(0)
            } else if name == "len" || name == "pop" || name == "map_keys"
                || name == "first" || name == "last" || name == "min_val" || name == "max_val"
                || name == "reverse" || name == "sort_array" || name == "unique" {
                Some(1)
            } else if matches!(name.as_str(), "map_get" | "map_has" | "map_remove" | "join" | "find" | "range_array") {
                Some(2)
            } else if matches!(name.as_str(), "read_file" | "read" | "file_exists" | "file_size" | "is_directory" | "file_ext" | "file_name" | "file_dir" | "read_lines" | "read_bytes" | "read_f32_le" | "list_dir" | "walk_dir") {
                Some(1)
            } else if matches!(name.as_str(), "path_join" | "split" | "regex_split" | "regex_find_all" | "capture_groups") {
                Some(2)
            } else if matches!(name.as_str(), "http_get" | "http_delete") {
                Some(1)
            } else if matches!(name.as_str(), "http_post" | "http_put") {
                Some(2)
            } else if matches!(name.as_str(), "json_parse" | "json_parse_array" | "json_stringify" | "load_data" | "read_csv") {
                Some(1)
            } else if matches!(name.as_str(), "filter" | "map_each" | "sort_by") {
                Some(2)
            } else if name == "reduce" {
                Some(3)
            } else if name == "slice" {
                Some(3)
            } else if matches!(name.as_str(), "gpu_sum" | "gpu_min" | "gpu_max" | "gpu_mean" | "gpu_product" | "gpu_variance" | "gpu_stddev" | "norm") {
                Some(1)
            } else if matches!(name.as_str(), "dot") {
                Some(2)
            } else if matches!(name.as_str(), "gpu_save_csv" | "gpu_save_binary") {
                Some(2)
            } else if SCALAR_FNS_1.contains(&name.as_str()) {
                Some(1)
            } else if SCALAR_FNS_2.contains(&name.as_str()) {
                Some(2)
            } else if SCALAR_FNS_3.contains(&name.as_str()) {
                Some(3)
            } else if name == "mem_copy" || name == "file_read_into_mem_u64" || name == "loom_copy" || name == "array_copy" {
                Some(5)
            } else if name == "file_read_into_mem" || name == "mem_u64_add" || name == "gguf_extract_tensor_raw" || name == "rt_staging_load"
                    || name == "decomposed_load_layer" || name == "decomposed_prefetch_layer"
                    || name == "vm_write_register" || name == "vm_dispatch"
                    || name == "vm_dispatch_indirect"
                    || name == "vm_dispatch_mem" || name == "vm_dispatch_indirect_mem"
                    || name == "loom_dispatch" || name == "loom_dispatch_jit" || name == "loom_read"
                    || name == "loom_mail_send" || name == "loom_pool_warm" {
                Some(4)
            } else if matches!(name.as_str(), "http_listen" | "http_accept" | "http_accept_nonblock"
                | "http_method" | "http_path" | "http_query" | "http_body") {
                Some(1)
            } else if name == "http_header" {
                Some(2)
            } else if name == "udp_send_to" {
                Some(4)
            } else if matches!(name.as_str(), "http_respond" | "http_respond_json" | "http_respond_html") {
                Some(3)
            } else if name == "http_respond_with_headers" {
                Some(4)
            } else if name == "http_respond_image" {
                Some(7)
            } else if matches!(name.as_str(), "tcp_connect" | "tcp_send" | "tcp_recv" | "tcp_listen"
                | "tcp_close" | "tcp_accept"
                | "udp_socket" | "udp_recv_from" | "socket_close") {
                // Net builtins — variable arity handled at runtime
                None
            } else if name == "term_image" {
                // 5 args (auto-detect) or 6 args (with mode override)
                if args.len() == 5 || args.len() == 6 { None } else { Some(5) }
            } else if name == "term_image_at" {
                Some(6)
            } else if name == "window_open" || name == "window_draw" {
                Some(3)
            } else if name == "format" || name == "assert" {
                // Variadic: format(template, ...) and assert(cond) or assert(cond, msg)
                None
            // ── GPU array ops (also handled in LetArray context) ──
            } else if matches!(name.as_str(), "gpu_cumsum" | "gpu_reverse"
                | "gpu_load_csv" | "gpu_load_binary" | "normalize") {
                Some(1)
            } else if matches!(name.as_str(), "gpu_fill" | "gpu_add" | "gpu_sub" | "gpu_mul" | "gpu_div"
                | "gpu_scale" | "gpu_pow" | "gpu_ema" | "gpu_concat" | "gpu_gather"
                | "gpu_topk" | "gpu_topk_indices") {
                Some(2)
            } else if matches!(name.as_str(), "gpu_clamp" | "gpu_where" | "gpu_range"
                | "gpu_random" | "gpu_scatter" | "mat_transpose") {
                Some(3)
            } else if name == "gpu_run" {
                // Variadic: gpu_run(kernel, out, ...inputs)
                None
            // ── LLM/GGUF ops ──
            } else if matches!(name.as_str(), "gguf_matvec" | "gguf_load_tensor") {
                // gguf_matvec: 4 args, gguf_load_tensor: 3-4 args
                None
            } else if name == "gguf_infer_layer" {
                Some(6)
            } else if name == "gguf_tokenize" {
                Some(2)
            } else if name == "rt_load_file_to_buffer" {
                Some(3)
            // ── I/O + map ops ──
            } else if matches!(name.as_str(), "write_file" | "write_csv" | "write_bytes" | "append_file"
                | "push" | "repeat" | "vm_present") {
                Some(2)
            } else if name == "map_set" {
                Some(3)
            } else if name == "map_values" || name == "exit" || name == "tcp_close" || name == "tcp_accept" {
                Some(1)
            } else if name == "clock" || name == "map" {
                Some(0)
            // ── aggregate fns (array or stream) ──
            } else if matches!(name.as_str(), "sum" | "min" | "max" | "count") {
                Some(1)
            // ── image I/O ──
            } else if name == "read_image" {
                Some(1)
            } else if name == "write_image" {
                Some(6)
            } else if name.starts_with("ir_") {
                // IR builder stdlib functions (stdlib/compiler/ir.flow) — 92 functions
                // Validated at runtime, not preflight (variable arity, expert API)
                None
            } else if defined_fns.contains(name) {
                // User-defined scalar function — arity checked at call time
                None
            } else {
                let all_fns: Vec<&str> = SCALAR_FNS_0.iter()
                    .chain(SCALAR_FNS_1.iter())
                    .chain(SCALAR_FNS_2.iter())
                    .chain(SCALAR_FNS_3.iter())
                    .chain(EXTRA_KNOWN_FNS.iter())
                    .copied()
                    .collect();
                let suggestion = suggest_str(name, &all_fns);
                sym_errors.push(PreflightError {
                    message: format!("Unknown scalar function '{}'", name),
                    suggestion: suggestion.map_or(String::new(), |s| format!("Did you mean '{}'?", s)),
                    line,
                });
                None
            };
            // Validate arg count
            if let Some(expected) = expected_arity {
                if args.len() != expected {
                    sym_errors.push(PreflightError {
                        message: format!(
                            "{}() requires {} argument{}, got {}",
                            name, expected,
                            if expected == 1 { "" } else { "s" },
                            args.len()
                        ),
                        suggestion: String::new(),
                        line,
                    });
                }
            }
            // Recurse into args — skip first arg for array/map functions (it's an array/map name, not a scalar)
            if matches!(name.as_str(), "len" | "pop" | "map_get" | "map_has" | "map_remove" | "map_keys"
                | "first" | "last" | "min_val" | "max_val" | "join" | "find"
                | "reverse" | "sort_array" | "unique" | "slice" | "json_stringify"
                | "filter" | "map_each" | "sort_by" | "reduce") {
                for arg in args.iter().skip(1) {
                    check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
                }
            } else {
                for arg in args {
                    check_scalar_expr(arg, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
                }
            }
        }
        ScalarExpr::Index { array: _, index } => {
            // Array/map name lives in separate namespace; validate index expression only
            check_scalar_expr(index, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
        }
        ScalarExpr::Lambda { params, body } => {
            // Recurse into body, treating lambda params as locally defined scalars
            let mut local_scalars = defined_scalars.clone();
            for p in params {
                local_scalars.insert(p.clone());
            }
            check_scalar_expr(body, defined_streams, &local_scalars, defined_fns, defined_arrays, sym_errors, line);
        }
        ScalarExpr::ArrayLiteral(elements) => {
            for elem in elements {
                check_scalar_expr(elem, defined_streams, defined_scalars, defined_fns, defined_arrays, sym_errors, line);
            }
        }
    }
}

/// Collect function/struct names from a module file (best-effort, non-blocking).
fn collect_module_fns(
    base_dir: &str,
    module: &str,
    defined_fns: &mut HashSet<String>,
    defined_structs: &mut std::collections::HashMap<String, Vec<String>>,
    defined_scalars: &mut HashSet<String>,
    defined_arrays: &mut HashSet<String>,
    defined_maps: &mut HashSet<String>,
    mutable_scalars: &mut HashSet<String>,
) {
    let local_path = resolve_path(base_dir, &format!("{}.flow", module));
    // Try local first, then stdlib fallback
    let module_path = if std::path::Path::new(&local_path).exists() {
        local_path
    } else {
        resolve_stdlib_path(&format!("{}.flow", module)).unwrap_or(local_path)
    };
    // Circular import guard
    let canonical = std::path::Path::new(&module_path)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(&module_path));
    let canon_str = canonical.to_string_lossy().into_owned();
    let already = PREFLIGHT_IMPORTED.with(|s| s.borrow().contains(&canon_str));
    if already { return; }
    PREFLIGHT_IMPORTED.with(|s| { s.borrow_mut().insert(canon_str); });
    if let Ok(source) = std::fs::read_to_string(&module_path) {
        if let Ok(prog) = octoflow_parser::parse(&source) {
            for (stmt, _span) in &prog.statements {
                match stmt {
                    Statement::FnDecl { name, .. } => {
                        defined_fns.insert(format!("{}.{}", module, name));
                        defined_fns.insert(name.clone());
                    }
                    Statement::ScalarFnDecl { name, .. } => {
                        defined_fns.insert(format!("{}.{}", module, name));
                        defined_fns.insert(name.clone());
                    }
                    Statement::StructDef { name, fields } => {
                        defined_structs.insert(format!("{}.{}", module, name), fields.clone());
                        defined_structs.insert(name.clone(), fields.clone());
                    }
                    Statement::LetDecl { name, value, mutable } => {
                        // Check for try() in module — register decomposed fields
                        if let ScalarExpr::FnCall { name: fn_name, .. } = value {
                            if fn_name == "try" {
                                for field in &["value", "ok", "error"] {
                                    defined_scalars.insert(format!("{}.{}.{}", module, name, field));
                                    defined_scalars.insert(format!("{}.{}", name, field));
                                }
                                continue;
                            }
                            // HTTP client functions in module
                            if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                                for field in &["status", "body", "ok", "error"] {
                                    defined_scalars.insert(format!("{}.{}.{}", module, name, field));
                                    defined_scalars.insert(format!("{}.{}", name, field));
                                }
                                continue;
                            }
                            // Command execution: exec(cmd, ...args) in module
                            if fn_name == "exec" {
                                for field in &["status", "output", "ok", "error"] {
                                    defined_scalars.insert(format!("{}.{}.{}", module, name, field));
                                    defined_scalars.insert(format!("{}.{}", name, field));
                                }
                                continue;
                            }
                            if fn_name == "video_open" {
                                defined_scalars.insert(format!("{}.{}", module, name));
                                defined_scalars.insert(name.clone());
                                for field in &["width", "height", "frames", "fps"] {
                                    defined_scalars.insert(format!("{}.{}.{}", module, name, field));
                                    defined_scalars.insert(format!("{}.{}", name, field));
                                }
                                continue;
                            }
                            if fn_name == "video_frame" {
                                for ch in &["r", "g", "b"] {
                                    defined_arrays.insert(format!("{}.{}.{}", module, name, ch));
                                    defined_arrays.insert(format!("{}.{}", name, ch));
                                }
                                continue;
                            }
                            // JSON/data functions in module
                            if fn_name == "json_parse" || fn_name == "load_data" {
                                defined_maps.insert(format!("{}.{}", module, name));
                                defined_maps.insert(name.clone());
                                continue;
                            }
                            if fn_name == "json_parse_array" || fn_name == "read_csv" {
                                defined_arrays.insert(format!("{}.{}", module, name));
                                defined_arrays.insert(name.clone());
                                continue;
                            }
                            if matches!(fn_name.as_str(), "filter" | "map_each" | "sort_by") {
                                defined_arrays.insert(format!("{}.{}", module, name));
                                defined_arrays.insert(name.clone());
                                continue;
                            }
                        }
                        defined_scalars.insert(format!("{}.{}", module, name));
                        defined_scalars.insert(name.clone());
                        if *mutable {
                            mutable_scalars.insert(name.clone());
                        }
                    }
                    Statement::ArrayDecl { name, mutable, .. } => {
                        defined_arrays.insert(format!("{}.{}", module, name));
                        defined_arrays.insert(name.clone());
                        // Also register as scalar for len() usage detection
                        defined_scalars.insert(format!("{}.{}", module, name));
                        defined_scalars.insert(name.clone());
                        if *mutable {
                            mutable_scalars.insert(name.clone());
                        }
                    }
                    Statement::ExternBlock { functions, .. } => {
                        for f in functions {
                            defined_fns.insert(format!("{}.{}", module, f.name));
                            defined_fns.insert(f.name.clone());
                        }
                    }
                    Statement::UseDecl { module: nested_module } => {
                        // Transitive import: resolve relative to this module's directory
                        let mod_base = std::path::Path::new(&module_path)
                            .parent()
                            .map(|p| p.to_string_lossy().into_owned())
                            .unwrap_or_else(|| base_dir.to_string());
                        collect_module_fns(&mod_base, nested_module, defined_fns, defined_structs,
                                           defined_scalars, defined_arrays, defined_maps, mutable_scalars);
                    }
                    _ => {}
                }
            }
        }
    }
}

fn resolve_path(base_dir: &str, path: &str) -> String {
    if std::path::Path::new(path).is_absolute() {
        path.to_string()
    } else {
        std::path::Path::new(base_dir).join(path).to_string_lossy().into_owned()
    }
}

/// Try to find a module in the stdlib directory relative to the executable.
fn resolve_stdlib_path(relative: &str) -> Option<String> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidates = [
                exe_dir.join("..").join("stdlib").join(relative),
                exe_dir.join("..").join("..").join("stdlib").join(relative),
                exe_dir.join("stdlib").join(relative),
            ];
            for candidate in &candidates {
                if let Ok(cp) = candidate.canonicalize() {
                    return Some(cp.to_string_lossy().into_owned());
                }
            }
        }
    }
    if let Ok(stdlib_dir) = std::env::var("OCTOFLOW_STDLIB") {
        let candidate = std::path::Path::new(&stdlib_dir).join(relative);
        if let Ok(cp) = candidate.canonicalize() {
            return Some(cp.to_string_lossy().into_owned());
        }
    }
    None
}

// ── "Did you mean?" via Levenshtein distance ────────────────────────

fn levenshtein(a: &str, b: &str) -> usize {
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

fn suggest(typo: &str, candidates: &HashSet<String>) -> Option<String> {
    candidates.iter()
        .filter_map(|c| {
            let d = levenshtein(typo, c);
            if d > 0 && d <= 2 { Some((d, c.clone())) } else { None }
        })
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

fn suggest_str(typo: &str, candidates: &[&str]) -> Option<String> {
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
    use octoflow_parser::parse;

    #[test]
    fn test_valid_program_passes() {
        let source = "stream data = tap(\"in.csv\")\nstream r = data |> multiply(2.0)\nemit(r, \"out.csv\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed);
    }

    #[test]
    fn test_unknown_operation() {
        let source = "stream data = tap(\"in.csv\")\nstream r = data |> frobnicate(2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(!report.checks[0].errors.is_empty()); // Operations check
        assert!(report.checks[0].errors[0].message.contains("frobnicate"));
    }

    #[test]
    fn test_undefined_stream() {
        let source = "stream r = prices |> multiply(2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(!report.checks[1].errors.is_empty()); // Symbols check
        assert!(report.checks[1].errors[0].message.contains("prices"));
    }

    #[test]
    fn test_wrong_arg_count() {
        let source = "stream data = tap(\"in.csv\")\nstream r = data |> clamp(1.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(!report.checks[2].errors.is_empty()); // Arguments check
        assert!(report.checks[2].errors[0].message.contains("clamp"));
    }

    #[test]
    fn test_numeric_warnings() {
        let source = "stream data = tap(\"in.csv\")\nstream r = data |> divide(2.0) |> sqrt()\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed); // warnings don't block
        assert!(report.checks[3].warnings.len() >= 2);
    }

    #[test]
    fn test_did_you_mean_suggestion() {
        let source = "stream data = tap(\"in.csv\")\nstream r = data |> negat()\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[0].errors[0].suggestion.contains("negate"));
    }

    #[test]
    fn test_levenshtein_basic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("negate", "negat"), 1);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn test_if_then_else_valid() {
        let source = "stream data = tap(\"in.csv\")\nlet mn = min(data)\nlet flag = if mn > 0.0 then 1.0 else 0.0\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed);
    }

    #[test]
    fn test_if_then_else_undefined_ref() {
        let source = "let flag = if missing > 0.0 then 1.0 else 0.0\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("missing")));
    }

    #[test]
    fn test_boolean_ops_valid() {
        let source = "stream data = tap(\"in.csv\")\nlet a = min(data)\nlet b = max(data)\nlet flag = a > 0.0 && b < 100.0\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed);
    }

    #[test]
    fn test_print_interpolation_valid() {
        let source = "stream data = tap(\"in.csv\")\nlet mn = min(data)\nlet mx = max(data)\nprint(\"range: {mn:.2} to {mx:.2}\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed);
    }

    #[test]
    fn test_print_interpolation_undefined_scalar() {
        let source = "stream data = tap(\"in.csv\")\nprint(\"value = {missing}\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("missing")));
    }

    #[test]
    fn test_error_includes_line_number() {
        let source = "stream data = tap(\"in.csv\")\nstream r = bogus |> multiply(2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        // The undefined stream error should be on line 2
        let err = &report.checks[1].errors[0];
        assert_eq!(err.line, 2, "error should report line 2");
        // Report Display format should include "line 2:"
        let display = format!("{}", report);
        assert!(display.contains("line 2:"), "display should include 'line 2:', got: {}", display);
    }

    // ── Phase 12 preflight tests: scalar functions + count ──────────

    #[test]
    fn test_scalar_fn_valid() {
        let source = "stream data = tap(\"in.csv\")\nlet mn = min(data)\nlet x = abs(mn)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed);
    }

    #[test]
    fn test_scalar_fn_nested_valid() {
        let source = "stream data = tap(\"in.csv\")\nlet mn = min(data)\nlet mx = max(data)\nlet x = sqrt(abs(mn - mx))\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed);
    }

    #[test]
    fn test_scalar_fn_unknown() {
        let source = "stream data = tap(\"in.csv\")\nlet mn = min(data)\nlet x = frobnicate(mn)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("frobnicate")));
    }

    #[test]
    fn test_scalar_fn_wrong_arity() {
        let source = "stream data = tap(\"in.csv\")\nlet mn = min(data)\nlet x = abs(mn, 2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("abs") && e.message.contains("1 argument")));
    }

    #[test]
    fn test_count_reduce_valid() {
        let source = "stream data = tap(\"in.csv\")\nlet n = count(data)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed);
    }

    // ── Phase 14 preflight tests: vec types ────────────────────────────

    #[test]
    fn test_vec3_valid() {
        let source = "let pos = vec3(1.0, 2.0, 3.0)\nlet x = pos.x\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_vec3_wrong_arity() {
        let source = "let pos = vec3(1.0, 2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
    }

    #[test]
    fn test_vec_component_access_valid() {
        let source = "let pos = vec3(1.0, 2.0, 3.0)\nlet dist = sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_vec_undefined_component() {
        // pos.w is not defined for vec3
        let source = "let pos = vec3(1.0, 2.0, 3.0)\nlet w = pos.w\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
    }

    #[test]
    fn test_struct_valid() {
        let source = "struct Entity(x, y, health)\nlet player = Entity(10.0, 20.0, 100.0)\nlet hp = player.health\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed);
    }

    #[test]
    fn test_struct_wrong_arity() {
        let source = "struct Point(x, y, z)\nlet p = Point(1.0, 2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
    }

    #[test]
    fn test_struct_undefined_field() {
        let source = "struct Point(x, y)\nlet p = Point(1.0, 2.0)\nlet z = p.z\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
    }

    // ── Phase 16 preflight tests: arrays ────────────────────────────────

    #[test]
    fn test_array_decl_valid() {
        let source = "let arr = [1.0, 2.0, 3.0]\nlet x = arr[0.0]\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_array_len_valid() {
        let source = "let arr = [1.0, 2.0, 3.0]\nlet n = len(arr)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_array_with_scalar_refs() {
        let source = "let a = 5.0\nlet arr = [a, a + 1.0]\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 17 preflight tests: mutable state ──────────────────────────

    #[test]
    fn test_let_mut_valid() {
        let source = "let mut x = 0.0\nx = x + 1.0\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_assign_to_immutable() {
        let source = "let x = 0.0\nx = x + 1.0\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("not declared as mutable")));
    }

    #[test]
    fn test_assign_to_undefined() {
        let source = "x = 1.0\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("Undefined")));
    }

    // ── Phase 19 preflight tests: while loops ────────────────────────────

    #[test]
    fn test_while_valid() {
        let source = "let mut x = 10.0\nwhile x > 0.0\n  x = x - 1.0\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_while_undefined_condition_ref() {
        let source = "while missing > 0.0\n  let x = 1.0\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("missing")));
    }

    #[test]
    fn test_while_body_assign_to_immutable() {
        let source = "let x = 5.0\nwhile x > 0.0\n  x = x - 1.0\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("not declared as mutable")));
    }

    #[test]
    fn test_while_body_let_and_assign() {
        let source = "let mut total = 0.0\nlet mut i = 5.0\nwhile i > 0.0\n  let step = i * 2.0\n  total = total + step\n  i = i - 1.0\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 20: for-loop preflight ──────────────────────────────

    #[test]
    fn test_for_valid() {
        let source = "let mut total = 0.0\nfor i in range(0, 5)\n  total = total + i\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_for_undefined_range_ref() {
        let source = "for i in range(0, n)\n  let x = i\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("n")));
    }

    #[test]
    fn test_for_body_assign_to_immutable() {
        let source = "let x = 5.0\nfor i in range(0, 3)\n  x = i\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("not declared as mutable")));
    }

    #[test]
    fn test_for_loop_var_in_body() {
        // The loop variable 'i' should be accessible inside the body
        let source = "for i in range(0, 5)\n  let doubled = i * 2.0\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 21: nested loop preflight ──────────────────────────────

    #[test]
    fn test_nested_for_in_for_valid() {
        let source = "let mut total = 0.0\nfor i in range(0, 3)\n  for j in range(0, 3)\n    total = total + i * j\n  end\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_nested_while_in_for_valid() {
        let source = "let mut total = 0.0\nfor i in range(1, 4)\n  let mut c = i\n  while c > 0.0\n    total = total + 1.0\n    c = c - 1.0\n  end\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_nested_for_in_while_valid() {
        let source = "let mut done = 0.0\nlet mut total = 0.0\nwhile done == 0.0\n  for i in range(0, 3)\n    total = total + i\n  end\n  done = 1.0\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_nested_inner_loop_undefined_ref() {
        let source = "for i in range(0, 3)\n  for j in range(0, missing)\n    let x = j\n  end\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("missing")));
    }

    #[test]
    fn test_nested_inner_loop_immutable_assign() {
        let source = "let x = 5.0\nfor i in range(0, 3)\n  while i > 0.0\n    x = 0.0\n  end\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("not declared as mutable")));
    }

    // ── Phase 22: Break / Continue ──────────────────────────────

    #[test]
    fn test_break_in_while_valid() {
        let source = "let mut x = 0.0\nwhile x < 10.0\n  break\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_continue_in_for_valid() {
        let source = "for i in range(0, 5)\n  continue\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_break_top_level_rejected() {
        let source = "break\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("loop")));
    }

    #[test]
    fn test_continue_top_level_rejected() {
        let source = "continue\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("loop")));
    }

    #[test]
    fn test_break_in_nested_loop_valid() {
        let source = "for i in range(0, 3)\n  for j in range(0, 5)\n    break\n  end\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 23: If/Elif/Else statement blocks ──────────────────

    #[test]
    fn test_if_block_valid() {
        let source = "let mut x = 0.0\nif true\n  x = 1.0\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_if_else_block_valid() {
        let source = "let mut x = 0.0\nif true\n  x = 1.0\nelse\n  x = 2.0\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_if_elif_else_valid() {
        let source = "let mut x = 0.0\nlet v = 5.0\nif v > 10.0\n  x = 3.0\nelif v > 3.0\n  x = 2.0\nelse\n  x = 1.0\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_if_block_undefined_ref() {
        let source = "if true\n  let x = missing\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("missing")));
    }

    #[test]
    fn test_if_block_in_loop_valid() {
        let source = "for i in range(0, 5)\n  if i > 2.0\n    break\n  end\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 25: Random ──────────────────────────────────────

    #[test]
    fn test_random_valid() {
        let source = "let r = random()\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_random_wrong_arity() {
        let source = "let r = random(42.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("random") && e.message.contains("0 argument")));
    }

    // ── Phase 27: Module State ──────────────────────────────────

    fn examples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        std::path::Path::new(manifest)
            .join("..").join("examples")
            .canonicalize().unwrap()
            .to_string_lossy().into_owned()
    }

    #[test]
    fn test_preflight_imported_scalar_fn_valid() {
        let source = "use test_scalar_mod\nlet r = triple(5.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, &examples_dir());
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_imported_struct_valid() {
        let source = "use test_struct_mod\nlet c = Color(1.0, 0.5, 0.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, &examples_dir());
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_missing_module_still_errors() {
        let source = "use nonexistent_module\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        assert!(report.checks[1].errors.iter().any(|e| e.message.contains("nonexistent_module")));
    }

    #[test]
    fn test_preflight_imported_constant_valid() {
        let source = "use test_const_mod\nlet r = PI + 1.0\n";
        let program = parse(source).unwrap();
        let report = validate(&program, &examples_dir());
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_imported_all_types_valid() {
        let source = "use test_all_mod\nlet s = SCALE\nlet r = add_one(s)\nlet p = Vec2(1.0, 2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, &examples_dir());
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 28: For-Each Loops — Preflight Tests ──────────────────

    #[test]
    fn test_preflight_foreach_valid() {
        let source = "let arr = [1.0, 2.0, 3.0]\nfor x in arr\n  let y = x\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_foreach_undefined_array() {
        let source = "for x in missing\n  let y = x\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "should fail for undefined array");
        let has_missing = report.checks.iter()
            .flat_map(|c| &c.errors)
            .any(|e| e.message.contains("missing"));
        assert!(has_missing, "should mention 'missing' in errors: {}", report);
    }

    #[test]
    fn test_preflight_foreach_var_visible_in_body() {
        let source = "let arr = [1.0, 2.0]\nfor x in arr\n  let y = x + 1.0\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "loop var should be visible in body: {}", report);
    }

    #[test]
    fn test_preflight_foreach_nested_in_for() {
        let source = "let arr = [1.0, 2.0]\nfor i in range(0, 3)\n  for x in arr\n    let y = x + i\n  end\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 29 preflight tests: array mutation ──────────────────────────

    #[test]
    fn test_preflight_array_assign_valid() {
        let source = "let mut arr = [1.0, 2.0, 3.0]\narr[0] = 99.0\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_array_assign_immutable() {
        let source = "let arr = [1.0, 2.0]\narr[0] = 5.0\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        let has_error = report.checks.iter().flat_map(|c| &c.errors)
            .any(|e| e.message.contains("not declared as mutable"));
        assert!(has_error, "expected mutability error");
    }

    #[test]
    fn test_preflight_push_valid() {
        let source = "let mut arr = [1.0]\npush(arr, 2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_push_immutable() {
        // R-02: push() auto-promotes arrays to mutable — no error expected
        let source = "let arr = [1.0]\npush(arr, 2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "push on non-mut array should auto-promote: {}", report);
    }

    #[test]
    fn test_preflight_array_assign_undefined() {
        let source = "unknown[0] = 5.0\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed);
        let has_error = report.checks.iter().flat_map(|c| &c.errors)
            .any(|e| e.message.contains("Undefined array"));
        assert!(has_error, "expected undefined array error");
    }

    #[test]
    fn test_preflight_array_mutation_in_loop() {
        let source = "let mut arr = []\nfor i in range(0, 3)\n  push(arr, i)\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 30a preflight tests: array params ──────────────────────────

    #[test]
    fn test_preflight_array_param_valid() {
        // Passing an array to a user-defined function should not error
        let source = "fn my_sum(arr)\n  let mut total = 0.0\n  for x in arr\n    total = total + x\n  end\n  return total\nend\nlet data = [1.0, 2.0, 3.0]\nlet s = my_sum(data)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_array_and_scalar_param_valid() {
        // Function with both array and scalar params
        let source = "fn arr_contains(arr, val)\n  for x in arr\n    if x == val\n      return 1.0\n    end\n  end\n  return 0.0\nend\nlet data = [1.0, 2.0]\nlet r = arr_contains(data, 2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 30b: HashMap preflight tests ──────────────────────────

    #[test]
    fn test_preflight_map_basic() {
        let source = "let mut m = map()\nmap_set(m, \"k\", 1.0)\nlet v = map_get(m, \"k\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_map_has_and_keys() {
        let source = "let mut m = map()\nmap_set(m, \"a\", 1.0)\nlet e = map_has(m, \"a\")\nlet k = map_keys(m)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_map_remove() {
        let source = "let mut m = map()\nmap_set(m, \"x\", 1.0)\nlet v = map_remove(m, \"x\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_map_len() {
        let source = "let mut m = map()\nmap_set(m, \"a\", 1.0)\nlet n = len(m)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_map_in_loop() {
        let source = "let mut m = map()\nfor i in range(0, 3)\n  map_set(m, \"key\", i)\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_map_in_if_block() {
        let source = "let mut m = map()\nif 1.0 > 0.0\n  map_set(m, \"yes\", 1.0)\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 31: File I/O preflight tests ───────────────────────

    #[test]
    fn test_preflight_file_io_functions() {
        let source = "let e = file_exists(\"test.txt\")\nlet s = file_size(\"test.txt\")\nlet d = is_directory(\".\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_path_utilities() {
        let source = "let ext = file_ext(\"data.csv\")\nlet name = file_name(\"/path/data.csv\")\nlet dir = file_dir(\"/path/data.csv\")\nlet p = path_join(\"dir\", \"file\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_write_file() {
        let source = "write_file(\"out.txt\", \"data\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_append_file() {
        let source = "append_file(\"log.txt\", \"entry\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_write_file_in_loop() {
        let source = "for i in range(0, 3)\n  write_file(\"out.txt\", \"data\")\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_write_file_in_if() {
        let source = "if 1.0 > 0.0\n  write_file(\"out.txt\", \"data\")\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_split_function() {
        let source = "let parts = split(\"a,b\", \",\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_read_lines() {
        let source = "let lines = read_lines(\"test.txt\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_list_dir() {
        let source = "let files = list_dir(\".\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    // ── Phase 32: String Operations + Type Conversion ──────────

    #[test]
    fn test_preflight_type_conversion() {
        let source = "let x = 3.14\nlet s = str(x)\nlet f = float(s)\nlet i = int(f)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_string_ops_1arg() {
        let source = "let a = trim(\"  hi  \")\nlet b = to_upper(\"hi\")\nlet c = to_lower(\"HI\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_string_ops_2arg() {
        let source = "let a = starts_with(\"hi\", \"h\")\nlet b = ends_with(\"hi\", \"i\")\nlet c = index_of(\"hi\", \"i\")\nlet d = char_at(\"hi\", 0)\nlet e = repeat(\"hi\", 3)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_string_ops_3arg() {
        let source = "let a = substr(\"hello\", 1, 3)\nlet b = replace(\"hi\", \"h\", \"b\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_str_wrong_arity() {
        let source = "let s = str(1, 2)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "should fail: str() takes 1 arg, got 2");
    }

    #[test]
    fn test_preflight_substr_wrong_arity() {
        let source = "let s = substr(\"hi\", 0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "should fail: substr() takes 3 args, got 2");
    }

    // ── Phase 33: Array Operations ─────────────────────────────

    #[test]
    fn test_preflight_array_ops_1arg() {
        let source = "let arr = [1.0, 2.0]\nlet f = first(arr)\nlet l = last(arr)\nlet m = min_val(arr)\nlet x = max_val(arr)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_array_ops_2arg() {
        let source = "let arr = [1.0, 2.0]\nlet s = join(arr, \",\")\nlet idx = find(arr, 1.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_array_ops_3arg() {
        let source = "let arr = [1.0, 2.0, 3.0]\nlet sub = slice(arr, 0, 2)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_array_returning() {
        let source = "let arr = [3.0, 1.0, 2.0]\nlet rev = reverse(arr)\nlet sorted = sort_array(arr)\nlet u = unique(arr)\nlet r = range_array(0, 5)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_type_of() {
        let source = "let t = type_of(3.14)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "Report: {}", report);
    }

    #[test]
    fn test_preflight_is_nan() {
        let source = "let x = 0.0 / 0.0\nlet r = is_nan(x)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "is_nan should pass preflight: {}", report);
    }

    #[test]
    fn test_preflight_is_inf() {
        let source = "let x = 1.0 / 0.0\nlet r = is_inf(x)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "is_inf should pass preflight: {}", report);
    }

    #[test]
    fn test_preflight_join_wrong_arity() {
        let source = "let arr = [1.0]\nlet s = join(arr)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "should fail: join() takes 2 args");
    }

    // ── Phase 34: try() preflight tests ─────────────────────────────

    #[test]
    fn test_preflight_try_valid() {
        let source = "let r = try(42.0)\nlet x = r.ok\nlet y = r.value\nlet z = r.error\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "try() with field access should pass preflight");
    }

    #[test]
    fn test_preflight_try_fields_defined() {
        // Fields .value, .ok, .error should be recognized
        let source = "let r = try(42.0)\nprint(\"{r.value} {r.ok} {r.error}\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "try() fields should be defined");
    }

    #[test]
    fn test_preflight_try_wrong_arity() {
        let source = "let r = try(1.0, 2.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "try(a, b) should fail arity check");
    }

    #[test]
    fn test_preflight_try_in_loop() {
        let source = "let arr = [1.0]\nfor x in arr\nlet r = try(x)\nlet v = r.ok\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "try() in loop should pass preflight");
    }

    // ── Phase 35: HTTP Client preflight ──

    #[test]
    fn test_preflight_http_get_valid() {
        let source = "let r = http_get(\"https://example.com\")\nlet s = r.status\nlet b = r.body\nlet o = r.ok\nlet e = r.error\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "http_get should pass preflight", );
    }

    #[test]
    fn test_preflight_http_post_valid() {
        let source = "let r = http_post(\"https://example.com\", \"body\")\nlet s = r.status\nlet o = r.ok\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "http_post should pass preflight", );
    }

    #[test]
    fn test_preflight_http_get_wrong_arity() {
        let source = "let r = http_get(\"a\", \"b\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "http_get with 2 args should fail preflight");
    }

    #[test]
    fn test_preflight_http_post_wrong_arity() {
        let source = "let r = http_post(\"a\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "http_post with 1 arg should fail preflight");
    }

    #[test]
    fn test_preflight_http_fields_defined() {
        let source = "let r = http_get(\"url\")\nlet check1 = r.status\nlet check2 = r.body\nlet check3 = r.ok\nlet check4 = r.error\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "http_get fields should be defined", );
    }

    #[test]
    fn test_preflight_http_in_loop() {
        let source = "for i in range(0, 1)\nlet r = http_get(\"url\")\nlet s = r.status\nend\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "http_get in loop should pass", );
    }

    #[test]
    fn test_preflight_http_delete_valid() {
        let source = "let r = http_delete(\"https://example.com\")\nlet s = r.status\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "http_delete should pass", );
    }

    #[test]
    fn test_preflight_http_put_valid() {
        let source = "let r = http_put(\"https://example.com\", \"data\")\nlet s = r.status\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "http_put should pass", );
    }

    // ── Phase 36: JSON I/O preflight tests ────────────────────────

    #[test]
    fn test_preflight_json_parse_valid() {
        let source = "let s = \"{}\"\nlet mut data = json_parse(s)\nlet k = map_get(data, \"x\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "json_parse should pass preflight");
    }

    #[test]
    fn test_preflight_json_parse_wrong_arity() {
        let source = "let data = json_parse(\"a\", \"b\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "json_parse with 2 args should fail");
    }

    #[test]
    fn test_preflight_json_parse_array_valid() {
        let source = "let s = \"[1]\"\nlet arr = json_parse_array(s)\nlet v = arr[0]\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "json_parse_array should pass preflight");
    }

    #[test]
    fn test_preflight_json_stringify_valid() {
        let source = "let mut m = map()\nlet s = json_stringify(m)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "json_stringify should pass preflight");
    }

    // ── Phase 37: Environment & OctoData preflight tests ────────────

    #[test]
    fn test_preflight_load_data_valid() {
        let source = "let mut config = load_data(\"settings.od\")\nlet v = map_get(config, \"key\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "load_data should pass preflight");
    }

    #[test]
    fn test_preflight_load_data_wrong_arity() {
        let source = "let mut data = load_data(\"a\", \"b\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "load_data with 2 args should fail");
    }

    #[test]
    fn test_preflight_save_data_valid() {
        let source = "let mut m = map()\nsave_data(\"out.od\", m)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "save_data should pass preflight");
    }

    #[test]
    fn test_preflight_time_env_os_valid() {
        let source = "let t = time()\nlet e = env(\"PATH\")\nlet o = os_name()\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "time/env/os_name should pass preflight");
    }

    #[test]
    fn test_preflight_filter_valid() {
        let source = "let nums = [1.0, 2.0, 3.0]\nlet big = filter(nums, fn(x) x > 1.0 end)\nlet n = len(big)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "filter with lambda should pass preflight");
    }

    #[test]
    fn test_preflight_reduce_valid() {
        let source = "let nums = [1.0, 2.0]\nlet total = reduce(nums, 0.0, fn(acc, x) acc + x end)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "reduce with lambda should pass preflight");
    }

    #[test]
    fn test_preflight_filter_wrong_arity() {
        let source = "let nums = [1.0]\nlet bad = filter(nums, fn(x) x end, 999.0)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "filter with 3 args should fail preflight");
    }

    #[test]
    fn test_preflight_lambda_undefined_ref() {
        let source = "let nums = [1.0]\nlet r = filter(nums, fn(x) x > undefined_var end)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "lambda referencing undefined var should fail");
    }

    #[test]
    fn test_preflight_hashmap_bracket() {
        let source = "let mut m = map()\nmap_set(m, \"k\", 1.0)\nlet v = m[\"k\"]\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "hashmap bracket access should pass preflight");
    }

    // ── Phase 39: read_csv + write_csv ────────────────────────────────

    #[test]
    fn test_preflight_read_csv_valid() {
        let source = "let data = read_csv(\"test.csv\")\nlet n = len(data)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "read_csv should pass preflight: {}", report);
    }

    #[test]
    fn test_preflight_read_csv_wrong_arity() {
        let source = "let data = read_csv(\"a.csv\", \"b.csv\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "read_csv with 2 args should fail");
    }

    #[test]
    fn test_preflight_gpu_matmul_valid() {
        let source = "let a = gpu_range(1.0, 7.0, 1.0)\nlet b = gpu_range(7.0, 13.0, 1.0)\nlet c = gpu_matmul(a, b, 2, 2, 3)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "gpu_matmul with 5 args should pass: {}", report);
    }

    #[test]
    fn test_preflight_gpu_matmul_wrong_arity() {
        let source = "let a = gpu_range(1.0, 7.0, 1.0)\nlet b = gpu_range(7.0, 13.0, 1.0)\nlet c = gpu_matmul(a, b, 2)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "gpu_matmul with 3 args should fail");
        let msg = format!("{}", report);
        assert!(msg.contains("gpu_matmul() requires 5 arguments"), "expected arity message, got: {}", msg);
    }

    #[test]
    fn test_preflight_write_csv_valid() {
        let source = "let data = read_csv(\"in.csv\")\nwrite_csv(\"out.csv\", data)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "write_csv should pass preflight: {}", report);
    }

    #[test]
    fn test_preflight_write_csv_undefined_array() {
        let source = "write_csv(\"out.csv\", nonexistent)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "write_csv with undefined array should fail");
    }

    #[test]
    fn test_preflight_read_csv_as_array() {
        // read_csv result should be usable as array (e.g., with len, filter)
        let source = "let data = read_csv(\"test.csv\")\nlet high = filter(data, fn(r) r[\"score\"] >= 90.0 end)\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "read_csv result should be treated as array: {}", report);
    }

    #[test]
    fn test_exec_no_args_preflight() {
        let source = "let r = exec()\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(!report.passed, "exec() with no args should fail preflight");
    }

    #[test]
    fn test_exec_variadic_preflight() {
        let source = "let r = exec(\"echo\", \"a\", \"b\", \"c\", \"d\")\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "exec() with many args should pass preflight");
    }

    #[test]
    fn test_preflight_ir_functions() {
        // IR builder functions (stdlib/compiler/ir.flow) should pass preflight via ir_ prefix
        let source = r#"let h = ir_new()
let b = ir_block("entry")
let gid = ir_load_gid()
let r1 = ir_const_f32(1.0)
let r2 = ir_fadd(gid, r1)
let ok = ir_emit_spirv("test.spv")
"#;
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "ir_* functions should pass preflight: {}", report);
    }

    #[test]
    fn test_preflight_ir_variables() {
        // IR builder module-level variables should pass preflight via ir_ prefix
        let source = "let x = ir_input_count\nlet y = ir_shared_size\n";
        let program = parse(source).unwrap();
        let report = validate(&program, ".");
        assert!(report.passed, "ir_* variables should pass preflight: {}", report);
    }
}
