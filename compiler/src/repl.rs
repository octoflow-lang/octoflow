//! Interactive REPL for OctoFlow — `octoflow repl`.

use crate::compiler::{Context, StmtResult, format_repl_value};
use crate::CliError;
use crate::Value;

use std::sync::atomic::{AtomicBool, Ordering};

static CTRL_C_PRESSED: AtomicBool = AtomicBool::new(false);

/// Set up Ctrl+C handler for graceful REPL exit.
fn ctrlc_setup() -> Result<(), ()> {
    #[cfg(target_os = "windows")]
    {
        extern "system" {
            fn SetConsoleCtrlHandler(
                handler: extern "system" fn(u32) -> i32,
                add: i32,
            ) -> i32;
        }
        extern "system" fn handler(_ctrl_type: u32) -> i32 {
            CTRL_C_PRESSED.store(true, Ordering::SeqCst);
            1 // handled
        }
        unsafe { SetConsoleCtrlHandler(handler, 1); }
    }
    #[cfg(not(target_os = "windows"))]
    {
        extern "C" {
            fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
        }
        extern "C" fn handler(_sig: i32) {
            CTRL_C_PRESSED.store(true, Ordering::SeqCst);
        }
        unsafe { signal(2, handler); } // SIGINT = 2
    }
    Ok(())
}

/// Load REPL history from ~/.octoflow/repl_history.
fn load_history() -> Vec<String> {
    if let Some(home) = home_dir() {
        let path = home.join(".octoflow").join("repl_history");
        if let Ok(content) = std::fs::read_to_string(&path) {
            return content.lines()
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    Vec::new()
}

/// Save REPL history to ~/.octoflow/repl_history (max 500 entries).
fn save_history(history: &[String]) {
    if let Some(home) = home_dir() {
        let dir = home.join(".octoflow");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("repl_history");
        let start = if history.len() > 500 { history.len() - 500 } else { 0 };
        let content: String = history[start..].iter()
            .map(|s| format!("{}\n", s))
            .collect();
        let _ = std::fs::write(&path, content);
    }
}

/// Get the user's home directory.
fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var("HOME").ok()
        .or_else(|| std::env::var("USERPROFILE").ok())
        .map(std::path::PathBuf::from)
}

/// Run the interactive REPL loop.
pub fn run_repl() -> Result<(), CliError> {
    let mut ctx = Context::new();

    // Banner
    eprintln!("OctoFlow v{} — GPU-native language", crate::VERSION);
    if let Some(name) = ctx.gpu_name() {
        eprintln!("GPU: {}", name);
    } else {
        eprintln!("GPU: none (CPU fallback)");
    }
    let stdlib_count = count_stdlib_modules();
    eprintln!("{} stdlib modules | :help | :time | :history", stdlib_count);
    eprintln!();

    // Ctrl+C handler — graceful exit
    let _ = ctrlc_setup();

    // Load persistent history
    let mut history: Vec<String> = load_history();

    let stdin = std::io::stdin();
    let mut buffer = String::new();
    let mut continuation = false;
    let mut last_value: Option<Value> = None;

    loop {
        // Print prompt
        if continuation {
            eprint!("... ");
        } else {
            eprint!("> ");
        }

        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => { save_history(&history); break; } // EOF (Ctrl+D)
            Ok(_) => {}
            Err(e) => {
                eprintln!("Read error: {}", e);
                break;
            }
        }

        // Ctrl+C handler
        if CTRL_C_PRESSED.load(Ordering::SeqCst) {
            save_history(&history);
            break;
        }

        // Trim trailing newline but keep content
        let line = line.trim_end_matches('\n').trim_end_matches('\r');

        if continuation {
            buffer.push('\n');
            buffer.push_str(line);
        } else {
            buffer = line.to_string();
        }

        // Check multi-line continuation
        if needs_continuation(&buffer) {
            continuation = true;
            continue;
        }
        continuation = false;

        let input = buffer.trim().to_string();
        if input.is_empty() {
            continue;
        }

        // Record in history (skip duplicates of last entry)
        if history.last().map_or(true, |last| last != &input) {
            history.push(input.clone());
        }

        // Check exit — accept all reasonable variants
        if input == "exit" || input == "quit"
            || input == ":exit" || input == ":quit" || input == ":q"
        {
            save_history(&history);
            break;
        }

        // Check special commands
        if input.starts_with(':') {
            handle_special(&input, &mut ctx, &last_value, &history);
            continue;
        }

        // Bind `_` to last expression value
        if let Some(ref val) = last_value {
            ctx.set_underscore(val.clone());
        }

        // Try parsing as a full program (statements)
        match octoflow_parser::parse(&input) {
            Ok(program) => {
                let mut had_error = false;
                for (stmt, span) in &program.statements {
                    match ctx.eval_statement(stmt, span) {
                        Ok(StmtResult::Silent) => {}
                        Ok(StmtResult::FnDefined(name)) => {
                            eprintln!("{} defined", name);
                        }
                        Ok(StmtResult::Printed) => {}
                        Ok(StmtResult::StreamCreated(name, count)) => {
                            eprintln!("stream {} ({} elements)", name, count);
                        }
                        Ok(StmtResult::Value(val)) => {
                            println!("{}", format_repl_value(&val));
                            last_value = Some(val);
                        }
                        Err(e) => {
                            eprintln!("error: {}", e);
                            had_error = true;
                            break;
                        }
                    }
                }
                if !had_error {
                    continue;
                }
            }
            Err(_) => {
                // Statement parse failed — try as bare expression
            }
        }

        // Try parsing as a bare expression
        match octoflow_parser::parse_expr(&input) {
            Ok(expr) => {
                match ctx.eval_expression(&expr) {
                    Ok(val) => {
                        println!("{}", format_repl_value(&val));
                        last_value = Some(val);
                    }
                    Err(e) => eprintln!("error: {}", e),
                }
            }
            Err(e) => {
                eprintln!("error: {}", e);
            }
        }
    }

    Ok(())
}

/// Check if the input needs more lines (unbalanced blocks, brackets, or strings).
pub fn needs_continuation(input: &str) -> bool {
    // Count block openers vs `end`
    let mut block_depth: i32 = 0;
    let mut bracket_depth: i32 = 0;
    let mut in_string = false;

    for line in input.lines() {
        let trimmed = line.trim();
        for ch in trimmed.chars() {
            if ch == '"' {
                in_string = !in_string;
            }
            if !in_string {
                if ch == '[' { bracket_depth += 1; }
                if ch == ']' { bracket_depth -= 1; }
            }
        }
        if !in_string {
            // Count block openers: fn, if, while, for (at start of line)
            let first_word = trimmed.split_whitespace().next().unwrap_or("");
            match first_word {
                "fn" | "if" | "while" | "for" => block_depth += 1,
                "end" => block_depth -= 1,
                // elif does NOT add a block level — it continues an existing if block
                _ => {}
            }
        }
    }

    // Unclosed string
    if in_string {
        return true;
    }
    // Unclosed brackets
    if bracket_depth > 0 {
        return true;
    }
    // Unmatched blocks
    if block_depth > 0 {
        return true;
    }

    false
}

/// Handle special REPL commands (`:help`, `:vars`, etc.).
fn handle_special(line: &str, ctx: &mut Context, last_value: &Option<Value>, history: &[String]) {
    let parts: Vec<&str> = line.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        ":help" | ":h" => {
            eprintln!("OctoFlow REPL Commands:");
            eprintln!("  :help          Show this help");
            eprintln!("  :vars          List all variables");
            eprintln!("  :fns           List all functions");
            eprintln!("  :streams       List all streams");
            eprintln!("  :arrays        List GPU arrays");
            eprintln!("  :type <name>   Show type of a variable");
            eprintln!("  :gpu           Show GPU device info");
            eprintln!("  :time          GPU benchmark (no args = smoke test)");
            eprintln!("  :time <expr>   Time any expression:  :time gpu_fill(1.0, 1000000)");
            eprintln!("  :reset         Clear all state");
            eprintln!("  :load <file>   Load and execute a .flow file");
            eprintln!("  :history       Show last 20 inputs");
            eprintln!("  :clear         Clear screen");
            eprintln!("  :q / :quit     Exit the REPL");
            eprintln!();
            eprintln!("Expressions are auto-printed. Use _ for last result:");
            eprintln!("  > 2 + 2");
            eprintln!("  4");
            eprintln!("  > _ * 3");
            eprintln!("  12");
        }
        ":vars" | ":v" => {
            let vars = ctx.list_variables();
            if vars.is_empty() {
                eprintln!("(no variables defined)");
            } else {
                for (name, ty, val) in &vars {
                    if name == "_" { continue; }
                    eprintln!("  {} : {} = {}", name, ty, val);
                }
            }
        }
        ":fns" | ":f" => {
            let fns = ctx.list_functions();
            if fns.is_empty() {
                eprintln!("(no functions defined)");
            } else {
                for (name, params) in &fns {
                    eprintln!("  fn {}({})", name, params);
                }
            }
        }
        ":streams" | ":s" => {
            let streams = ctx.list_streams();
            if streams.is_empty() {
                eprintln!("(no streams)");
            } else {
                for (name, len) in &streams {
                    eprintln!("  {} [{} elements]", name, len);
                }
            }
        }
        ":arrays" | ":a" => {
            let arrays = ctx.list_gpu_arrays();
            if arrays.is_empty() {
                eprintln!("(no GPU arrays)");
            } else {
                let total_elements: usize = arrays.iter().map(|(_, len, _)| *len).sum();
                for (name, len, loc) in &arrays {
                    let mb = (*len as f64) * 4.0 / (1024.0 * 1024.0);
                    eprintln!("  {} [{} elements, {:.1} MB, {}]", name, len, mb, loc);
                }
                let total_mb = (total_elements as f64) * 4.0 / (1024.0 * 1024.0);
                eprintln!("  total: {} arrays, {} elements, {:.1} MB", arrays.len(), total_elements, total_mb);
            }
        }
        ":type" | ":t" => {
            if arg.is_empty() {
                eprintln!("Usage: :type <name>");
            } else {
                let vars = ctx.list_variables();
                if let Some((_, ty, val)) = vars.iter().find(|(n, _, _)| n == arg) {
                    eprintln!("{} : {} = {}", arg, ty, val);
                } else {
                    eprintln!("'{}' not found", arg);
                }
            }
        }
        ":gpu" => {
            if let Some(name) = ctx.gpu_name() {
                eprintln!("GPU: {}", name);
                let arrays = ctx.list_gpu_arrays();
                let resident = arrays.iter().filter(|(_, _, loc)| *loc == "gpu").count();
                let cpu = arrays.iter().filter(|(_, _, loc)| *loc == "cpu").count();
                eprintln!("  arrays: {} total ({} gpu-resident, {} cpu)", arrays.len(), resident, cpu);
            } else {
                eprintln!("GPU: not available (CPU fallback mode)");
            }
        }
        ":time" => {
            if arg.is_empty() {
                // Quick GPU smoke test benchmark
                run_quick_benchmark(ctx);
                return;
            } else {
                let start = std::time::Instant::now();

                // Try as statement first
                match octoflow_parser::parse(arg) {
                    Ok(program) => {
                        let mut had_result = false;
                        for (stmt, span) in &program.statements {
                            match ctx.eval_statement(stmt, span) {
                                Ok(StmtResult::Value(val)) => {
                                    println!("{}", format_repl_value(&val));
                                    had_result = true;
                                }
                                Ok(StmtResult::Printed) => { had_result = true; }
                                Ok(_) => { had_result = true; }
                                Err(e) => {
                                    eprintln!("error: {}", e);
                                    return;
                                }
                            }
                        }
                        if had_result {
                            let elapsed = start.elapsed();
                            eprintln!("  [{:.3} ms]", elapsed.as_secs_f64() * 1000.0);
                            return;
                        }
                    }
                    Err(_) => {}
                }

                // Fallback: try as expression
                match octoflow_parser::parse_expr(arg) {
                    Ok(expr) => {
                        match ctx.eval_expression(&expr) {
                            Ok(val) => {
                                let elapsed = start.elapsed();
                                println!("{}", format_repl_value(&val));
                                eprintln!("  [{:.3} ms]", elapsed.as_secs_f64() * 1000.0);
                            }
                            Err(e) => eprintln!("error: {}", e),
                        }
                    }
                    Err(e) => eprintln!("error: {}", e),
                }
            }
        }
        ":reset" => {
            ctx.reset();
            eprintln!("State cleared.");
        }
        ":load" | ":l" => {
            if arg.is_empty() {
                eprintln!("Usage: :load <file.flow>");
            } else {
                let start = std::time::Instant::now();
                match ctx.load_file(arg) {
                    Ok(()) => {
                        let elapsed = start.elapsed();
                        eprintln!("Loaded {} [{:.1} ms]", arg, elapsed.as_secs_f64() * 1000.0);
                    }
                    Err(e) => eprintln!("error: {}", e),
                }
            }
        }
        ":clear" | ":cls" => {
            // ANSI clear screen
            eprint!("\x1b[2J\x1b[H");
        }
        ":history" | ":hist" => {
            let start = if history.len() > 20 { history.len() - 20 } else { 0 };
            if history[start..].is_empty() {
                eprintln!("(no history)");
            } else {
                for (i, entry) in history[start..].iter().enumerate() {
                    eprintln!("  {:3}  {}", start + i + 1, entry);
                }
            }
        }
        ":_" => {
            // Show last result
            if let Some(ref val) = last_value {
                println!("{}", format_repl_value(val));
            } else {
                eprintln!("(no previous result)");
            }
        }
        _ => {
            eprintln!("Unknown command: {}. Type :help for commands.", cmd);
        }
    }
}

/// Count stdlib .flow modules in the stdlib directory.
fn count_stdlib_modules() -> usize {
    let stdlib_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..").join("stdlib");
    if !stdlib_path.exists() {
        return 0;
    }
    count_flow_files(&stdlib_path)
}

fn count_flow_files(dir: &std::path::Path) -> usize {
    let mut count = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip test directories
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name == "tests" || name == "test_data" || name == "examples" {
                    continue;
                }
                count += count_flow_files(&path);
            } else if path.extension().map_or(false, |e| e == "flow") {
                count += 1;
            }
        }
    }
    count
}

/// Quick GPU smoke test — runs when user types `:time` with no args.
fn run_quick_benchmark(ctx: &mut Context) {
    if !ctx.has_gpu() {
        eprintln!("No GPU detected. Running CPU-only.");
        eprintln!("Use :time <expr> to benchmark anything.");
        return;
    }

    let gpu_name = ctx.gpu_name().unwrap_or_else(|| "Unknown GPU".to_string());
    eprintln!("GPU benchmark ({}):", gpu_name);

    let tests: &[(&str, &str)] = &[
        ("gpu_fill  1M", "let _bq = gpu_fill(1.0, 1000000.0)"),
        ("gpu_add   1M", "let _bq2 = gpu_add(_bq, _bq)"),
        ("gpu_sum   1M", "let _bq3 = gpu_sum(_bq)"),
    ];

    for (label, code) in tests {
        match octoflow_parser::parse(code) {
            Ok(program) => {
                let start = std::time::Instant::now();
                let mut ok = true;
                for (stmt, span) in &program.statements {
                    if let Err(e) = ctx.eval_statement(stmt, span) {
                        eprintln!("  {} .... error: {}", label, e);
                        ok = false;
                        break;
                    }
                }
                if ok {
                    let elapsed = start.elapsed();
                    eprintln!("  {} .... {:.1}ms", label, elapsed.as_secs_f64() * 1000.0);
                }
            }
            Err(e) => eprintln!("  {} .... parse error: {}", label, e),
        }
    }

    eprintln!("GPU is ready. Use :time <expr> to benchmark anything.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;
    use octoflow_parser::ast::Span;

    #[allow(dead_code)]
    fn span() -> Span {
        Span { line: 1, col: 1 }
    }

    #[test]
    fn test_context_arithmetic() {
        let mut ctx = Context::new();
        let expr = octoflow_parser::parse_expr("2 + 2").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 4.0);
    }

    #[test]
    fn test_context_variable_persist() {
        let mut ctx = Context::new();
        let program = octoflow_parser::parse("let x = 10").unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("x + 1").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 11.0);
    }

    #[test]
    fn test_context_string_ops() {
        let mut ctx = Context::new();
        let program = octoflow_parser::parse("let s = \"hello\"").unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        // len
        let expr = octoflow_parser::parse_expr("len(s)").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 5.0);

        // contains
        let expr2 = octoflow_parser::parse_expr("contains(s, \"ell\")").unwrap();
        let val2 = ctx.eval_expression(&expr2).unwrap();
        assert_eq!(val2.as_float().unwrap(), 1.0);
    }

    #[test]
    fn test_context_multiline_fn() {
        let mut ctx = Context::new();
        let src = "fn double(n)\n  return n * 2.0\nend";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("double(5)").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 10.0);
    }

    #[test]
    fn test_context_struct() {
        let mut ctx = Context::new();
        let src = "struct Point(x, y)\nlet p = Point(3.0, 4.0)";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("p.x + p.y").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 7.0);
    }

    #[test]
    fn test_context_array() {
        let mut ctx = Context::new();
        let program = octoflow_parser::parse("let arr = [10.0, 20.0, 30.0]").unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("arr[1]").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 20.0);
    }

    #[test]
    fn test_context_vec() {
        let mut ctx = Context::new();
        let program = octoflow_parser::parse("let v = vec3(1.0, 2.0, 3.0)").unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("v.y").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 2.0);
    }

    #[test]
    fn test_context_reset() {
        let mut ctx = Context::new();
        let program = octoflow_parser::parse("let x = 42").unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        // x should be defined
        let expr = octoflow_parser::parse_expr("x").unwrap();
        assert!(ctx.eval_expression(&expr).is_ok());

        ctx.reset();

        // x should no longer exist
        let expr2 = octoflow_parser::parse_expr("x").unwrap();
        assert!(ctx.eval_expression(&expr2).is_err());
    }

    #[test]
    fn test_needs_continuation_fn() {
        assert!(needs_continuation("fn foo()"));
    }

    #[test]
    fn test_needs_continuation_complete() {
        assert!(!needs_continuation("let x = 5"));
    }

    #[test]
    fn test_needs_continuation_bracket() {
        assert!(needs_continuation("[1, 2"));
    }

    #[test]
    fn test_needs_continuation_if_block() {
        assert!(needs_continuation("if x > 0"));
        assert!(!needs_continuation("if x > 0\n  print(1)\nend"));
    }

    #[test]
    fn test_needs_continuation_elif_no_extra_depth() {
        // elif should NOT add block depth; this is a complete if/elif/end
        let src = "if x > 0\n  print(1)\nelif x < 0\n  print(2)\nend";
        assert!(!needs_continuation(src));
    }

    #[test]
    fn test_needs_continuation_while_block() {
        assert!(needs_continuation("while x > 0"));
        assert!(!needs_continuation("while x > 0\n  x = x - 1\nend"));
    }

    #[test]
    fn test_needs_continuation_nested_blocks() {
        assert!(needs_continuation("fn foo()\n  if x > 0"));
        assert!(!needs_continuation("fn foo()\n  if x > 0\n    return 1\n  end\nend"));
    }

    #[test]
    fn test_format_repl_float() {
        assert_eq!(format_repl_value(&Value::Float(4.0)), "4");
        assert_eq!(format_repl_value(&Value::Float(3.14)), "3.14");
        assert_eq!(format_repl_value(&Value::Float(-2.0)), "-2");
    }

    #[test]
    fn test_format_repl_string() {
        assert_eq!(format_repl_value(&Value::Str("hi".into())), "\"hi\"");
    }

    // ── Underscore last-result binding ─────────────────

    #[test]
    fn test_underscore_last_result() {
        let mut ctx = Context::new();
        // Set _ to 42
        ctx.set_underscore(Value::Float(42.0));
        let expr = octoflow_parser::parse_expr("_ + 1").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 43.0);
    }

    #[test]
    fn test_underscore_string_last_result() {
        let mut ctx = Context::new();
        ctx.set_underscore(Value::Str("hello".into()));
        let expr = octoflow_parser::parse_expr("len(_)").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 5.0);
    }

    // ── Context state queries ────────────────────────

    #[test]
    fn test_list_streams_empty() {
        let ctx = Context::new();
        assert!(ctx.list_streams().is_empty());
    }

    #[test]
    fn test_list_gpu_arrays_empty() {
        let ctx = Context::new();
        assert!(ctx.list_gpu_arrays().is_empty());
    }

    #[test]
    fn test_state_counts() {
        let mut ctx = Context::new();
        let (s, a, h, st, f) = ctx.state_counts();
        assert_eq!(s, 0);
        assert_eq!(a, 0);
        assert_eq!(h, 0);
        assert_eq!(st, 0);
        assert_eq!(f, 0);

        // Add a variable
        let program = octoflow_parser::parse("let x = 5").unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let (s2, _, _, _, _) = ctx.state_counts();
        assert_eq!(s2, 1);
    }

    // ── Count stdlib modules ────────────────────────

    #[test]
    fn test_count_stdlib_modules() {
        // Just verify it returns a positive number (we have 75+ modules)
        let count = count_stdlib_modules();
        assert!(count > 50, "Expected 50+ stdlib modules, got {}", count);
    }

    // ── Phase 27: Module State — REPL Context Tests ────────────

    fn examples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        std::path::Path::new(manifest)
            .join("..").join("examples")
            .canonicalize().unwrap()
            .to_string_lossy().into_owned()
    }

    #[test]
    fn test_context_use_scalar_fn() {
        let mut ctx = Context::new();
        ctx.set_base_dir(&examples_dir());
        let program = octoflow_parser::parse("use test_scalar_mod").unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("triple(4.0)").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 12.0);
    }

    #[test]
    fn test_context_use_struct() {
        let mut ctx = Context::new();
        ctx.set_base_dir(&examples_dir());
        let program = octoflow_parser::parse("use test_struct_mod\nlet c = Color(1.0, 0.5, 0.0)").unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("c.r").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 1.0);
    }

    #[test]
    fn test_context_use_constant() {
        let mut ctx = Context::new();
        ctx.set_base_dir(&examples_dir());
        let program = octoflow_parser::parse("use test_const_mod").unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("PI").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert!((val.as_float().unwrap() - 3.14159).abs() < 0.001);
    }

    #[test]
    fn test_context_use_array() {
        let mut ctx = Context::new();
        ctx.set_base_dir(&examples_dir());
        let program = octoflow_parser::parse("use test_array_mod").unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("arr[2]").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 30.0);
    }

    // ── Phase 28: For-Each Loops — REPL Context Tests ────────────

    #[test]
    fn test_context_foreach_basic() {
        let mut ctx = Context::new();
        let src = "let arr = [10.0, 20.0, 30.0]\nlet mut total = 0.0\nfor x in arr\n  total = total + x\nend";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("total").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 60.0);
    }

    #[test]
    fn test_context_foreach_with_break() {
        let mut ctx = Context::new();
        let src = "let arr = [1.0, 2.0, 3.0, 4.0, 5.0]\nlet mut total = 0.0\nfor x in arr\n  if x > 3.0\n    break\n  end\n  total = total + x\nend";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("total").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 6.0); // 1+2+3
    }

    #[test]
    fn test_context_foreach_empty_array() {
        let mut ctx = Context::new();
        let src = "let arr = []\nlet mut total = 99.0\nfor x in arr\n  total = total + x\nend";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("total").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 99.0); // unchanged
    }

    #[test]
    fn test_context_foreach_continue() {
        let mut ctx = Context::new();
        let src = "let arr = [1.0, 2.0, 3.0, 4.0]\nlet mut total = 0.0\nfor x in arr\n  if x == 2.0\n    continue\n  end\n  total = total + x\nend";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("total").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 8.0); // 1+3+4
    }

    // ── Phase 29 REPL tests: array mutation ──────────────────────────

    #[test]
    fn test_context_array_assign() {
        let mut ctx = Context::new();
        let src = "let mut arr = [10.0, 20.0, 30.0]\narr[1] = 99.0";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("arr[1]").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 99.0);
    }

    #[test]
    fn test_context_array_push() {
        let mut ctx = Context::new();
        let src = "let mut arr = [1.0]\npush(arr, 2.0)\npush(arr, 3.0)";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let expr = octoflow_parser::parse_expr("len(arr)").unwrap();
        let val = ctx.eval_expression(&expr).unwrap();
        assert_eq!(val.as_float().unwrap(), 3.0);
    }

    #[test]
    fn test_context_array_pop() {
        let mut ctx = Context::new();
        let src = "let mut arr = [10.0, 20.0, 30.0]\nlet x = pop(arr)";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let val_x = ctx.eval_expression(&octoflow_parser::parse_expr("x").unwrap()).unwrap();
        assert_eq!(val_x.as_float().unwrap(), 30.0);
        let val_len = ctx.eval_expression(&octoflow_parser::parse_expr("len(arr)").unwrap()).unwrap();
        assert_eq!(val_len.as_float().unwrap(), 2.0);
    }

    #[test]
    fn test_context_array_immutable_assign_fails() {
        let mut ctx = Context::new();
        let src = "let arr = [1.0, 2.0]";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
        let src2 = "arr[0] = 5.0";
        let program2 = octoflow_parser::parse(src2).unwrap();
        let result = ctx.eval_statement(&program2.statements[0].0, &program2.statements[0].1);
        assert!(result.is_err());
    }

    // ── Phase 30a REPL tests: array params ──────────────────────────

    #[test]
    fn test_context_fn_array_param() {
        let mut ctx = Context::new();
        let src = "fn my_sum(arr)\n  let mut total = 0.0\n  for x in arr\n    total = total + x\n  end\n  return total\nend\nlet data = [10.0, 20.0, 30.0]\nlet s = my_sum(data)\n";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
    }

    #[test]
    fn test_history_save_load_roundtrip() {
        let tmp = std::env::temp_dir().join("octoflow_repl_test");
        let _ = std::fs::create_dir_all(&tmp);
        let hist_path = tmp.join("repl_history");
        // Write history
        let entries = vec!["let x = 1".to_string(), "print(x)".to_string(), "x + 1".to_string()];
        let content: String = entries.iter().map(|s| format!("{}\n", s)).collect();
        std::fs::write(&hist_path, &content).unwrap();
        // Read back
        let loaded: Vec<String> = std::fs::read_to_string(&hist_path).unwrap()
            .lines().map(|s| s.to_string()).filter(|s| !s.is_empty()).collect();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0], "let x = 1");
        assert_eq!(loaded[2], "x + 1");
        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_context_fn_array_param_mutate() {
        let mut ctx = Context::new();
        let src = "fn double_all(arr)\n  let n = len(arr)\n  for i in range(0, n)\n    arr[i] = arr[i] * 2.0\n  end\n  return 0.0\nend\nlet mut data = [1.0, 2.0, 3.0]\nlet _ = double_all(data)\n";
        let program = octoflow_parser::parse(src).unwrap();
        for (stmt, span) in &program.statements {
            ctx.eval_statement(stmt, &span).unwrap();
        }
    }
}
