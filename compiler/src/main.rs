//! OctoFlow CLI — compile and run .flow programs on the GPU.
//!
//! Usage:
//!   `octoflow run <program.flow> [--set name=value]... [-i path] [-o path] [--watch]`
//!   `octoflow check <program.flow>` — pre-flight validation (no GPU)
//!   `octoflow graph <program.flow>` — show dispatch plan (no GPU)

use octoflow_cli::CliError;
use octoflow_cli::Overrides;
use octoflow_cli::PermScope;
use octoflow_cli::Value;
use octoflow_cli::{chat, compiler, mcp, preflight, range_tracker, lint, repl, update};
use std::time::SystemTime;

fn main() {
    // Spawn with 16 MB stack to support deep recursion in user functions
    // (e.g. recursive factorial, nested for-each, complex map operations).
    let stack_size = 16 * 1024 * 1024;
    let builder = std::thread::Builder::new().stack_size(stack_size);
    let handler = builder.spawn(real_main).expect("failed to spawn main thread");
    if let Err(e) = handler.join() {
        eprintln!("fatal: {:?}", e);
        std::process::exit(1);
    }
}

fn real_main() {
    use octoflow_cli::VERSION;
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("OctoFlow v{} — GPU-native language", VERSION);
        eprintln!("CPU on demand, not GPU on demand.");
        eprintln!();
        eprintln!("USAGE:");
        eprintln!("  octoflow chat                AI coding assistant (local LLM or API)");
        eprintln!("  octoflow new <template> <name>  Scaffold a new project");
        eprintln!("  octoflow repl                Interactive REPL");
        eprintln!("  octoflow run <file.flow>     Execute a program");
        eprintln!("  octoflow run -w <file.flow>  Watch mode (re-run on save)");
        eprintln!("  octoflow build <file.flow>   Bundle program + imports into one file");
        eprintln!("  octoflow test [dir]          Run fn test_*() functions");
        eprintln!("  octoflow check <file.flow>   Pre-flight validation (no GPU)");
        eprintln!("  octoflow graph <file.flow>   Show dispatch plan (no GPU)");
        eprintln!("  octoflow mcp-serve           MCP server (JSON-RPC 2.0 over stdio)");
        eprintln!("  octoflow update              Check for updates and self-update");
        eprintln!("  octoflow help <domain>       Show stdlib modules for a domain");
        eprintln!();
        eprintln!("DOMAINS:");
        eprintln!("  data       CSV, dataframes, transforms, I/O");
        eprintln!("  stats      Descriptive, hypothesis, timeseries, risk");
        eprintln!("  ml         Regression, clustering, neural nets, metrics");
        eprintln!("  ai         Inference, tokenizer, fine-tuning, RAG");
        eprintln!("  science    Signal processing, physics, optimization");
        eprintln!("  media      Image filters, audio, color, drawing");
        eprintln!("  devops     Log parsing, filesystem, process, config");
        eprintln!("  sys        Platform, args, timer, memory, testing");
        eprintln!("  web        HTTP client, URL, auth, JSON utilities");
        eprintln!("  db         GPU-accelerated database, queries, indexing");
        eprintln!("  crypto     Hashing, encoding, secure random");
        eprintln!("  net        TCP, UDP, WebSocket, P2P");
        eprintln!();
        eprintln!("RUN OPTIONS:");
        eprintln!("  --set name=value   Override a scalar value");
        eprintln!("  -i path            Override input file path");
        eprintln!("  -o path            Override output file path");
        eprintln!("  --watch, -w        Re-run on file changes");
        eprintln!("  -q, --quiet        Suppress timing output");
        eprintln!("  --allow-read[=path]  Allow file read operations (optionally scoped)");
        eprintln!("  --allow-write[=path] Allow file write operations (optionally scoped)");
        eprintln!("  --allow-net[=host]   Allow network access (optionally scoped to host)");
        eprintln!("  --allow-exec[=path]  Allow command execution (optionally scoped)");
        eprintln!("  --allow-ffi        Allow FFI calls to native shared libraries");
        eprintln!("  --max-iters N      Override while-loop iteration limit (default 10000)");
        eprintln!("  --gpu-max-mb N     Limit GPU memory allocation to N megabytes");
        eprintln!("  --verbose          Verbose inference logging (per-weight debug output)");
        eprintln!("  --output json      Structured JSON output for all commands (agent mode)");
        eprintln!("  --stdin-as <var>   Pipe stdin into a variable before execution");
        eprintln!("  --format json      Output errors as JSON (machine-readable)");
        if args.len() < 2 {
            std::process::exit(1);
        }
        return;
    }

    let subcommand = &args[1];

    // --version / -V
    if subcommand == "--version" || subcommand == "-V" {
        println!("OctoFlow {}", VERSION);
        return;
    }

    // These subcommands don't require additional args
    if subcommand != "repl" && subcommand != "help" && subcommand != "chat"
        && subcommand != "new" && subcommand != "test" && subcommand != "build"
        && subcommand != "mcp-serve" && subcommand != "update" && subcommand != "search" && args.len() < 3
    {
        eprintln!("Usage: octoflow {} <program.flow> [options]", subcommand);
        std::process::exit(1);
    }

    let result = match subcommand.as_str() {
        "run" => run(&args[2..]),
        "check" => check(&args[2..]),
        "graph" => graph(&args[2]),
        "repl" => repl::run_repl(),
        "chat" => chat::run_chat(&args[2..]),
        "new" => new_project(&args[2..]),
        "test" => run_tests(&args[2..]),
        "build" => build(&args[2..]),
        "mcp-serve" => {
            let overrides = mcp::parse_mcp_args(&args[2..]);
            mcp::serve_mcp(overrides)
        }
        "search" => search(&args[2..]),
        "update" => update::run_update(&args[2..]),
        "help" => {
            let domain = args.get(2).map(|s| s.as_str()).unwrap_or("");
            print_domain_help(domain);
            Ok(())
        }
        _ => {
            eprintln!("Unknown subcommand: '{}'. Run octoflow --help for usage.", subcommand);
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

/// Parse `--set name=value` into (name, Value).
fn parse_set_arg(s: &str) -> Result<(String, Value), CliError> {
    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(CliError::Compile(format!(
            "--set requires name=value format, got '{}'", s)));
    }
    let name = parts[0].to_string();
    let value = match parts[1].parse::<f32>() {
        Ok(f) => Value::Float(f),
        Err(_) => Value::Str(parts[1].to_string()),
    };
    Ok((name, value))
}

/// Parsed run options.
struct RunOpts<'a> {
    flow_file: &'a str,
    overrides: Overrides,
    watch: bool,
    quiet: bool,
    format_json: bool,
    /// --output json: wrap ALL output (success + error) in structured JSON envelope.
    output_json: bool,
    /// --stdin-as <varname>: read stdin into a variable before execution.
    stdin_as: Option<String>,
}

/// Parse run subcommand args: `<program.flow> [--set name=value]... [-i path] [-o path] [--watch|-w] [-q]`
fn parse_run_args(args: &[String]) -> Result<RunOpts<'_>, CliError> {
    if args.is_empty() {
        return Err(CliError::Compile("run requires a .flow file".into()));
    }

    let flow_file = &args[0];
    let mut overrides = Overrides::default();
    let mut watch = false;
    let mut quiet = false;
    let mut format_json = false;
    let mut output_json = false;
    let mut stdin_as: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--set" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--set requires name=value".into()));
                }
                let (name, val) = parse_set_arg(&args[i + 1])?;
                overrides.scalars.insert(name, val);
                i += 2;
            }
            "-i" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("-i requires a path".into()));
                }
                overrides.input_path = Some(args[i + 1].clone());
                i += 2;
            }
            "-o" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("-o requires a path".into()));
                }
                overrides.output_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--watch" | "-w" => {
                watch = true;
                i += 1;
            }
            "-q" | "--quiet" => {
                quiet = true;
                i += 1;
            }
            "--allow-read" => {
                overrides.allow_read = PermScope::AllowAll;
                i += 1;
            }
            "--allow-write" => {
                overrides.allow_write = PermScope::AllowAll;
                i += 1;
            }
            "--allow-net" => {
                overrides.allow_net = PermScope::AllowAll;
                i += 1;
            }
            "--allow-exec" => {
                overrides.allow_exec = PermScope::AllowAll;
                i += 1;
            }
            "--allow-ffi" => {
                eprintln!("warning: --allow-ffi enables loading arbitrary native code. Only use with trusted .flow programs.");
                overrides.allow_ffi = true;
                i += 1;
            }
            "--verbose" => {
                overrides.verbose = true;
                i += 1;
            }
            "--format" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--format requires a value (json)".into()));
                }
                if args[i + 1] == "json" {
                    format_json = true;
                } else {
                    return Err(CliError::Compile(format!("unknown format: '{}'. Supported: json", args[i + 1])));
                }
                i += 2;
            }
            "--output" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--output requires a value (json)".into()));
                }
                if args[i + 1] == "json" {
                    output_json = true;
                } else {
                    return Err(CliError::Compile(format!("unknown output format: '{}'. Supported: json", args[i + 1])));
                }
                i += 2;
            }
            "--stdin-as" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--stdin-as requires a variable name".into()));
                }
                stdin_as = Some(args[i + 1].clone());
                i += 2;
            }
            "--max-iters" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--max-iters requires a number".into()));
                }
                let n = args[i + 1].parse::<usize>().map_err(|_| {
                    CliError::Compile(format!("--max-iters: invalid number '{}'", args[i + 1]))
                })?;
                overrides.max_iters = Some(n);
                i += 2;
            }
            "--gpu-max-mb" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--gpu-max-mb requires a number".into()));
                }
                let mb = args[i + 1].parse::<u64>().map_err(|_| {
                    CliError::Compile(format!("--gpu-max-mb: invalid number '{}'", args[i + 1]))
                })?;
                overrides.gpu_max_bytes = Some(mb * 1024 * 1024);
                i += 2;
            }
            other if other.starts_with("--allow-read=") => {
                let path = &other["--allow-read=".len()..];
                match &mut overrides.allow_read {
                    PermScope::AllowScoped(ref mut paths) => paths.push(std::path::PathBuf::from(path)),
                    _ => overrides.allow_read = PermScope::AllowScoped(vec![std::path::PathBuf::from(path)]),
                }
                i += 1;
            }
            other if other.starts_with("--allow-write=") => {
                let path = &other["--allow-write=".len()..];
                match &mut overrides.allow_write {
                    PermScope::AllowScoped(ref mut paths) => paths.push(std::path::PathBuf::from(path)),
                    _ => overrides.allow_write = PermScope::AllowScoped(vec![std::path::PathBuf::from(path)]),
                }
                i += 1;
            }
            other if other.starts_with("--allow-net=") => {
                let host = &other["--allow-net=".len()..];
                match &mut overrides.allow_net {
                    PermScope::AllowScoped(ref mut paths) => paths.push(std::path::PathBuf::from(host)),
                    _ => overrides.allow_net = PermScope::AllowScoped(vec![std::path::PathBuf::from(host)]),
                }
                i += 1;
            }
            other if other.starts_with("--allow-exec=") => {
                let path = &other["--allow-exec=".len()..];
                match &mut overrides.allow_exec {
                    PermScope::AllowScoped(ref mut paths) => paths.push(std::path::PathBuf::from(path)),
                    _ => overrides.allow_exec = PermScope::AllowScoped(vec![std::path::PathBuf::from(path)]),
                }
                i += 1;
            }
            other => {
                return Err(CliError::Compile(format!("unknown option: '{}'", other)));
            }
        }
    }

    Ok(RunOpts { flow_file, overrides, watch, quiet, format_json, output_json, stdin_as })
}

fn run(args: &[String]) -> Result<(), CliError> {
    let opts = parse_run_args(args)?;

    if opts.watch {
        return watch_and_run(opts.flow_file, &opts.overrides);
    }

    // --output json: capture all output and return structured JSON envelope
    if opts.output_json {
        return run_json_mode(opts.flow_file, &opts.overrides, opts.stdin_as.as_deref());
    }

    let start = std::time::Instant::now();
    let result = if let Some(ref var_name) = opts.stdin_as {
        run_once_with_stdin(opts.flow_file, &opts.overrides, var_name)
    } else {
        run_once(opts.flow_file, &opts.overrides)
    };
    if !opts.quiet {
        let elapsed = start.elapsed();
        eprintln!();
        if result.is_ok() {
            eprintln!("{}", format_timing(elapsed, true));
        } else {
            eprintln!("{}", format_timing(elapsed, false));
        }
    }
    // Enhanced error display with source context or JSON format
    if let Err(ref e) = result {
        let source = std::fs::read_to_string(opts.flow_file).ok();
        if opts.format_json {
            eprintln!("{}", octoflow_cli::format_error_json_full(e, source.as_deref()));
            std::process::exit(1);
        }
        // Show source line context for runtime errors
        if let Some(ref src) = source {
            eprintln!("{}", octoflow_cli::format_error_with_source(e, src));
            std::process::exit(1);
        }
    }
    result
}

/// Run in --output json mode: capture stdout and return structured JSON envelope.
fn run_json_mode(flow_file: &str, overrides: &Overrides, stdin_as: Option<&str>) -> Result<(), CliError> {
    let start = std::time::Instant::now();

    // Read and parse
    let source = std::fs::read_to_string(flow_file)
        .map_err(|e| CliError::Io(format!("{}: {}", flow_file, e)))?;

    let mut program = octoflow_parser::parse(&source)
        .map_err(|e| CliError::Parse(format!("{}", e)))?;

    let base_dir = std::path::Path::new(flow_file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string());

    // Pre-flight validation
    let report = preflight::validate(&program, &base_dir);
    if !report.passed {
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let source_ref = std::fs::read_to_string(flow_file).ok();
        let err = CliError::Compile("pre-flight check failed".into());
        let error_json = octoflow_cli::format_error_json_full(&err, source_ref.as_deref());
        println!("{{\"status\":\"error\",\"error\":{},\"exit_code\":1,\"time_ms\":{:.1}}}", error_json, elapsed_ms);
        return Ok(());
    }

    // Inject --stdin-as variable if provided
    if let Some(var_name) = stdin_as {
        let mut input = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)
            .map_err(|e| CliError::Io(format!("reading stdin: {}", e)))?;
        // Prepend a `let <varname> = "<escaped>"` statement
        let escaped = input.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
        let prefix = format!("let {} = \"{}\"\n", var_name, escaped);
        let combined = format!("{}{}", prefix, source);
        program = octoflow_parser::parse(&combined)
            .map_err(|e| CliError::Parse(format!("{}", e)))?;
    }

    // Capture output and execute
    compiler::capture_output_start();
    let exec_result = compiler::execute(&program, &base_dir, overrides);
    let output = compiler::capture_output_take();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    match exec_result {
        Ok(_) => {
            let escaped_output = json_escape_str(&output);
            println!("{{\"status\":\"success\",\"output\":\"{}\",\"exit_code\":0,\"time_ms\":{:.1}}}", escaped_output, elapsed_ms);
        }
        Err(ref e) => {
            let error_json = octoflow_cli::format_error_json_full(e, Some(&source));
            let escaped_output = json_escape_str(&output);
            println!("{{\"status\":\"error\",\"output\":\"{}\",\"error\":{},\"exit_code\":1,\"time_ms\":{:.1}}}", escaped_output, error_json, elapsed_ms);
        }
    }
    Ok(())
}

/// Run a program with stdin piped into a variable.
fn run_once_with_stdin(flow_file: &str, overrides: &Overrides, var_name: &str) -> Result<(), CliError> {
    let source = std::fs::read_to_string(flow_file)
        .map_err(|e| CliError::Io(format!("{}: {}", flow_file, e)))?;

    // Read stdin
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)
        .map_err(|e| CliError::Io(format!("reading stdin: {}", e)))?;

    // Prepend a let statement to inject the variable
    let escaped = input.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
    let combined = format!("let {} = \"{}\"\n{}", var_name, escaped, source);

    let program = octoflow_parser::parse(&combined)
        .map_err(|e| CliError::Parse(format!("{}", e)))?;

    let base_dir = std::path::Path::new(flow_file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string());

    let report = preflight::validate(&program, &base_dir);
    if !report.passed {
        eprintln!("PRE-FLIGHT REPORT: {}", flow_file);
        eprintln!("{}", report);
        return Err(CliError::Compile("pre-flight check failed".into()));
    }

    compiler::execute(&program, &base_dir, overrides)?;
    Ok(())
}

/// Escape a string for JSON output.
fn json_escape_str(s: &str) -> String {
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
            c => out.push(c),
        }
    }
    out
}

fn format_timing(elapsed: std::time::Duration, success: bool) -> String {
    let icon = if success { "ok" } else { "err" };
    let secs = elapsed.as_secs_f64();
    if secs >= 1.0 {
        format!("{} {:.1}s", icon, secs)
    } else {
        format!("{} {}ms", icon, elapsed.as_millis())
    }
}

/// Execute a .flow program once (parse, validate, run).
fn run_once(flow_file: &str, overrides: &Overrides) -> Result<(), CliError> {
    // Read source file
    let source = std::fs::read_to_string(flow_file)
        .map_err(|e| CliError::Io(format!("{}: {}", flow_file, e)))?;

    // Parse
    let program = octoflow_parser::parse(&source)
        .map_err(|e| CliError::Parse(format!("{}", e)))?;

    // Determine base directory for relative path resolution
    let base_dir = std::path::Path::new(flow_file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string());

    // Pre-flight validation
    let report = preflight::validate(&program, &base_dir);
    if !report.passed {
        eprintln!("PRE-FLIGHT REPORT: {}", flow_file);
        eprintln!("{}", report);
        return Err(CliError::Compile("pre-flight check failed".into()));
    }

    // Print warnings (if any) but continue
    let has_warnings = report.checks.iter().any(|c| !c.warnings.is_empty());
    if has_warnings {
        eprintln!("PRE-FLIGHT REPORT: {}", flow_file);
        eprintln!("{}", report);
        eprintln!();
    }

    // Log overrides
    if !overrides.scalars.is_empty() {
        let pairs: Vec<String> = overrides.scalars.iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        eprintln!("Overrides: {}", pairs.join(", "));
    }
    if let Some(ref p) = overrides.input_path {
        eprintln!("Input override: {}", p);
    }
    if let Some(ref p) = overrides.output_path {
        eprintln!("Output override: {}", p);
    }

    // Execute
    compiler::execute(&program, &base_dir, overrides)?;

    Ok(())
}

/// Watch mode: run the program, then poll for file changes and re-run.
fn watch_and_run(flow_file: &str, overrides: &Overrides) -> Result<(), CliError> {
    loop {
        // Run the program — print errors but don't exit
        match run_once(flow_file, overrides) {
            Ok(()) => {}
            Err(e) => eprintln!("error: {}", e),
        }

        // Collect files to watch (main file + use imports)
        let watch_files = collect_watched_files(flow_file);
        let mtimes = get_modification_times(&watch_files);

        eprintln!();
        eprintln!("--- Watching {} file{} for changes (Ctrl+C to stop) ---",
            watch_files.len(),
            if watch_files.len() == 1 { "" } else { "s" });

        // Poll for changes every 500ms
        loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let new_mtimes = get_modification_times(&watch_files);
            if new_mtimes != mtimes {
                eprintln!();
                eprintln!("--- File changed, recompiling... ---");
                eprintln!();
                break;
            }
        }
    }
}

/// Collect the main .flow file and any `use`-imported module files.
fn collect_watched_files(flow_file: &str) -> Vec<String> {
    let mut files = vec![flow_file.to_string()];

    // Parse to find use imports (best-effort — if parsing fails, just watch main file)
    if let Ok(source) = std::fs::read_to_string(flow_file) {
        if let Ok(program) = octoflow_parser::parse(&source) {
            let base_dir = std::path::Path::new(flow_file)
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| ".".to_string());

            for (stmt, _) in &program.statements {
                if let octoflow_parser::ast::Statement::UseDecl { module } = stmt {
                    let module_path = if base_dir.is_empty() || base_dir == "." {
                        format!("{}.flow", module)
                    } else {
                        format!("{}/{}.flow", base_dir, module)
                    };
                    if std::path::Path::new(&module_path).exists() {
                        files.push(module_path);
                    }
                }
            }
        }
    }

    files
}

/// Get modification times for a list of files.
fn get_modification_times(files: &[String]) -> Vec<Option<SystemTime>> {
    files.iter()
        .map(|f| std::fs::metadata(f).ok().and_then(|m| m.modified().ok()))
        .collect()
}

fn print_domain_help(domain: &str) {
    let text = match domain {
        "data" => "\
stdlib/data/ — Data Science

  use csv            read_csv, write_csv, csv_column, csv_filter
  use dataframe      df_from_csv, df_where, df_sort, df_group_by, df_join
  use transform      normalize (GPU), standardize (GPU), diff, cumsum, rank
  use io             gpu_load, gpu_save, json_load, json_save
  use pipeline       GPU-resident chainable data operations",

        "stats" => "\
stdlib/stats/ — Statistics & Math

  use descriptive    mean, median, stddev, skewness, kurtosis, describe
  use hypothesis     t_test, paired_t_test, chi_squared, anova, z_test
  use correlation    pearson, spearman, covariance, linear_fit
  use distribution   normal_pdf, normal_cdf, uniform_random, normal_random
  use timeseries     sma, ema, bollinger, rsi, macd, returns, drawdown
  use risk           sharpe_ratio, sortino, max_drawdown, VaR, volatility
  use math_ext       factorial, gcd, lcm, is_prime, PI, E, TAU",

        "ml" => "\
stdlib/ml/ — Machine Learning

  use preprocess     train_test_split, feature_scale, encode_labels
  use regression     linear_regression (GPU), ridge, polynomial_fit
  use classify       knn, logistic_regression, naive_bayes
  use cluster        kmeans (GPU), distance_matrix (GPU), dbscan
  use nn             dense_layer (GPU), relu, sigmoid, softmax, forward
  use metrics        accuracy, precision, recall, f1, confusion_matrix
  use linalg         mat_inverse, solve, eigenvalues, svd",

        "ai" => "\
stdlib/ai/ — AI & Inference

  use inference      model_load (GGUF/ONNX), generate, embed, batch_generate
  use tokenizer      tokenizer_load (BPE/SPM), encode, decode
  use attention      multi_head_attention (GPU), kv_cache, rotary_embedding
  use serve          serve_model (OpenAI-compatible API), stream_generate
  use quantize       quantize (GPU), dequantize, calibrate
  use finetune       lora_init, lora_train, lora_merge, dataset_load
  use train          forward_pass (GPU), backward_pass (GPU), training_loop
  use embedding      embed_text, cosine_similarity (GPU), semantic_search
  use rag            chunk_text, build_index, search_index, rag_generate",

        "science" => "\
stdlib/science/ — Scientific Computing

  use signal         convolve (GPU), cross_correlate (GPU), bandpass, hamming
  use physics        integrate_euler, integrate_rk4, n_body (GPU), wave_eq (GPU)
  use interpolate    linear_interp, cubic_spline, bilinear_interp
  use optimize       gradient_descent, golden_section, newton_raphson
  use constants      PI, E, SPEED_OF_LIGHT, PLANCK, BOLTZMANN, AVOGADRO",

        "media" => "\
stdlib/media/ — Media & Image Processing

  use image          img_load, img_save, img_resize (GPU), img_crop, img_rotate (GPU)
  use filter         grayscale, brightness, contrast, blur, sharpen, edge_detect (GPU)
  use transform      img_scale (GPU), img_perspective (GPU), img_warp (GPU)
  use color          rgb_to_hsl, hex_to_rgb, blend, palette_extract (GPU)
  use draw           draw_line, draw_rect, draw_circle, draw_text
  use audio          wav_load, wav_save, audio_gain (GPU), audio_normalize (GPU)
  use convert        img_to_png, img_to_jpg, batch_convert",

        "devops" => "\
stdlib/devops/ — DevOps & Scripting

  use log            parse_log, log_filter, log_stats, tail_log, grep
  use fs             walk_dir, find_files, file_info, glob, copy_file, move_file
  use process        run, pipe, env_get, env_set
  use template       render, load_template
  use config         parse_ini, parse_env, parse_toml",

        "sys" => "\
stdlib/sys/ — Systems Programming

  use platform       os_name, os_arch, is_windows, is_linux, gpu_info
  use args           parse_args, arg_flag, arg_value, arg_required
  use timer          stopwatch, elapsed, benchmark
  use memory         gpu_mem_usage, buffer_info
  use test           assert_eq, assert_near, test_suite, run_tests",

        "web" => "\
stdlib/web/ — Web & API

  use http           get, post, put, delete, response_json, response_status
  use url            url_parse, url_encode, url_decode, query_string
  use auth           basic_auth, bearer_token, hmac_sign, api_key_header
  use json_util      json_query, json_flatten, json_merge, json_pretty",

        "db" => "\
stdlib/db/ — Native Database (GPU-Accelerated)

  use core           db_create, db_open, db_close, db_table, db_info
  use write          db_insert, db_insert_batch, db_append, db_update
  use query          db_select, db_where (GPU), db_order (GPU), db_group (GPU)
  use index          db_create_index, db_drop_index
  use storage        columnar .od storage, WAL, compaction, db_to_gpu
  use migrate        db_import_csv, db_export_csv, db_import_json",

        "crypto" => "\
stdlib/crypto/ — Security

  use hash           sha256, hmac_sha256, simple_hash, hash_file
  use encoding       base64_encode/decode, hex_encode/decode, url_encode
  use random_util    random_bytes, random_hex, uuid_v4, random_token",

        "net" => "\
stdlib/net/ — Networking

  use tcp            tcp_connect, tcp_listen, tcp_accept, tcp_send, tcp_recv
  use udp            udp_bind, udp_send_to, udp_recv_from
  use websocket      ws_connect, ws_send, ws_recv, ws_close
  use protocol       msg_pack, msg_unpack
  use p2p            peer_discover, peer_connect, peer_broadcast
  use server         http_serve, route, response, static_files",

        "" => {
            eprintln!("Usage: octoflow help <domain>");
            eprintln!("Run octoflow --help to see available domains.");
            return;
        }
        _ => {
            eprintln!("Unknown domain: \"{}\". Run octoflow --help to see available domains.", domain);
            return;
        }
    };

    println!("{}", text);
}

fn search(args: &[String]) -> Result<(), CliError> {
    // octoflow search "query terms" [--dir path] [--top N]
    if args.is_empty() {
        eprintln!("Usage: octoflow search \"query\" [--dir path] [--top N]");
        std::process::exit(1);
    }

    let query = &args[0];
    let mut dir = ".".to_string();
    let mut top = 10u32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" if i + 1 < args.len() => {
                dir = args[i + 1].replace('\\', "/");
                i += 2;
            }
            "--top" if i + 1 < args.len() => {
                top = args[i + 1].parse().unwrap_or(10);
                i += 2;
            }
            _ => { i += 1; }
        }
    }

    // Build .flow script that delegates to the search engine
    let script = format!(
        "use \"stdlib/search/engine\"\noctosearch(\"{}\", \"{}\", {})\n",
        query.replace('\\', "\\\\").replace('"', "\\\""),
        dir,
        top
    );

    let program = octoflow_parser::parse(&script)
        .map_err(|e| CliError::Parse(format!("{}", e)))?;

    let mut overrides = Overrides::default();
    overrides.allow_read = PermScope::AllowAll;

    let base_dir = ".";
    compiler::execute(&program, base_dir, &overrides)?;
    Ok(())
}

fn new_project(args: &[String]) -> Result<(), CliError> {
    use octoflow_cli::templates;

    if args.is_empty() {
        eprintln!("Usage: octoflow new <template> [project-name]");
        eprintln!();
        eprintln!("Templates:");
        for (name, desc) in templates::TEMPLATES {
            eprintln!("  {:<12} {}", name, desc);
        }
        return Ok(());
    }

    let template_name = &args[0];
    let project_name = if args.len() > 1 {
        args[1].clone()
    } else {
        template_name.to_string()
    };

    // Get the template content (validates the template name)
    let content = templates::get_template(template_name)?;
    let readme = templates::get_readme(template_name, &project_name);

    // Create project directory
    let project_dir = std::path::Path::new(&project_name);
    if project_dir.exists() {
        return Err(CliError::Io(format!("directory already exists: {}", project_name)));
    }

    std::fs::create_dir_all(&project_dir)
        .map_err(|e| CliError::Io(format!("create directory: {}", e)))?;

    // Write main.flow
    let main_path = project_dir.join("main.flow");
    std::fs::write(&main_path, content)
        .map_err(|e| CliError::Io(format!("write main.flow: {}", e)))?;

    // Write README.md
    let readme_path = project_dir.join("README.md");
    std::fs::write(&readme_path, readme)
        .map_err(|e| CliError::Io(format!("write README.md: {}", e)))?;

    // For dashboard template, create a sample data.csv
    if template_name == "dashboard" {
        let sample_csv = "name,value\nalpha,42\nbeta,17\ngamma,93\ndelta,8\nepsilon,55\n";
        let csv_path = project_dir.join("data.csv");
        std::fs::write(&csv_path, sample_csv)
            .map_err(|e| CliError::Io(format!("write data.csv: {}", e)))?;
    }

    // For script template, create a sample input.txt
    if template_name == "script" {
        let sample_input = "Hello World\nfoo bar baz\n\noctoflow is great\n";
        let input_path = project_dir.join("input.txt");
        std::fs::write(&input_path, sample_input)
            .map_err(|e| CliError::Io(format!("write input.txt: {}", e)))?;
    }

    eprintln!("Created {}/", project_name);
    eprintln!("  main.flow    — your program");
    eprintln!("  README.md    — how to run");
    if template_name == "dashboard" {
        eprintln!("  data.csv     — sample data");
    }
    if template_name == "script" {
        eprintln!("  input.txt    — sample input");
    }
    eprintln!();
    eprintln!("Run:");
    eprintln!("  cd {}", project_name);
    eprintln!("  octoflow run main.flow --allow-read --allow-write --allow-net");

    Ok(())
}

fn check(args: &[String]) -> Result<(), CliError> {
    if args.is_empty() {
        return Err(CliError::Compile("check requires a .flow file".into()));
    }
    let flow_file = &args[0];
    let output_json = args.iter().any(|a| a == "--output") && args.iter().any(|a| a == "json");

    // Read source file
    let source = std::fs::read_to_string(flow_file)
        .map_err(|e| CliError::Io(format!("{}: {}", flow_file, e)))?;

    // Parse
    let parse_result = octoflow_parser::parse(&source);
    if output_json {
        if let Err(ref e) = parse_result {
            println!("{{\"valid\":false,\"errors\":[{{\"code\":\"E001\",\"message\":\"{}\",\"line\":0}}],\"warnings\":[]}}",
                json_escape_str(&format!("{}", e)));
            return Ok(());
        }
    }
    let program = parse_result.map_err(|e| CliError::Parse(format!("{}", e)))?;

    // Determine base directory
    let base_dir = std::path::Path::new(flow_file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string());

    // Validate
    let report = preflight::validate(&program, &base_dir);

    if output_json {
        // JSON output mode
        let valid = report.passed;
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        for check_item in &report.checks {
            for err in &check_item.errors {
                errors.push(format!(
                    "{{\"message\":\"{}\",\"line\":{},\"suggestion\":\"{}\"}}",
                    json_escape_str(&err.message), err.line, json_escape_str(&err.suggestion)
                ));
            }
            for warn in &check_item.warnings {
                warnings.push(format!(
                    "{{\"message\":\"{}\",\"line\":{},\"suggestion\":\"{}\"}}",
                    json_escape_str(&warn.message), warn.line, json_escape_str(&warn.suggestion)
                ));
            }
        }
        // Also run lint for warnings
        let lint_report = lint::analyze(&program, &base_dir);
        for warn in &lint_report.warnings {
            warnings.push(format!(
                "{{\"code\":\"{}\",\"message\":\"{}\",\"line\":{},\"suggestion\":\"{}\"}}",
                warn.code, json_escape_str(&warn.message), warn.line, json_escape_str(&warn.suggestion)
            ));
        }
        println!("{{\"valid\":{},\"errors\":[{}],\"warnings\":[{}]}}",
            valid, errors.join(","), warnings.join(","));
        return Ok(());
    }

    // Human output mode
    eprintln!("PRE-FLIGHT REPORT: {}", flow_file);
    eprintln!("{}", report);

    // Range analysis (always print in check mode)
    let range_report = range_tracker::analyze(&program, &base_dir);
    eprintln!();
    eprintln!("RANGE ANALYSIS: {}", flow_file);
    eprintln!("{}", range_report);

    // Lint analysis (always print in check mode)
    let lint_report = lint::analyze(&program, &base_dir);
    eprintln!();
    eprintln!("LINT ANALYSIS: {}", flow_file);
    eprintln!("{}", lint_report);

    if report.passed {
        eprintln!();
        eprintln!("No errors found.");
        Ok(())
    } else {
        Err(CliError::Compile("pre-flight check failed".into()))
    }
}

fn graph(flow_file: &str) -> Result<(), CliError> {
    // Read source file
    let source = std::fs::read_to_string(flow_file)
        .map_err(|e| CliError::Io(format!("{}: {}", flow_file, e)))?;

    // Parse
    let program = octoflow_parser::parse(&source)
        .map_err(|e| CliError::Parse(format!("{}", e)))?;

    // Generate and print dispatch plan
    println!("Dispatch plan:");
    let steps = compiler::plan(&program);
    for step in &steps {
        println!("{}", step);
    }

    Ok(())
}

/// `octoflow test [dir]` — discover and run `fn test_*()` functions in .flow files.
///
/// Scans the current directory (or specified directory) and `tests/` subdirectory
/// for .flow files containing `fn test_*()` functions. Runs each one and reports
/// pass/fail summary. Exit code 1 on any failure.
fn run_tests(args: &[String]) -> Result<(), CliError> {
    use std::time::Instant;

    // Parse args: [dir] [--filter pattern]
    let mut dir = ".";
    let mut filter: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--filter" | "-f" => {
                if i + 1 < args.len() {
                    filter = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err(CliError::Compile("--filter requires a pattern".into()));
                }
            }
            _ => {
                if dir == "." { dir = &args[i]; }
                i += 1;
            }
        }
    }

    let total_start = Instant::now();

    // Collect .flow files recursively from the target directory and tests/ subdirectory
    let mut flow_files = Vec::new();
    collect_flow_files(dir, &mut flow_files);
    let tests_dir = format!("{}/tests", dir);
    if std::path::Path::new(&tests_dir).is_dir() {
        collect_flow_files(&tests_dir, &mut flow_files);
    }

    if flow_files.is_empty() {
        eprintln!("No .flow files found in '{}'", dir);
        return Ok(());
    }

    // Scan each file for fn test_*() declarations
    let mut total_pass = 0usize;
    let mut total_fail = 0usize;
    let mut total_skip = 0usize;

    for file_path in &flow_files {
        let source = match std::fs::read_to_string(file_path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Find test function names: lines matching `fn test_*(`
        let test_names: Vec<String> = source.lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                if trimmed.starts_with("fn test_") {
                    // Extract function name: `fn test_foo(` → `test_foo`
                    let after_fn = &trimmed[3..]; // skip "fn "
                    if let Some(paren) = after_fn.find('(') {
                        let name = after_fn[..paren].trim();
                        if name.starts_with("test_") {
                            return Some(name.to_string());
                        }
                    }
                }
                None
            })
            .collect();

        if test_names.is_empty() {
            continue;
        }

        let relative = file_path.strip_prefix(&format!("{}/", dir)).unwrap_or(file_path);
        eprintln!("--- {} ({} tests) ---", relative, test_names.len());

        // Parse the file once to check for syntax errors
        let _program = match octoflow_parser::parse(&source) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  SKIP (parse error: {})", e);
                total_skip += test_names.len();
                continue;
            }
        };

        let base_dir = std::path::Path::new(file_path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());

        // For each test function, create a program that calls it
        for test_name in &test_names {
            // Apply --filter: skip non-matching tests
            if let Some(ref pat) = filter {
                if !test_name.contains(pat.as_str()) {
                    total_skip += 1;
                    continue;
                }
            }

            // Build a program that includes the file's code + a call to the test function
            let test_source = format!("{}\n{}()\n", source, test_name);
            let test_program = match octoflow_parser::parse(&test_source) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  FAIL {} (parse: {})", test_name, e);
                    total_fail += 1;
                    continue;
                }
            };

            // Grant read + write permissions for test files (but not net/exec)
            let overrides = Overrides {
                allow_read: PermScope::AllowAll,
                allow_write: PermScope::AllowAll,
                ..Default::default()
            };

            // Capture output to suppress test noise
            let t0 = Instant::now();
            compiler::capture_output_start();
            let result = compiler::execute(&test_program, &base_dir, &overrides);
            let _captured = compiler::capture_output_take();
            let elapsed = t0.elapsed();

            match result {
                Ok(_) => {
                    eprintln!("  PASS {} ({:.1}ms)", test_name, elapsed.as_secs_f64() * 1000.0);
                    total_pass += 1;
                }
                Err(e) => {
                    eprintln!("  FAIL {} ({:.1}ms) — {}", test_name, elapsed.as_secs_f64() * 1000.0, e);
                    total_fail += 1;
                }
            }
        }
    }

    let total_elapsed = total_start.elapsed();
    eprintln!();
    eprintln!("{} passed, {} failed, {} skipped ({:.2}s)",
        total_pass, total_fail, total_skip, total_elapsed.as_secs_f64());

    if total_fail > 0 {
        std::process::exit(1);
    }

    Ok(())
}

/// Collect .flow files in a directory (recursive with depth limit).
fn collect_flow_files(dir: &str, out: &mut Vec<String>) {
    collect_flow_files_recursive(dir, out, 0);
    out.sort();
}

fn collect_flow_files_recursive(dir: &str, out: &mut Vec<String>, depth: usize) {
    if depth > 10 { return; } // prevent symlink cycles
    let entries = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(s) = path.to_str() {
                collect_flow_files_recursive(s, out, depth + 1);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("flow") {
            if let Some(s) = path.to_str() {
                out.push(s.to_string());
            }
        }
    }
}

// ── octoflow build — bundle .flow + stdlib into a single distributable ───────

/// Build a self-contained .flow bundle from a program and all its dependencies.
fn build(args: &[String]) -> Result<(), CliError> {
    if args.is_empty() {
        eprintln!("Usage: octoflow build <program.flow> [options]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  -o, --output <path>    Output file (default: <name>.bundle.flow)");
        eprintln!("  --list                 List dependencies without bundling");
        eprintln!();
        eprintln!("Bundles a .flow program with all its imports into a single file.");
        eprintln!("The output can run with just: octoflow run bundle.flow");
        return Ok(());
    }

    let mut entry_file = String::new();
    let mut output_file: Option<String> = None;
    let mut list_only = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--output requires a path".into()));
                }
                output_file = Some(args[i + 1].clone());
                i += 2;
            }
            "--list" => {
                list_only = true;
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(CliError::Compile(format!("unknown build option: '{}'", other)));
            }
            path => {
                if entry_file.is_empty() {
                    entry_file = path.to_string();
                } else {
                    return Err(CliError::Compile(format!("unexpected argument: '{}'", path)));
                }
                i += 1;
            }
        }
    }

    if entry_file.is_empty() {
        return Err(CliError::Compile("no input file specified".into()));
    }

    if !std::path::Path::new(&entry_file).exists() {
        return Err(CliError::Io(format!("file not found: {}", entry_file)));
    }

    // Resolve default output name
    let output = output_file.unwrap_or_else(|| {
        let stem = std::path::Path::new(&entry_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        format!("{}.bundle.flow", stem)
    });

    let base_dir = std::path::Path::new(&entry_file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string());

    // Phase 1: Trace all imports recursively
    eprintln!("Tracing imports from {}...", entry_file);
    let mut visited = std::collections::HashSet::new();
    let mut modules: Vec<(String, String, String)> = Vec::new(); // (module_name, path, source)
    trace_imports(&entry_file, &base_dir, &mut visited, &mut modules)?;

    if list_only {
        eprintln!("Dependencies ({}):", modules.len());
        for (name, path, source) in &modules {
            let lines = source.lines().count();
            eprintln!("  {} ({}, {} lines)", name, path, lines);
        }
        // Also show main file
        let main_source = std::fs::read_to_string(&entry_file)
            .map_err(|e| CliError::Io(format!("{}: {}", entry_file, e)))?;
        let main_lines = main_source.lines().count();
        let total_lines: usize = modules.iter().map(|(_, _, s)| s.lines().count()).sum::<usize>() + main_lines;
        eprintln!("  {} (entry, {} lines)", entry_file, main_lines);
        eprintln!("Total: {} files, {} lines", modules.len() + 1, total_lines);
        return Ok(());
    }

    // Phase 2: Read entry file source
    let entry_source = std::fs::read_to_string(&entry_file)
        .map_err(|e| CliError::Io(format!("{}: {}", entry_file, e)))?;

    // Phase 3: Bundle — concatenate modules (deps first), then entry
    let mut bundle = String::new();
    bundle.push_str("// OctoFlow Bundle — generated by `octoflow build`\n");
    bundle.push_str(&format!("// Source: {}\n", entry_file));
    bundle.push_str(&format!("// Modules: {}\n\n", modules.len()));

    // Write each module's source (with use declarations stripped)
    for (name, path, source) in &modules {
        bundle.push_str(&format!("// ── module: {} (from {}) ────\n", name, path));
        for line in source.lines() {
            let trimmed = line.trim();
            // Strip use declarations — the modules are already inlined
            if trimmed.starts_with("use ") && !trimmed.starts_with("use \"") {
                // Bare `use module` — skip, it's inlined
                continue;
            }
            if trimmed.starts_with("use \"") {
                // Path import `use "../path/module"` — skip, it's inlined
                continue;
            }
            bundle.push_str(line);
            bundle.push('\n');
        }
        bundle.push('\n');
    }

    // Write entry file source (strip use declarations)
    bundle.push_str("// ── entry ────\n");
    for line in entry_source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("use ") && !trimmed.starts_with("use \"") {
            continue;
        }
        if trimmed.starts_with("use \"") {
            continue;
        }
        bundle.push_str(line);
        bundle.push('\n');
    }

    // Phase 4: Validate the bundle parses correctly
    match octoflow_parser::parse(&bundle) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("warning: bundle has parse error (may still work): {}", e);
        }
    }

    // Phase 5: Write output
    std::fs::write(&output, &bundle)
        .map_err(|e| CliError::Io(format!("write {}: {}", output, e)))?;

    let total_lines = bundle.lines().count();
    let total_bytes = bundle.len();
    eprintln!(
        "Built {} ({} modules, {} lines, {:.1} KB) → {}",
        entry_file,
        modules.len(),
        total_lines,
        total_bytes as f64 / 1024.0,
        output
    );

    Ok(())
}

/// Recursively trace all `use` imports from a .flow file.
/// Modules are collected in dependency order (leaves first).
fn trace_imports(
    file_path: &str,
    base_dir: &str,
    visited: &mut std::collections::HashSet<String>,
    modules: &mut Vec<(String, String, String)>,
) -> Result<(), CliError> {
    let canonical = std::path::Path::new(file_path)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(file_path));
    let canon_str = canonical.to_string_lossy().into_owned();

    if visited.contains(&canon_str) {
        return Ok(()); // Already traced (circular import)
    }
    visited.insert(canon_str);

    let source = std::fs::read_to_string(file_path)
        .map_err(|e| CliError::Io(format!("{}: {}", file_path, e)))?;

    let program = octoflow_parser::parse(&source)
        .map_err(|e| CliError::Parse(format!("error in {}: {}", file_path, e)))?;

    for (stmt, _span) in &program.statements {
        if let octoflow_parser::ast::Statement::UseDecl { module } = stmt {
            // Resolve module path (same logic as runtime import_module)
            let module_path = if module.contains('/') || module.contains('\\') {
                // Path-style import: allow ".." traversal
                let p = std::path::Path::new(base_dir).join(format!("{}.flow", module));
                match p.canonicalize() {
                    Ok(cp) => cp.to_string_lossy().into_owned(),
                    Err(_) => {
                        // Try without canonicalize (file might not exist yet)
                        p.to_string_lossy().into_owned()
                    }
                }
            } else {
                // Bare name: resolve in base_dir
                let p = std::path::Path::new(base_dir).join(format!("{}.flow", module));
                p.to_string_lossy().into_owned()
            };

            if !std::path::Path::new(&module_path).exists() {
                // Try stdlib paths relative to the OctoFlow installation
                // Check common locations: stdlib/, ../stdlib/, ../../stdlib/
                let stdlib_candidates = [
                    format!("stdlib/{}.flow", module),
                    format!("stdlib/{}/{}.flow", module, module),
                    format!("../stdlib/{}.flow", module),
                    format!("../stdlib/{}/{}.flow", module, module),
                ];
                let mut found = false;
                for candidate in &stdlib_candidates {
                    if std::path::Path::new(candidate).exists() {
                        let module_base = std::path::Path::new(candidate)
                            .parent()
                            .map(|p| p.to_string_lossy().into_owned())
                            .unwrap_or_else(|| ".".to_string());
                        trace_imports(candidate, &module_base, visited, modules)?;
                        let mod_source = std::fs::read_to_string(candidate)
                            .map_err(|e| CliError::Io(format!("{}: {}", candidate, e)))?;
                        modules.push((module.clone(), candidate.clone(), mod_source));
                        found = true;
                        break;
                    }
                }
                if !found {
                    eprintln!("warning: module '{}' not found (expected {})", module, module_path);
                }
                continue;
            }

            // Recursively trace this module's imports
            let module_base = std::path::Path::new(&module_path)
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| ".".to_string());
            trace_imports(&module_path, &module_base, visited, modules)?;

            // Add this module (after its dependencies — depth-first order)
            let mod_source = std::fs::read_to_string(&module_path)
                .map_err(|e| CliError::Io(format!("{}: {}", module_path, e)))?;
            modules.push((module.clone(), module_path, mod_source));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_set_arg_valid() {
        let (name, val) = parse_set_arg("factor=3.5").unwrap();
        assert_eq!(name, "factor");
        assert!((val.as_float().unwrap() - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_parse_set_arg_negative() {
        let (name, val) = parse_set_arg("offset=-10.0").unwrap();
        assert_eq!(name, "offset");
        assert!((val.as_float().unwrap() - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_parse_set_arg_no_equals() {
        assert!(parse_set_arg("nope").is_err());
    }

    #[test]
    fn test_parse_set_arg_string_value() {
        let (name, val) = parse_set_arg("x=abc").unwrap();
        assert_eq!(name, "x");
        assert_eq!(val.as_str().unwrap(), "abc");
    }

    #[test]
    fn test_parse_run_args_file_only() {
        let args: Vec<String> = vec!["prog.flow".into()];
        let opts = parse_run_args(&args).unwrap();
        let (file, ov, watch) = (opts.flow_file, &opts.overrides, opts.watch);
        assert_eq!(file, "prog.flow");
        assert!(ov.scalars.is_empty());
        assert!(ov.input_path.is_none());
        assert!(ov.output_path.is_none());
        assert!(!watch);
    }

    #[test]
    fn test_parse_run_args_with_set() {
        let args: Vec<String> = vec![
            "prog.flow".into(), "--set".into(), "x=5.0".into(),
        ];
        let opts = parse_run_args(&args).unwrap();
        let (file, ov) = (opts.flow_file, &opts.overrides);
        assert_eq!(file, "prog.flow");
        assert!((ov.scalars["x"].as_float().unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_run_args_with_io() {
        let args: Vec<String> = vec![
            "prog.flow".into(), "-i".into(), "in.csv".into(),
            "-o".into(), "out.csv".into(),
        ];
        let opts = parse_run_args(&args).unwrap();
        let (file, ov) = (opts.flow_file, &opts.overrides);
        assert_eq!(file, "prog.flow");
        assert_eq!(ov.input_path.as_deref(), Some("in.csv"));
        assert_eq!(ov.output_path.as_deref(), Some("out.csv"));
    }

    #[test]
    fn test_parse_run_args_all_options() {
        let args: Vec<String> = vec![
            "prog.flow".into(), "--set".into(), "a=1.0".into(),
            "--set".into(), "b=2.0".into(), "-i".into(), "in.csv".into(),
            "-o".into(), "out.csv".into(),
        ];
        let opts = parse_run_args(&args).unwrap();
        let ov = &opts.overrides;
        assert_eq!(ov.scalars.len(), 2);
        assert!((ov.scalars["a"].as_float().unwrap() - 1.0).abs() < 1e-6);
        assert!((ov.scalars["b"].as_float().unwrap() - 2.0).abs() < 1e-6);
        assert_eq!(ov.input_path.as_deref(), Some("in.csv"));
        assert_eq!(ov.output_path.as_deref(), Some("out.csv"));
    }

    #[test]
    fn test_parse_run_args_empty() {
        let args: Vec<String> = vec![];
        assert!(parse_run_args(&args).is_err());
    }

    #[test]
    fn test_parse_run_args_unknown_flag() {
        let args: Vec<String> = vec!["prog.flow".into(), "--unknown".into()];
        assert!(parse_run_args(&args).is_err());
    }

    // --- Phase 13: --watch mode tests ---

    #[test]
    fn test_parse_run_args_with_watch_long() {
        let args: Vec<String> = vec!["prog.flow".into(), "--watch".into()];
        let opts = parse_run_args(&args).unwrap();
        let (file, watch) = (opts.flow_file, opts.watch);
        assert_eq!(file, "prog.flow");
        assert!(watch);
    }

    #[test]
    fn test_parse_run_args_with_watch_short() {
        let args: Vec<String> = vec!["prog.flow".into(), "-w".into()];
        let opts = parse_run_args(&args).unwrap();
        let (file, watch) = (opts.flow_file, opts.watch);
        assert_eq!(file, "prog.flow");
        assert!(watch);
    }

    #[test]
    fn test_parse_run_args_watch_with_set() {
        let args: Vec<String> = vec![
            "prog.flow".into(), "--set".into(), "x=5.0".into(), "-w".into(),
        ];
        let opts = parse_run_args(&args).unwrap();
        let (file, ov, watch) = (opts.flow_file, &opts.overrides, opts.watch);
        assert_eq!(file, "prog.flow");
        assert!((ov.scalars["x"].as_float().unwrap() - 5.0).abs() < 1e-6);
        assert!(watch);
    }

    #[test]
    fn test_parse_run_args_watch_with_all_options() {
        let args: Vec<String> = vec![
            "prog.flow".into(), "--watch".into(),
            "--set".into(), "a=1.0".into(),
            "-i".into(), "in.csv".into(), "-o".into(), "out.csv".into(),
        ];
        let opts = parse_run_args(&args).unwrap();
        let (file, ov, watch) = (opts.flow_file, &opts.overrides, opts.watch);
        assert_eq!(file, "prog.flow");
        assert!(watch);
        assert!((ov.scalars["a"].as_float().unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(ov.input_path.as_deref(), Some("in.csv"));
        assert_eq!(ov.output_path.as_deref(), Some("out.csv"));
    }

    #[test]
    fn test_collect_watched_files_nonexistent() {
        // Non-existent file — should still return it (watch will detect creation)
        let files = collect_watched_files("nonexistent_test_file.flow");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], "nonexistent_test_file.flow");
    }

    #[test]
    fn test_get_modification_times_nonexistent() {
        let files = vec!["nonexistent_test_file.flow".to_string()];
        let mtimes = get_modification_times(&files);
        assert_eq!(mtimes.len(), 1);
        assert!(mtimes[0].is_none());
    }

    #[test]
    fn test_get_modification_times_real_file() {
        // Use Cargo.toml as a known-existing file
        let files = vec!["Cargo.toml".to_string()];
        let mtimes = get_modification_times(&files);
        assert_eq!(mtimes.len(), 1);
        assert!(mtimes[0].is_some());
    }

    #[test]
    fn test_collect_watched_files_with_imports() {
        // Use the cinematic_photo.flow example which uses filters
        let files = collect_watched_files("examples/cinematic_photo.flow");
        assert!(files.len() >= 1);
        assert_eq!(files[0], "examples/cinematic_photo.flow");
        // If filters.flow exists as an import, it should be collected
        if files.len() > 1 {
            assert!(files[1].contains("filters.flow"));
        }
    }

    #[test]
    fn test_allow_net_flag() {
        let args = vec!["test.flow".into(), "--allow-net".into()];
        let opts = parse_run_args(&args).unwrap();
        let (file, ov) = (opts.flow_file, &opts.overrides);
        assert_eq!(file, "test.flow");
        assert!(ov.allow_net.is_allowed());
    }

    #[test]
    fn test_allow_read_scoped() {
        let args = vec!["test.flow".into(), "--allow-read=./data".into()];
        let opts = parse_run_args(&args).unwrap();
        match &opts.overrides.allow_read {
            PermScope::AllowScoped(paths) => {
                assert_eq!(paths.len(), 1);
                assert_eq!(paths[0].to_str().unwrap(), "./data");
            }
            other => panic!("Expected AllowScoped, got {:?}", other),
        }
    }

    #[test]
    fn test_allow_write_scoped() {
        let args = vec!["test.flow".into(), "--allow-write=./output".into()];
        let opts = parse_run_args(&args).unwrap();
        match &opts.overrides.allow_write {
            PermScope::AllowScoped(paths) => {
                assert_eq!(paths.len(), 1);
                assert_eq!(paths[0].to_str().unwrap(), "./output");
            }
            other => panic!("Expected AllowScoped, got {:?}", other),
        }
    }

    #[test]
    fn test_allow_read_multiple_scoped() {
        let args = vec![
            "test.flow".into(),
            "--allow-read=./data".into(),
            "--allow-read=./config".into(),
        ];
        let opts = parse_run_args(&args).unwrap();
        match &opts.overrides.allow_read {
            PermScope::AllowScoped(paths) => {
                assert_eq!(paths.len(), 2);
                assert_eq!(paths[0].to_str().unwrap(), "./data");
                assert_eq!(paths[1].to_str().unwrap(), "./config");
            }
            other => panic!("Expected AllowScoped, got {:?}", other),
        }
    }

    #[test]
    fn test_allow_net_scoped() {
        let args = vec!["test.flow".into(), "--allow-net=api.example.com".into()];
        let opts = parse_run_args(&args).unwrap();
        match &opts.overrides.allow_net {
            PermScope::AllowScoped(paths) => {
                assert_eq!(paths.len(), 1);
                assert_eq!(paths[0].to_str().unwrap(), "api.example.com");
            }
            other => panic!("Expected AllowScoped, got {:?}", other),
        }
    }

    #[test]
    fn test_allow_exec_scoped() {
        let args = vec!["test.flow".into(), "--allow-exec=./scripts".into()];
        let opts = parse_run_args(&args).unwrap();
        match &opts.overrides.allow_exec {
            PermScope::AllowScoped(paths) => {
                assert_eq!(paths.len(), 1);
                assert_eq!(paths[0].to_str().unwrap(), "./scripts");
            }
            other => panic!("Expected AllowScoped, got {:?}", other),
        }
    }

    // --- build tests ---

    #[test]
    fn test_trace_imports_no_imports() {
        // Create a temp file with no imports
        let dir = std::env::temp_dir();
        let file = dir.join("test_build_no_imports.flow");
        std::fs::write(&file, "print(\"hello\")").unwrap();

        let mut visited = std::collections::HashSet::new();
        let mut modules = Vec::new();
        let base = dir.to_string_lossy().into_owned();
        trace_imports(file.to_str().unwrap(), &base, &mut visited, &mut modules).unwrap();

        assert!(modules.is_empty());
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn test_trace_imports_with_import() {
        // Create two temp files: main imports helper
        let dir = std::env::temp_dir().join("test_build_with_import");
        std::fs::create_dir_all(&dir).ok();

        let helper = dir.join("helper.flow");
        std::fs::write(&helper, "fn greet(name)\n    print(\"Hello {name}\")\nend").unwrap();

        let main_file = dir.join("main.flow");
        std::fs::write(&main_file, "use helper\ngreet(\"World\")").unwrap();

        let mut visited = std::collections::HashSet::new();
        let mut modules = Vec::new();
        let base = dir.to_string_lossy().into_owned();
        trace_imports(main_file.to_str().unwrap(), &base, &mut visited, &mut modules).unwrap();

        assert_eq!(modules.len(), 1);
        assert_eq!(modules[0].0, "helper");
        assert!(modules[0].2.contains("fn greet"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_trace_imports_circular() {
        // Create two files that import each other — should not infinite loop
        let dir = std::env::temp_dir().join("test_build_circular");
        std::fs::create_dir_all(&dir).ok();

        let a = dir.join("a.flow");
        let b = dir.join("b.flow");
        std::fs::write(&a, "use b\nfn fa()\n    return 1.0\nend").unwrap();
        std::fs::write(&b, "use a\nfn fb()\n    return 2.0\nend").unwrap();

        let mut visited = std::collections::HashSet::new();
        let mut modules = Vec::new();
        let base = dir.to_string_lossy().into_owned();
        let result = trace_imports(a.to_str().unwrap(), &base, &mut visited, &mut modules);
        assert!(result.is_ok());
        // Both modules should be found (depth-first, no duplication)
        assert!(modules.len() <= 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_build_no_args() {
        // Should succeed (prints help)
        let result = build(&[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_missing_file() {
        let result = build(&["nonexistent.flow".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_produces_output() {
        let dir = std::env::temp_dir().join("test_build_output");
        std::fs::create_dir_all(&dir).ok();

        let source = dir.join("app.flow");
        std::fs::write(&source, "print(\"hello\")").unwrap();

        let output = dir.join("app.bundle.flow");
        let result = build(&[
            source.to_str().unwrap().to_string(),
            "-o".to_string(),
            output.to_str().unwrap().to_string(),
        ]);
        assert!(result.is_ok());
        assert!(output.exists());

        let content = std::fs::read_to_string(&output).unwrap();
        assert!(content.contains("OctoFlow Bundle"));
        assert!(content.contains("print(\"hello\")"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_build_strips_use_declarations() {
        let dir = std::env::temp_dir().join("test_build_strip_use");
        std::fs::create_dir_all(&dir).ok();

        let helper = dir.join("utils.flow");
        std::fs::write(&helper, "fn double(x)\n    return x * 2\nend").unwrap();

        let main_file = dir.join("app.flow");
        std::fs::write(&main_file, "use utils\nlet x = double(5.0)\nprint(\"{x}\")").unwrap();

        let output = dir.join("app.bundle.flow");
        let result = build(&[
            main_file.to_str().unwrap().to_string(),
            "-o".to_string(),
            output.to_str().unwrap().to_string(),
        ]);
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&output).unwrap();
        // use declarations should be stripped
        assert!(!content.contains("use utils"));
        // But the function should be inlined
        assert!(content.contains("fn double(x)"));
        // And the entry code should be there
        assert!(content.contains("let x = double(5.0)"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ─── Test runner tests ──────────────────────────────────────

    #[test]
    fn test_collect_flow_files_recursive() {
        // Create temp dir structure: tmp/a.flow, tmp/sub/b.flow, tmp/sub/deep/c.flow
        let dir = std::env::temp_dir().join("octoflow_test_collect");
        let _ = std::fs::remove_dir_all(&dir);
        let sub = dir.join("sub");
        let deep = sub.join("deep");
        std::fs::create_dir_all(&deep).unwrap();

        std::fs::write(dir.join("a.flow"), "fn test_a()\nend\n").unwrap();
        std::fs::write(sub.join("b.flow"), "fn test_b()\nend\n").unwrap();
        std::fs::write(deep.join("c.flow"), "fn test_c()\nend\n").unwrap();
        // Also a non-.flow file that should be ignored
        std::fs::write(dir.join("readme.txt"), "ignore me").unwrap();

        let mut files = Vec::new();
        collect_flow_files(dir.to_str().unwrap(), &mut files);

        assert_eq!(files.len(), 3, "should find 3 .flow files recursively");
        assert!(files.iter().any(|f| f.ends_with("a.flow")));
        assert!(files.iter().any(|f| f.ends_with("b.flow")));
        assert!(files.iter().any(|f| f.ends_with("c.flow")));
        // Should be sorted
        assert!(files[0] < files[1]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_collect_flow_files_empty_dir() {
        let dir = std::env::temp_dir().join("octoflow_test_collect_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut files = Vec::new();
        collect_flow_files(dir.to_str().unwrap(), &mut files);
        assert!(files.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_collect_flow_files_nonexistent_dir() {
        let mut files = Vec::new();
        collect_flow_files("/nonexistent/path/xyz", &mut files);
        assert!(files.is_empty());
    }

    #[test]
    fn test_run_tests_filter_requires_pattern() {
        let args = vec!["--filter".to_string()];
        let result = run_tests(&args);
        assert!(result.is_err());
    }
}
