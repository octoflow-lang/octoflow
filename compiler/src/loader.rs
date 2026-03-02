// loader.rs — Minimal Rust Bootstrap for Self-Hosted OctoFlow
//
// This is the < 500 line Rust loader that invokes eval.flow for compilation.
// eval.flow handles: lex → parse → preflight → eval → codegen → execute
//
// Usage: loader.rs reads argv, loads eval.flow, passes control to .flow compiler

use std::env;
use std::fs;
use std::process;
use std::path::PathBuf;

pub fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("OctoFlow Self-Hosted Compiler");
        eprintln!("Usage: octoflow-selfhosted <command> [options]");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  run <file.flow>     Compile and run .flow file");
        eprintln!("  compile <file.flow> Compile to SPIR-V (future)");
        eprintln!();
        process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "run" => {
            if args.len() < 3 {
                eprintln!("Error: 'run' requires a .flow file");
                eprintln!("Usage: octoflow-selfhosted run <file.flow>");
                process::exit(1);
            }

            let source_file = &args[2];
            run_selfhosted(source_file, &args[3..]);
        }
        "--version" => {
            println!("OctoFlow v1.16 (Self-Hosted via eval.flow)");
            println!("Compiler: 22,128 lines .flow");
            println!("Loader: {} lines Rust", count_lines(file!()));
        }
        "--help" | "help" => {
            println!("OctoFlow Self-Hosted Compiler");
            println!();
            println!("This is a minimal Rust loader (<500 lines) that invokes");
            println!("eval.flow (22,128 lines) for compilation and execution.");
            println!();
            println!("Commands:");
            println!("  run <file>          Compile and execute .flow file");
            println!("  --version           Show version info");
            println!("  --help              Show this help");
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            eprintln!("Try: octoflow-selfhosted --help");
            process::exit(1);
        }
    }
}

fn run_selfhosted(source_file: &str, extra_args: &[String]) {
    // Step 1: Locate eval.flow (relative to binary location)
    let exe_path = env::current_exe()
        .expect("Failed to get executable path");
    let exe_dir = exe_path.parent()
        .expect("Failed to get executable directory");

    // Assume stdlib is at: <exe_dir>/../../stdlib/compiler/eval.flow
    // This works for: target/release/octoflow-selfhosted
    let eval_flow_path = exe_dir
        .join("..")
        .join("..")
        .join("stdlib")
        .join("compiler")
        .join("eval.flow")
        .canonicalize()
        .unwrap_or_else(|_| exe_dir.join("..").join("..").join("stdlib").join("compiler").join("eval.flow"));

    if !eval_flow_path.exists() {
        eprintln!("Error: eval.flow not found at {:?}", eval_flow_path);
        eprintln!();
        eprintln!("Expected location: <exe_dir>/../../stdlib/compiler/eval.flow");
        eprintln!("This loader requires eval.flow to be present for self-hosted compilation.");
        process::exit(1);
    }

    // Step 2: Read source file
    let source_path = PathBuf::from(source_file);
    if !source_path.exists() {
        eprintln!("Error: Source file not found: {}", source_file);
        process::exit(1);
    }

    let _source_code = fs::read_to_string(&source_path)
        .unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", source_file, e);
            process::exit(1);
        });

    // Step 3: For now, call the existing Rust compiler with a note
    // TODO: Once loader is fully implemented, this will invoke eval.flow directly

    println!("=== OctoFlow Self-Hosted Mode ===");
    println!("Source: {}", source_file);
    println!("Compiler: eval.flow (22,128 lines)");
    println!("Loader: loader.rs ({} lines)", count_lines(file!()));
    println!();
    println!("Note: Full self-hosted execution not yet wired.");
    println!("      eval.flow exists and works (Stage 6 verified).");
    println!("      Next step: Invoke eval.flow::compile_and_run() from Rust.");
    println!();

    // Self-hosted compilation via eval.flow
    println!("Invoking eval.flow compiler...");
    println!();

    // Find octoflow binary (existing Rust compiler)
    let octoflow_bin = exe_dir.join("octoflow.exe");
    if !octoflow_bin.exists() {
        eprintln!("Error: octoflow binary not found at {:?}", octoflow_bin);
        process::exit(1);
    }

    // Set environment variable for eval.flow
    let abs_source = fs::canonicalize(&source_path)
        .unwrap_or_else(|e| {
            eprintln!("Error: Cannot resolve path {}: {}", source_file, e);
            process::exit(1);
        });

    let abs_source_str = abs_source.to_string_lossy().to_string();

    // Invoke: octoflow run stdlib/compiler/eval.flow --allow-read --allow-write --allow-ffi --allow-net --allow-exec
    // with FLOW_INPUT=<user's source file>
    let mut cmd = process::Command::new(&octoflow_bin);
    cmd.arg("run")
        .arg(&eval_flow_path)
        .arg("--allow-read")
        .arg("--allow-write")
        .arg("--allow-ffi")
        .arg("--allow-net")
        .arg("--allow-exec")
        .env("FLOW_INPUT", &abs_source_str);

    // Pass through extra args as FLOW_ARGS
    if !extra_args.is_empty() {
        let flow_args = extra_args.join(" ");
        cmd.env("FLOW_ARGS", flow_args);
    }

    // Execute and forward output
    let status = cmd.status()
        .unwrap_or_else(|e| {
            eprintln!("Error executing eval.flow: {}", e);
            process::exit(1);
        });

    process::exit(status.code().unwrap_or(1));
}

fn count_lines(file_path: &str) -> usize {
    let content = fs::read_to_string(file_path).unwrap_or_default();
    content.lines().count()
}
