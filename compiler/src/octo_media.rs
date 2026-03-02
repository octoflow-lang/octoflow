//! OctoMedia CLI — GPU-accelerated image processing powered by OctoFlow.
//!
//! Usage:
//!   `octo-media apply <filter.flow> <input> [<input>...] [-o <output>]`
//!   `octo-media presets` — list built-in filter presets

use octoflow_cli::CliError;
use octoflow_cli::Overrides;
use octoflow_cli::compiler;

use octoflow_parser::ast::{Expr, Program, Statement};

/// Built-in preset filter sources. Each is a complete .flow program
/// with placeholder tap/emit paths that get rewritten per-image.
fn get_preset(name: &str) -> Option<&'static str> {
    match name {
        "cinematic" => Some("\
stream img = tap(\"INPUT\")
stream s1 = img |> add(20.0) |> clamp(0.0, 255.0)
stream s2 = s1 |> subtract(128.0) |> multiply(1.2) |> add(128.0) |> clamp(0.0, 255.0)
stream s3 = s2 |> warm(25.0) |> clamp(0.0, 255.0)
stream out = s3 |> divide(255.0) |> pow(0.9) |> multiply(255.0) |> clamp(0.0, 255.0)
emit(out, \"OUTPUT\")
"),
        "brighten" => Some("\
stream img = tap(\"INPUT\")
stream out = img |> add(30.0) |> clamp(0.0, 255.0)
emit(out, \"OUTPUT\")
"),
        "darken" => Some("\
stream img = tap(\"INPUT\")
stream out = img |> subtract(30.0) |> clamp(0.0, 255.0)
emit(out, \"OUTPUT\")
"),
        "high_contrast" => Some("\
stream img = tap(\"INPUT\")
stream out = img |> subtract(128.0) |> multiply(1.4) |> add(128.0) |> clamp(0.0, 255.0)
emit(out, \"OUTPUT\")
"),
        "warm" => Some("\
stream img = tap(\"INPUT\")
stream out = img |> warm(25.0) |> clamp(0.0, 255.0)
emit(out, \"OUTPUT\")
"),
        "cool" => Some("\
stream img = tap(\"INPUT\")
stream out = img |> cool(25.0) |> clamp(0.0, 255.0)
emit(out, \"OUTPUT\")
"),
        "invert" => Some("\
stream img = tap(\"INPUT\")
stream out = img |> negate() |> add(255.0) |> clamp(0.0, 255.0)
emit(out, \"OUTPUT\")
"),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "apply" => cmd_apply(&args[2..]),
        "presets" => cmd_presets(),
        "help" | "--help" | "-h" => { print_usage(); Ok(()) }
        other => {
            eprintln!("Unknown command: '{}'. Use 'apply' or 'presets'.", other);
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn print_usage() {
    eprintln!("OctoMedia — GPU-accelerated image processing");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  octo-media apply <preset|filter.flow> <input.jpg> [-o output.png]");
    eprintln!("  octo-media apply <preset|filter.flow> <img1> <img2> ... -o <output_dir/>");
    eprintln!("  octo-media presets");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  octo-media apply cinematic photo.jpg");
    eprintln!("  octo-media apply warm photo.jpg -o warm_photo.png");
    eprintln!("  octo-media apply cinematic.flow photo.jpg -o graded.png");
    eprintln!("  octo-media apply cinematic a.jpg b.jpg c.jpg -o processed/");
}

/// `octo-media apply <filter.flow> <inputs...> [-o <output>]`
fn cmd_apply(args: &[String]) -> Result<(), CliError> {
    if args.is_empty() {
        return Err(CliError::Compile("apply requires a filter file and at least one input image".into()));
    }

    // Parse args: filter_file, inputs, optional -o output
    let filter_file = &args[0];
    let mut inputs: Vec<&str> = Vec::new();
    let mut output: Option<&str> = None;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "-o" && i + 1 < args.len() {
            output = Some(&args[i + 1]);
            i += 2;
        } else {
            inputs.push(&args[i]);
            i += 1;
        }
    }

    if inputs.is_empty() {
        return Err(CliError::Compile("no input images specified".into()));
    }

    // Validate inputs are images
    for input in &inputs {
        if !octoflow_cli::image_io::is_image_path(input) {
            return Err(CliError::Compile(format!(
                "'{}' is not a supported image format (use .png, .jpg, .jpeg)", input)));
        }
    }

    // Resolve filter: built-in preset or .flow file
    let (filter_program, filter_dir) = if let Some(preset_source) = get_preset(filter_file) {
        eprintln!("[preset: {}]", filter_file);
        let program = octoflow_parser::parse(preset_source)
            .map_err(|e| CliError::Parse(format!("preset '{}': {}", filter_file, e)))?;
        (program, ".".to_string())
    } else {
        let source = std::fs::read_to_string(filter_file)
            .map_err(|e| CliError::Io(format!("{}: {}", filter_file, e)))?;
        let program = octoflow_parser::parse(&source)
            .map_err(|e| CliError::Parse(format!("{}", e)))?;
        let dir = std::path::Path::new(filter_file)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());
        (program, dir)
    };

    // Batch mode: multiple inputs need a directory output
    let batch = inputs.len() > 1;
    if batch {
        if let Some(out) = output {
            // Ensure output directory exists
            std::fs::create_dir_all(out)
                .map_err(|e| CliError::Io(format!("cannot create output directory '{}': {}", out, e)))?;
        }
    }

    let mut total_files = 0;
    let mut total_elements = 0;

    for input_path in &inputs {
        let out_path = resolve_output_path(input_path, output, batch)?;

        eprintln!("{} -> {}", input_path, out_path);

        let program = rewrite_io_paths(&filter_program, input_path, &out_path);
        let (elements, _) = compiler::execute(&program, &filter_dir, &Overrides::default())?;

        total_files += 1;
        total_elements += elements;
    }

    eprintln!();
    eprintln!("Done: {} file{}, {} elements processed",
        total_files,
        if total_files == 1 { "" } else { "s" },
        total_elements);

    Ok(())
}

/// Determine the output path for a given input.
fn resolve_output_path(input: &str, output: Option<&str>, batch: bool) -> Result<String, CliError> {
    match output {
        Some(out) if batch => {
            // Batch mode: output is a directory, put file there
            let filename = std::path::Path::new(input)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".into());
            Ok(format!("{}/{}.png", out.trim_end_matches('/').trim_end_matches('\\'), filename))
        }
        Some(out) => {
            // Single file: use as-is
            Ok(out.to_string())
        }
        None => {
            // Auto-generate: input.jpg -> input_filtered.png
            let path = std::path::Path::new(input);
            let stem = path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".into());
            let parent = path.parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| ".".into());
            Ok(format!("{}/{}_filtered.png", parent, stem))
        }
    }
}

/// Rewrite a program's tap/emit paths for the given input/output image.
fn rewrite_io_paths(program: &Program, input: &str, output: &str) -> Program {
    let statements = program.statements.iter().map(|(stmt, span)| {
        let new_stmt = match stmt {
            Statement::StreamDecl { name, expr } => {
                Statement::StreamDecl {
                    name: name.clone(),
                    expr: rewrite_tap_expr(expr, input),
                }
            }
            Statement::Emit { expr, path: _ } => {
                Statement::Emit {
                    expr: expr.clone(),
                    path: output.to_string(),
                }
            }
            other => other.clone(),
        };
        (new_stmt, *span)
    }).collect();
    Program { statements }
}

/// Recursively replace Tap paths in an expression tree.
fn rewrite_tap_expr(expr: &Expr, new_path: &str) -> Expr {
    match expr {
        Expr::Tap { .. } => Expr::Tap { path: new_path.to_string() },
        Expr::Pipe { input, stages } => Expr::Pipe {
            input: Box::new(rewrite_tap_expr(input, new_path)),
            stages: stages.clone(),
        },
        other => other.clone(),
    }
}

/// `octo-media presets` — list built-in filter presets.
fn cmd_presets() -> Result<(), CliError> {
    println!("Built-in filter presets:");
    println!();
    println!("  cinematic    Warm color grading — brightness, contrast, warm tone, gamma");
    println!("  brighten     Simple brightness boost (+30)");
    println!("  darken       Reduce brightness (-30)");
    println!("  high_contrast  Strong contrast enhancement (1.4x)");
    println!("  warm         Warm color shift (R+25, B-25)");
    println!("  cool         Cool color shift (R-25, B+25)");
    println!("  invert       Color inversion");
    println!();
    println!("Usage: octo-media apply <preset_name> <input.jpg> [-o output.png]");
    println!("   or: octo-media apply <filter.flow> <input.jpg> [-o output.png]");
    Ok(())
}
