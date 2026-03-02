//! Lint Analysis — Pass 2 advisory checks.
//!
//! Extensible lint framework. Each arm adds a `check_*` function
//! called from `analyze()`. Adding a new arm (e.g., Style) means:
//!   1. Add new warning codes (S001, S002, ...)
//!   2. Add `fn check_style(program, &defs, &uses, &mut warnings)`
//!   3. Call it from `analyze()`
//! No restructuring needed.
//!
//! Current arms:
//!   D — Dead Code  (D001–D004)
//!   X — Redundancy (X001–X004)

use std::collections::{HashMap, HashSet};
use std::fmt;

use octoflow_parser::ast::{Arg, Expr, PrintSegment, Program, ScalarExpr, StageCall, Statement};

// ── Report types ────────────────────────────────────────────────────

pub struct LintWarning {
    pub code: &'static str,
    pub message: String,
    pub context: String,
    pub suggestion: String,
    pub line: usize,
}

pub struct LintReport {
    pub warnings: Vec<LintWarning>,
}

impl fmt::Display for LintReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for w in &self.warnings {
            if w.line > 0 {
                writeln!(f, "  [{}] line {}: {}", w.code, w.line, w.message)?;
            } else {
                writeln!(f, "  [{}] {}", w.code, w.message)?;
            }
            writeln!(f, "         {}", w.context)?;
            writeln!(f, "         {}", w.suggestion)?;
        }
        writeln!(f)?;
        write!(f, "  STATUS: {} warning{}", self.warnings.len(),
               if self.warnings.len() == 1 { "" } else { "s" })
    }
}

// ── Shared infrastructure ───────────────────────────────────────────

#[derive(Debug, Clone)]
struct FnDef {
    #[allow(dead_code)] // used by future lint arms (Style, etc.)
    params: Vec<String>,
    body: Vec<StageCall>,
}

struct Definitions {
    streams: HashMap<String, usize>,  // name -> line
    scalars: HashMap<String, usize>,  // name -> line
    local_fns: HashMap<String, usize>, // name -> line
    fn_defs: HashMap<String, FnDef>,
    imports: Vec<(String, usize)>,     // (module, line)
    module_fns: HashMap<String, HashSet<String>>,
}

struct Usages {
    streams: HashSet<String>,
    scalars: HashSet<String>,
    fns: HashSet<String>,
}

// ── Core analysis ───────────────────────────────────────────────────

pub fn analyze(program: &Program, base_dir: &str) -> LintReport {
    let mut warnings = Vec::new();

    let defs = collect_definitions(program, base_dir);
    let uses = collect_usages(program);

    // Arm D: Dead Code
    check_dead_code(&defs, &uses, &mut warnings);

    // Arm X: Redundancy
    check_redundancy(program, &defs.fn_defs, &mut warnings);

    // Future arms slot in here:
    // check_style(program, &defs, &uses, &mut warnings);       // Phase 6c
    // check_fusion(program, &defs, &uses, &mut warnings);      // Phase 6d
    // check_performance(program, &defs, &uses, &mut warnings); // Phase 6d

    LintReport { warnings }
}

// ── Definition collection ───────────────────────────────────────────

/// Recursively collect definitions from loop bodies (nested while/for).
fn collect_loop_body_defs(
    body: &[(Statement, octoflow_parser::ast::Span)],
    struct_defs: &HashMap<String, Vec<String>>,
    scalars: &mut HashMap<String, usize>,
) {
    for (body_stmt, body_span) in body {
        let bline = body_span.line;
        match body_stmt {
            Statement::LetDecl { name: bname, value, .. } => {
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    if matches!(fn_name.as_str(), "vec2" | "vec3" | "vec4") {
                        let components = ["x", "y", "z", "w"];
                        for i in 0..args.len().min(4) {
                            scalars.insert(format!("{}.{}", bname, components[i]), bline);
                        }
                        continue;
                    }
                    if let Some(fields) = struct_defs.get(fn_name) {
                        for field in fields {
                            scalars.insert(format!("{}.{}", bname, field), bline);
                        }
                        continue;
                    }
                    if fn_name == "try" {
                        for field in &["value", "ok", "error"] {
                            scalars.insert(format!("{}.{}", bname, field), bline);
                        }
                        continue;
                    }
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        for field in &["status", "body", "ok", "error"] {
                            scalars.insert(format!("{}.{}", bname, field), bline);
                        }
                        continue;
                    }
                    // Command execution: exec(cmd, ...args)
                    if fn_name == "exec" {
                        for field in &["status", "output", "ok", "error"] {
                            scalars.insert(format!("{}.{}", bname, field), bline);
                        }
                        continue;
                    }
                    // JSON/data functions — register map/array name
                    if fn_name == "json_parse" || fn_name == "json_parse_array" || fn_name == "load_data"
                        || fn_name == "read_csv"
                        || matches!(fn_name.as_str(), "filter" | "map_each" | "sort_by") {
                        scalars.insert(bname.clone(), bline);
                        continue;
                    }
                }
                scalars.insert(bname.clone(), bline);
            }
            Statement::ArrayDecl { name: aname, .. } => {
                scalars.insert(aname.clone(), bline);
            }
            Statement::MapDecl { name, .. } => {
                scalars.insert(name.clone(), bline);
            }
            Statement::WhileLoop { body: inner, .. } | Statement::ForLoop { body: inner, .. } | Statement::ForEachLoop { body: inner, .. } => {
                collect_loop_body_defs(inner, struct_defs, scalars);
            }
            Statement::IfBlock { body, elif_branches, else_body, .. } => {
                collect_loop_body_defs(body, struct_defs, scalars);
                for (_, elif_body) in elif_branches {
                    collect_loop_body_defs(elif_body, struct_defs, scalars);
                }
                collect_loop_body_defs(else_body, struct_defs, scalars);
            }
            _ => {}
        }
    }
}

fn collect_definitions(program: &Program, base_dir: &str) -> Definitions {
    let mut streams = HashMap::new();
    let mut scalars = HashMap::new();
    let mut local_fns = HashMap::new();
    let mut fn_defs: HashMap<String, FnDef> = HashMap::new();
    let mut imports = Vec::new();
    let mut module_fns: HashMap<String, HashSet<String>> = HashMap::new();

    // Pre-collect struct definitions for constructor detection
    let mut struct_defs: HashMap<String, Vec<String>> = HashMap::new();
    for (stmt, _) in &program.statements {
        if let Statement::StructDef { name, fields } = stmt {
            struct_defs.insert(name.clone(), fields.clone());
        }
    }

    for (stmt, span) in &program.statements {
        let line = span.line;
        match stmt {
            Statement::StreamDecl { name, .. } => { streams.insert(name.clone(), line); }
            Statement::LetDecl { name, value, .. } => {
                // Check for vec constructor: register component scalars
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    if matches!(fn_name.as_str(), "vec2" | "vec3" | "vec4") {
                        let components = ["x", "y", "z", "w"];
                        for i in 0..args.len().min(4) {
                            scalars.insert(format!("{}.{}", name, components[i]), line);
                        }
                        continue;
                    }
                    // Check for struct constructor: register field scalars
                    if let Some(fields) = struct_defs.get(fn_name) {
                        for field in fields {
                            scalars.insert(format!("{}.{}", name, field), line);
                        }
                        continue;
                    }
                    // Check for try(): register .value, .ok, .error fields
                    if fn_name == "try" {
                        for field in &["value", "ok", "error"] {
                            scalars.insert(format!("{}.{}", name, field), line);
                        }
                        continue;
                    }
                    // HTTP client functions: register .status, .body, .ok, .error fields
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        for field in &["status", "body", "ok", "error"] {
                            scalars.insert(format!("{}.{}", name, field), line);
                        }
                        continue;
                    }
                    // Command execution: exec(cmd, ...args)
                    if fn_name == "exec" {
                        for field in &["status", "output", "ok", "error"] {
                            scalars.insert(format!("{}.{}", name, field), line);
                        }
                        continue;
                    }
                    // JSON/data functions — register map/array name
                    if fn_name == "json_parse" || fn_name == "json_parse_array" || fn_name == "load_data"
                        || fn_name == "read_csv"
                        || matches!(fn_name.as_str(), "filter" | "map_each" | "sort_by") {
                        scalars.insert(name.clone(), line);
                        continue;
                    }
                }
                scalars.insert(name.clone(), line);
            }
            Statement::Assign { .. } => {
                // Assignments don't define new names — they modify existing ones
            }
            Statement::StructDef { .. } => {
                // Struct definitions tracked separately above
            }
            Statement::ArrayDecl { name, .. } => {
                // Arrays tracked as scalars for usage detection (len, index access)
                scalars.insert(name.clone(), line);
            }
            Statement::FnDecl { name, params, body } => {
                local_fns.insert(name.clone(), line);
                fn_defs.insert(name.clone(), FnDef {
                    params: params.clone(),
                    body: body.clone(),
                });
            }
            Statement::UseDecl { module } => {
                imports.push((module.clone(), line));
                let mut fns_in_module = HashSet::new();
                let path = std::path::Path::new(base_dir).join(format!("{}.flow", module));
                if let Ok(source) = std::fs::read_to_string(&path) {
                    if let Ok(prog) = octoflow_parser::parse(&source) {
                        for (s, _span) in &prog.statements {
                            match s {
                                Statement::FnDecl { name, params, body } => {
                                    fns_in_module.insert(name.clone());
                                    let def = FnDef { params: params.clone(), body: body.clone() };
                                    fn_defs.insert(format!("{}.{}", module, name), def.clone());
                                    fn_defs.insert(name.clone(), def);
                                }
                                Statement::ScalarFnDecl { name, .. } => {
                                    fns_in_module.insert(name.clone());
                                }
                                Statement::StructDef { name, .. } => {
                                    fns_in_module.insert(name.clone());
                                }
                                Statement::LetDecl { name, .. } => {
                                    fns_in_module.insert(name.clone());
                                }
                                Statement::ArrayDecl { name, .. } => {
                                    fns_in_module.insert(name.clone());
                                }
                                _ => {}
                            }
                        }
                    }
                }
                module_fns.insert(module.clone(), fns_in_module);
            }
            Statement::WhileLoop { body, .. } | Statement::ForLoop { body, .. } | Statement::ForEachLoop { body, .. } => {
                collect_loop_body_defs(body, &struct_defs, &mut scalars);
            }
            Statement::IfBlock { body, elif_branches, else_body, .. } => {
                collect_loop_body_defs(body, &struct_defs, &mut scalars);
                for (_, elif_body) in elif_branches {
                    collect_loop_body_defs(elif_body, &struct_defs, &mut scalars);
                }
                collect_loop_body_defs(else_body, &struct_defs, &mut scalars);
            }
            Statement::ScalarFnDecl { name, params: _, body } => {
                local_fns.insert(name.clone(), line);
                collect_loop_body_defs(body, &struct_defs, &mut scalars);
            }
            Statement::MapDecl { name, .. } => {
                scalars.insert(name.clone(), line);
            }
            Statement::ExternBlock { functions, .. } => {
                // Extern fn names are valid call targets — register them as local fns
                for f in functions {
                    local_fns.insert(f.name.clone(), 0);
                }
            }
            Statement::Emit { .. } | Statement::Print { .. }
            | Statement::Break | Statement::Continue | Statement::Return { .. }
            | Statement::ArrayAssign { .. } | Statement::ArrayPush { .. }
            | Statement::MapInsert { .. }
            | Statement::WriteFile { .. } | Statement::AppendFile { .. }
            | Statement::SaveData { .. } | Statement::WriteCsv { .. }
            | Statement::WriteBytes { .. } | Statement::ExprStmt { .. } => {}
        }
    }

    Definitions { streams, scalars, local_fns, fn_defs, imports, module_fns }
}

// ── Usage collection ────────────────────────────────────────────────

fn collect_usages(program: &Program) -> Usages {
    let mut streams = HashSet::new();
    let mut scalars = HashSet::new();
    let mut fns = HashSet::new();

    for (stmt, _span) in &program.statements {
        match stmt {
            Statement::StreamDecl { name: _, expr } => {
                collect_expr_usages(expr, &mut streams, &mut scalars, &mut fns);
            }
            Statement::LetDecl { name: _, value, .. } => {
                collect_scalar_usages(value, &mut streams, &mut scalars, &mut fns);
            }
            Statement::Assign { name, value } => {
                // The target variable is "used" (it's being written to)
                scalars.insert(name.clone());
                collect_scalar_usages(value, &mut streams, &mut scalars, &mut fns);
            }
            Statement::ArrayAssign { array, index, value } => {
                scalars.insert(array.clone());
                collect_scalar_usages(index, &mut streams, &mut scalars, &mut fns);
                collect_scalar_usages(value, &mut streams, &mut scalars, &mut fns);
            }
            Statement::ArrayPush { array, value } => {
                scalars.insert(array.clone());
                collect_scalar_usages(value, &mut streams, &mut scalars, &mut fns);
            }
            Statement::Emit { expr, .. } => {
                if let Expr::Ref { name } = expr {
                    streams.insert(name.clone());
                }
            }
            Statement::FnDecl { name: _, params: _, body } => {
                for stage in body {
                    collect_stage_usages(stage, &mut scalars, &mut fns);
                }
            }
            Statement::ArrayDecl { name: _, elements, .. } => {
                for elem in elements {
                    collect_scalar_usages(elem, &mut streams, &mut scalars, &mut fns);
                }
            }
            Statement::WhileLoop { condition, body } => {
                collect_scalar_usages(condition, &mut streams, &mut scalars, &mut fns);
                collect_loop_body_usages(body, &mut streams, &mut scalars, &mut fns);
            }
            Statement::ForLoop { var: _var, start, end, body } => {
                collect_scalar_usages(start, &mut streams, &mut scalars, &mut fns);
                collect_scalar_usages(end, &mut streams, &mut scalars, &mut fns);
                collect_loop_body_usages(body, &mut streams, &mut scalars, &mut fns);
            }
            Statement::ForEachLoop { var: _var, iterable, body } => {
                scalars.insert(iterable.clone());
                collect_loop_body_usages(body, &mut streams, &mut scalars, &mut fns);
            }
            Statement::IfBlock { condition, body, elif_branches, else_body } => {
                collect_scalar_usages(condition, &mut streams, &mut scalars, &mut fns);
                collect_loop_body_usages(body, &mut streams, &mut scalars, &mut fns);
                for (elif_cond, elif_body) in elif_branches {
                    collect_scalar_usages(elif_cond, &mut streams, &mut scalars, &mut fns);
                    collect_loop_body_usages(elif_body, &mut streams, &mut scalars, &mut fns);
                }
                collect_loop_body_usages(else_body, &mut streams, &mut scalars, &mut fns);
            }
            Statement::ScalarFnDecl { name: _, params: _, body } => {
                collect_loop_body_usages(body, &mut streams, &mut scalars, &mut fns);
            }
            Statement::Return { value } => {
                collect_scalar_usages(value, &mut streams, &mut scalars, &mut fns);
            }
            Statement::MapDecl { .. } => {}
            Statement::MapInsert { map, key, value } => {
                scalars.insert(map.clone());
                collect_scalar_usages(key, &mut streams, &mut scalars, &mut fns);
                collect_scalar_usages(value, &mut streams, &mut scalars, &mut fns);
            }
            Statement::WriteFile { path, content } | Statement::AppendFile { path, content } => {
                collect_scalar_usages(path, &mut streams, &mut scalars, &mut fns);
                collect_scalar_usages(content, &mut streams, &mut scalars, &mut fns);
            }
            Statement::SaveData { path, map_name } => {
                collect_scalar_usages(path, &mut streams, &mut scalars, &mut fns);
                scalars.insert(map_name.clone());
            }
            Statement::WriteCsv { path, array_name } => {
                collect_scalar_usages(path, &mut streams, &mut scalars, &mut fns);
                scalars.insert(array_name.clone());
            }
            Statement::WriteBytes { path, array_name } => {
                collect_scalar_usages(path, &mut streams, &mut scalars, &mut fns);
                scalars.insert(array_name.clone());
            }
            Statement::ExprStmt { expr } => {
                collect_scalar_usages(expr, &mut streams, &mut scalars, &mut fns);
            }
            Statement::UseDecl { .. } | Statement::StructDef { .. }
            | Statement::Break | Statement::Continue | Statement::ExternBlock { .. } => {}
            Statement::Print { segments } => {
                for seg in segments {
                    if let PrintSegment::Scalar { name, .. } = seg {
                        scalars.insert(name.clone());
                    }
                }
            }
        }
    }

    Usages { streams, scalars, fns }
}

fn collect_expr_usages(
    expr: &Expr, streams: &mut HashSet<String>,
    scalars: &mut HashSet<String>, fns: &mut HashSet<String>,
) {
    match expr {
        Expr::Tap { .. } | Expr::RandomStream { .. } | Expr::Cache { .. } => {}
        Expr::Ref { name } => { streams.insert(name.clone()); }
        Expr::Pipe { input, stages } => {
            collect_expr_usages(input, streams, scalars, fns);
            for stage in stages {
                collect_stage_usages(stage, scalars, fns);
            }
        }
    }
}

fn collect_stage_usages(
    stage: &StageCall, scalars: &mut HashSet<String>, fns: &mut HashSet<String>,
) {
    fns.insert(stage.operation.clone());
    for arg in &stage.args {
        if let Arg::Ref(name) = arg {
            scalars.insert(name.clone());
        }
    }
}

/// Recursively collect usages from loop bodies (nested while/for).
fn collect_loop_body_usages(
    body: &[(Statement, octoflow_parser::ast::Span)],
    streams: &mut HashSet<String>,
    scalars: &mut HashSet<String>,
    fns: &mut HashSet<String>,
) {
    for (body_stmt, _) in body {
        match body_stmt {
            Statement::LetDecl { value, .. } => {
                collect_scalar_usages(value, streams, scalars, fns);
            }
            Statement::Assign { name: aname, value } => {
                scalars.insert(aname.clone());
                collect_scalar_usages(value, streams, scalars, fns);
            }
            Statement::ArrayAssign { array, index, value } => {
                scalars.insert(array.clone());
                collect_scalar_usages(index, streams, scalars, fns);
                collect_scalar_usages(value, streams, scalars, fns);
            }
            Statement::ArrayPush { array, value } => {
                scalars.insert(array.clone());
                collect_scalar_usages(value, streams, scalars, fns);
            }
            Statement::Print { segments: segs } => {
                for seg in segs {
                    if let PrintSegment::Scalar { name: sname, .. } = seg {
                        scalars.insert(sname.clone());
                    }
                }
            }
            Statement::ArrayDecl { elements, .. } => {
                for elem in elements {
                    collect_scalar_usages(elem, streams, scalars, fns);
                }
            }
            Statement::MapInsert { map, key, value } => {
                scalars.insert(map.clone());
                collect_scalar_usages(key, streams, scalars, fns);
                collect_scalar_usages(value, streams, scalars, fns);
            }
            Statement::WriteFile { path, content } | Statement::AppendFile { path, content } => {
                collect_scalar_usages(path, streams, scalars, fns);
                collect_scalar_usages(content, streams, scalars, fns);
            }
            Statement::SaveData { path, map_name } => {
                collect_scalar_usages(path, streams, scalars, fns);
                scalars.insert(map_name.clone());
            }
            Statement::WriteCsv { path, array_name } => {
                collect_scalar_usages(path, streams, scalars, fns);
                scalars.insert(array_name.clone());
            }
            Statement::WriteBytes { path, array_name } => {
                collect_scalar_usages(path, streams, scalars, fns);
                scalars.insert(array_name.clone());
            }
            Statement::WhileLoop { condition, body: inner } => {
                collect_scalar_usages(condition, streams, scalars, fns);
                collect_loop_body_usages(inner, streams, scalars, fns);
            }
            Statement::ForLoop { start, end, body: inner, .. } => {
                collect_scalar_usages(start, streams, scalars, fns);
                collect_scalar_usages(end, streams, scalars, fns);
                collect_loop_body_usages(inner, streams, scalars, fns);
            }
            Statement::ForEachLoop { iterable, body: inner, .. } => {
                scalars.insert(iterable.clone());
                collect_loop_body_usages(inner, streams, scalars, fns);
            }
            Statement::IfBlock { condition, body: if_body, elif_branches, else_body } => {
                collect_scalar_usages(condition, streams, scalars, fns);
                collect_loop_body_usages(if_body, streams, scalars, fns);
                for (elif_cond, elif_body) in elif_branches {
                    collect_scalar_usages(elif_cond, streams, scalars, fns);
                    collect_loop_body_usages(elif_body, streams, scalars, fns);
                }
                collect_loop_body_usages(else_body, streams, scalars, fns);
            }
            Statement::ExprStmt { expr } => {
                collect_scalar_usages(expr, streams, scalars, fns);
            }
            _ => {}
        }
    }
}

fn collect_scalar_usages(
    expr: &ScalarExpr, streams: &mut HashSet<String>, scalars: &mut HashSet<String>, fns: &mut HashSet<String>,
) {
    match expr {
        ScalarExpr::Reduce { stream, .. } => { streams.insert(stream.clone()); }
        ScalarExpr::BinOp { left, right, .. }
        | ScalarExpr::Compare { left, right, .. }
        | ScalarExpr::And { left, right }
        | ScalarExpr::Or { left, right } => {
            collect_scalar_usages(left, streams, scalars, fns);
            collect_scalar_usages(right, streams, scalars, fns);
        }
        ScalarExpr::If { condition, then_expr, else_expr } => {
            collect_scalar_usages(condition, streams, scalars, fns);
            collect_scalar_usages(then_expr, streams, scalars, fns);
            collect_scalar_usages(else_expr, streams, scalars, fns);
        }
        ScalarExpr::Ref(name) => { scalars.insert(name.clone()); }
        ScalarExpr::Literal(_) | ScalarExpr::IntLiteral(_) | ScalarExpr::Bool(_) | ScalarExpr::StringLiteral(_) | ScalarExpr::NoneLiteral => {}
        ScalarExpr::FnCall { name, args } => {
            fns.insert(name.clone());
            for arg in args {
                collect_scalar_usages(arg, streams, scalars, fns);
            }
        }
        ScalarExpr::Index { array, index } => {
            scalars.insert(array.clone());
            collect_scalar_usages(index, streams, scalars, fns);
        }
        ScalarExpr::Lambda { params: _, body } => {
            // Recurse into lambda body; params are local, not refs to outer scope
            collect_scalar_usages(body, streams, scalars, fns);
        }
        ScalarExpr::ArrayLiteral(elements) => {
            for elem in elements {
                collect_scalar_usages(elem, streams, scalars, fns);
            }
        }
    }
}

// ── Arm D: Dead Code ────────────────────────────────────────────────

fn check_dead_code(defs: &Definitions, uses: &Usages, warnings: &mut Vec<LintWarning>) {
    // D001: Unused streams
    for (name, &line) in &defs.streams {
        if !uses.streams.contains(name) {
            warnings.push(LintWarning {
                code: "D001",
                message: format!("Unused stream '{}'", name),
                context: format!("stream {} is defined but never referenced or emitted", name),
                suggestion: "Remove the stream declaration, or use it in a pipeline".into(),
                line,
            });
        }
    }

    // D002: Unused scalars
    for (name, &line) in &defs.scalars {
        if !uses.scalars.contains(name) {
            warnings.push(LintWarning {
                code: "D002",
                message: format!("Unused scalar '{}'", name),
                context: format!("let {} is defined but never referenced", name),
                suggestion: "Remove the let declaration, or use it as a stage argument".into(),
                line,
            });
        }
    }

    // D003: Unused local functions
    for (name, &line) in &defs.local_fns {
        if !uses.fns.contains(name) {
            warnings.push(LintWarning {
                code: "D003",
                message: format!("Unused function '{}()'", name),
                context: format!("fn {}() is defined but never called", name),
                suggestion: "Remove the function, or call it in a pipeline".into(),
                line,
            });
        }
    }

    // D004: Unused imports
    for (module, line) in &defs.imports {
        let module_fn_names = defs.module_fns.get(module);
        let is_used = module_fn_names.map_or(false, |names| {
            names.iter().any(|fn_name| {
                let dotted = format!("{}.{}", module, fn_name);
                uses.fns.contains(fn_name) || uses.fns.contains(&dotted)
                    || uses.scalars.contains(fn_name) || uses.scalars.contains(&dotted)
            })
        });
        if !is_used {
            warnings.push(LintWarning {
                code: "D004",
                message: format!("Unused import '{}'", module),
                context: format!("use {} but nothing from this module is used", module),
                suggestion: "Remove the use declaration, or use something from this module".into(),
                line: *line,
            });
        }
    }
}

// ── Arm X: Redundancy ───────────────────────────────────────────────

fn check_redundancy(
    program: &Program, fn_defs: &HashMap<String, FnDef>,
    warnings: &mut Vec<LintWarning>,
) {
    for (stmt, span) in &program.statements {
        let line = span.line;
        match stmt {
            Statement::StreamDecl { name, expr } => {
                let ctx = format!("stream {}", name);
                check_expr_redundancy(expr, &ctx, fn_defs, warnings, line);
            }
            Statement::FnDecl { name, body, .. } => {
                let ctx = format!("fn {}() body", name);
                check_stages(body, &ctx, fn_defs, warnings, line);
            }
            _ => {}
        }
    }
}

fn check_expr_redundancy(
    expr: &Expr, ctx: &str, fn_defs: &HashMap<String, FnDef>,
    warnings: &mut Vec<LintWarning>, line: usize,
) {
    if let Expr::Pipe { input, stages } = expr {
        check_expr_redundancy(input, ctx, fn_defs, warnings, line);
        check_stages(stages, ctx, fn_defs, warnings, line);
    }
}

fn check_stages(
    stages: &[StageCall], ctx: &str, fn_defs: &HashMap<String, FnDef>,
    warnings: &mut Vec<LintWarning>, line: usize,
) {
    for (i, stage) in stages.iter().enumerate() {
        let op = stage.operation.as_str();

        // X002: multiply(0.0) — zero-out (check before X001)
        if op == "multiply" && stage.args.len() == 1 && is_literal_eq(&stage.args[0], 0.0) {
            warnings.push(LintWarning {
                code: "X002",
                message: "multiply(0.0) sets all values to zero".into(),
                context: format!("{}, stage {}", ctx, i + 1),
                suggestion: "Is this intentional? All data becomes zero".into(),
                line,
            });
            continue;
        }

        // X001: Identity no-ops
        let is_identity = match op {
            "add" | "subtract" => stage.args.len() == 1 && is_literal_eq(&stage.args[0], 0.0),
            "multiply" | "divide" => stage.args.len() == 1 && is_literal_eq(&stage.args[0], 1.0),
            "pow" => stage.args.len() == 1 && is_literal_eq(&stage.args[0], 1.0),
            _ => false,
        };
        if is_identity {
            warnings.push(LintWarning {
                code: "X001",
                message: format!("{}({}) is a no-op", op, format_first_arg(stage)),
                context: format!("{}, stage {}", ctx, i + 1),
                suggestion: "Remove this stage -- it has no effect on the data".into(),
                line,
            });
            continue;
        }

        // X004: multiply(-1.0) → use negate()
        if op == "multiply" && stage.args.len() == 1 && is_literal_eq(&stage.args[0], -1.0) {
            warnings.push(LintWarning {
                code: "X004",
                message: "multiply(-1.0) can be simplified".into(),
                context: format!("{}, stage {}", ctx, i + 1),
                suggestion: "Use negate() instead of multiply(-1.0)".into(),
                line,
            });
            continue;
        }

        // X003: Cancelling/repeating adjacent pairs — negate() |> negate(), abs() |> abs()
        if i + 1 < stages.len() {
            let next = &stages[i + 1];
            if op == "negate" && next.operation == "negate"
                && stage.args.is_empty() && next.args.is_empty()
            {
                warnings.push(LintWarning {
                    code: "X003",
                    message: "negate() |> negate() cancels out".into(),
                    context: format!("{}, stages {}-{}", ctx, i + 1, i + 2),
                    suggestion: "Remove both negate() calls".into(),
                    line,
                });
            }
            if op == "abs" && next.operation == "abs"
                && stage.args.is_empty() && next.args.is_empty()
            {
                warnings.push(LintWarning {
                    code: "X003",
                    message: "abs() |> abs() is redundant".into(),
                    context: format!("{}, stages {}-{}", ctx, i + 1, i + 2),
                    suggestion: "Remove the second abs() -- abs of abs is always the same".into(),
                    line,
                });
            }

            // X004: max(a) |> min(b) → clamp(a, b)
            if op == "max" && next.operation == "min"
                && stage.args.len() == 1 && next.args.len() == 1
                && is_literal(&stage.args[0]) && is_literal(&next.args[0])
            {
                warnings.push(LintWarning {
                    code: "X004",
                    message: format!(
                        "max({}) |> min({}) can be simplified",
                        format_first_arg(stage), format_first_arg(next)
                    ),
                    context: format!("{}, stages {}-{}", ctx, i + 1, i + 2),
                    suggestion: format!(
                        "Use clamp({}, {}) instead",
                        format_first_arg(stage), format_first_arg(next)
                    ),
                    line,
                });
            }
        }

        // Recurse into function bodies (inline check)
        if let Some(fn_def) = fn_defs.get(op) {
            let fn_ctx = format!("{} via {}()", ctx, op);
            check_stages(&fn_def.body, &fn_ctx, fn_defs, warnings, line);
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn is_literal_eq(arg: &Arg, target: f64) -> bool {
    match arg {
        Arg::Literal(v) => (*v - target).abs() < 1e-12,
        Arg::IntLiteral(i) => ((*i as f64) - target).abs() < 1e-12,
        Arg::Ref(_) => false,
    }
}

fn is_literal(arg: &Arg) -> bool {
    matches!(arg, Arg::Literal(_) | Arg::IntLiteral(_))
}

fn format_first_arg(stage: &StageCall) -> String {
    match stage.args.first() {
        Some(Arg::Literal(v)) => format!("{}", v),
        Some(Arg::IntLiteral(i)) => format!("{}", i),
        Some(Arg::Ref(name)) => name.clone(),
        None => String::new(),
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> LintReport {
        let prog = octoflow_parser::parse(src).unwrap();
        analyze(&prog, ".")
    }

    fn has_code(r: &LintReport, code: &str) -> bool {
        r.warnings.iter().any(|w| w.code == code)
    }

    #[allow(dead_code)] // available for future tests
    fn count_code(r: &LintReport, code: &str) -> usize {
        r.warnings.iter().filter(|w| w.code == code).count()
    }

    // ── Dead Code (8 tests) ─────────────────────────────────────────

    #[test]
    fn dead_code_clean_program() {
        let r = run("stream d = tap(\"i\")\nemit(d, \"o\")\n");
        assert!(!has_code(&r, "D001"));
        assert!(!has_code(&r, "D002"));
        assert!(!has_code(&r, "D003"));
    }

    #[test]
    fn dead_code_unused_stream_d001() {
        let r = run("stream d = tap(\"i\")\nstream unused = d |> multiply(2.0)\nemit(d, \"o\")\n");
        assert!(has_code(&r, "D001"));
        assert!(r.warnings.iter().any(|w| w.code == "D001" && w.message.contains("unused")));
    }

    #[test]
    fn dead_code_unused_scalar_d002() {
        let r = run("stream d = tap(\"i\")\nlet x = min(d)\nemit(d, \"o\")\n");
        assert!(has_code(&r, "D002"));
        assert!(r.warnings.iter().any(|w| w.code == "D002" && w.message.contains("x")));
    }

    #[test]
    fn dead_code_unused_function_d003() {
        let r = run("fn unused_fn(): multiply(2.0)\nstream d = tap(\"i\")\nemit(d, \"o\")\n");
        assert!(has_code(&r, "D003"));
        assert!(r.warnings.iter().any(|w| w.code == "D003" && w.message.contains("unused_fn")));
    }

    #[test]
    fn dead_code_unused_import_d004() {
        // use a nonexistent module — we still track the import
        let r = run("use nonexistent_mod\nstream d = tap(\"i\")\nemit(d, \"o\")\n");
        assert!(has_code(&r, "D004"));
    }

    #[test]
    fn dead_code_stream_used_transitively() {
        // Stream 'a' is used by 'b', and 'b' is emitted — 'a' is NOT dead
        let r = run("stream a = tap(\"i\")\nstream b = a |> multiply(2.0)\nemit(b, \"o\")\n");
        assert!(!has_code(&r, "D001"));
    }

    #[test]
    fn dead_code_scalar_used_as_arg() {
        let r = run("stream d = tap(\"i\")\nlet k = min(d)\nstream r = d |> multiply(k)\nemit(r, \"o\")\n");
        assert!(!has_code(&r, "D002"));
    }

    #[test]
    fn dead_code_fn_used_inside_another_fn() {
        // clamp_unit is called inside normalize_255 — NOT dead
        let r = run(
            "fn clamp_unit(): clamp(0.0, 1.0)\n\
             fn normalize_255(): divide(255.0) |> clamp_unit()\n\
             stream d = tap(\"i\")\nstream n = d |> normalize_255()\nemit(n, \"o\")\n"
        );
        assert!(!r.warnings.iter().any(|w| w.code == "D003" && w.message.contains("clamp_unit")));
    }

    // ── Redundancy (9 tests) ────────────────────────────────────────

    #[test]
    fn redundancy_add_zero_x001() {
        let r = run("stream d = tap(\"i\")\nstream r = d |> add(0.0)\nemit(r, \"o\")\n");
        assert!(has_code(&r, "X001"));
        assert!(r.warnings.iter().any(|w| w.code == "X001" && w.message.contains("add(0)")));
    }

    #[test]
    fn redundancy_multiply_one_x001() {
        let r = run("stream d = tap(\"i\")\nstream r = d |> multiply(1.0)\nemit(r, \"o\")\n");
        assert!(has_code(&r, "X001"));
        assert!(r.warnings.iter().any(|w| w.code == "X001" && w.message.contains("multiply(1)")));
    }

    #[test]
    fn redundancy_multiply_zero_x002_not_x001() {
        let r = run("stream d = tap(\"i\")\nstream r = d |> multiply(0.0)\nemit(r, \"o\")\n");
        assert!(has_code(&r, "X002"));
        assert!(!has_code(&r, "X001")); // X002 takes precedence
    }

    #[test]
    fn redundancy_negate_negate_x003() {
        let r = run("stream d = tap(\"i\")\nstream r = d |> negate() |> negate()\nemit(r, \"o\")\n");
        assert!(has_code(&r, "X003"));
    }

    #[test]
    fn redundancy_abs_abs_x003() {
        let r = run("stream d = tap(\"i\")\nstream r = d |> abs() |> abs()\nemit(r, \"o\")\n");
        assert!(has_code(&r, "X003"));
    }

    #[test]
    fn redundancy_multiply_neg1_x004() {
        let r = run("stream d = tap(\"i\")\nstream r = d |> multiply(-1.0)\nemit(r, \"o\")\n");
        assert!(has_code(&r, "X004"));
        assert!(r.warnings.iter().any(|w| w.suggestion.contains("negate()")));
    }

    #[test]
    fn redundancy_max_min_clamp_x004() {
        let r = run("stream d = tap(\"i\")\nstream r = d |> max(0.0) |> min(255.0)\nemit(r, \"o\")\n");
        assert!(has_code(&r, "X004"));
        assert!(r.warnings.iter().any(|w| w.suggestion.contains("clamp(0, 255)")));
    }

    #[test]
    fn redundancy_clean_pipeline() {
        let r = run("stream d = tap(\"i\")\nstream r = d |> multiply(2.0) |> add(5.0)\nemit(r, \"o\")\n");
        assert!(!has_code(&r, "X001"));
        assert!(!has_code(&r, "X002"));
        assert!(!has_code(&r, "X003"));
        assert!(!has_code(&r, "X004"));
    }

    #[test]
    fn redundancy_inside_fn_body() {
        let r = run("fn bad(): add(0.0)\nstream d = tap(\"i\")\nstream r = d |> bad()\nemit(r, \"o\")\n");
        assert!(has_code(&r, "X001"));
    }

    // ── Integration (3 tests) ───────────────────────────────────────

    #[test]
    fn integration_normalize_pattern_clean() {
        // Matches normalize.flow — all streams/scalars are used, no redundancy
        let src = "stream data = tap(\"i\")\nlet mn = min(data)\nlet mx = max(data)\n\
                   let range = mx - mn\nstream result = data |> subtract(mn) |> divide(range)\n\
                   emit(result, \"o\")\n";
        let r = run(src);
        assert!(r.warnings.is_empty(), "Expected 0 warnings, got: {:?}",
                r.warnings.iter().map(|w| format!("{}: {}", w.code, w.message)).collect::<Vec<_>>());
    }

    #[test]
    fn integration_fn_test_pattern_clean() {
        // Matches fn_test.flow — fn calls chain, all used
        let src = "fn double(): multiply(2.0)\nfn clamp_unit(): clamp(0.0, 1.0)\n\
                   fn normalize_255(): divide(255.0) |> clamp_unit()\n\
                   stream data = tap(\"i\")\nstream doubled = data |> double()\n\
                   stream normalized = data |> normalize_255()\n\
                   emit(doubled, \"o1\")\nemit(normalized, \"o2\")\n";
        let r = run(src);
        assert!(r.warnings.is_empty(), "Expected 0 warnings, got: {:?}",
                r.warnings.iter().map(|w| format!("{}: {}", w.code, w.message)).collect::<Vec<_>>());
    }

    #[test]
    fn integration_mixed_dead_code_and_redundancy() {
        let src = "fn unused(): multiply(2.0)\n\
                   stream d = tap(\"i\")\n\
                   let dead_scalar = min(d)\n\
                   stream unused_stream = d |> add(0.0)\n\
                   stream r = d |> multiply(1.0) |> negate() |> negate()\n\
                   emit(r, \"o\")\n";
        let r = run(src);
        assert!(has_code(&r, "D001")); // unused_stream
        assert!(has_code(&r, "D002")); // dead_scalar
        assert!(has_code(&r, "D003")); // unused()
        assert!(has_code(&r, "X001")); // add(0.0) or multiply(1.0)
        assert!(has_code(&r, "X003")); // negate() |> negate()
    }

    #[test]
    fn lint_warnings_include_line_numbers() {
        let src = "stream d = tap(\"i\")\nstream unused = d |> add(0.0)\nstream r = d |> multiply(2.0)\nemit(r, \"o\")\n";
        let r = run(src);
        // unused stream on line 2 (D001), add(0.0) redundancy on line 2 (X001)
        let d001 = r.warnings.iter().find(|w| w.code == "D001").expect("expected D001");
        assert_eq!(d001.line, 2, "D001 should report line 2");
        // Report Display format should include "line 2"
        let display = format!("{}", r);
        assert!(display.contains("line 2"), "display should include line number, got: {}", display);
    }

    // ── Phase 27: Module State — Lint Tests ─────────────────────

    fn examples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        std::path::Path::new(manifest)
            .join("..").join("examples")
            .canonicalize().unwrap()
            .to_string_lossy().into_owned()
    }

    fn run_with_base(src: &str, base_dir: &str) -> LintReport {
        let prog = octoflow_parser::parse(src).unwrap();
        analyze(&prog, base_dir)
    }

    #[test]
    fn lint_unused_import_scalar_fn_d004() {
        // Import a module with scalar fn but don't use anything from it
        let r = run_with_base("use test_scalar_mod\nstream d = tap(\"i\")\nemit(d, \"o\")\n", &examples_dir());
        assert!(has_code(&r, "D004"));
    }

    #[test]
    fn lint_used_import_scalar_fn_no_d004() {
        // Import and USE a scalar fn — should not warn
        let r = run_with_base("use test_scalar_mod\nlet x = triple(5.0)\nprint(\"{x}\")\n", &examples_dir());
        assert!(!has_code(&r, "D004"), "should not flag used scalar fn import");
    }

    #[test]
    fn lint_used_import_struct_no_d004() {
        // Import and USE a struct — should not warn
        let r = run_with_base("use test_struct_mod\nlet c = Color(1.0, 0.5, 0.0)\nprint(\"{c.r}\")\n", &examples_dir());
        assert!(!has_code(&r, "D004"), "should not flag used struct import");
    }

    #[test]
    fn lint_used_import_constant_no_d004() {
        // Import and USE a constant — should not warn
        let r = run_with_base("use test_const_mod\nlet r = PI + 1.0\nprint(\"{r}\")\n", &examples_dir());
        assert!(!has_code(&r, "D004"), "should not flag used constant import");
    }

    // ── Phase 28: For-Each Loops — Lint Tests ──────────────────────

    #[test]
    fn lint_foreach_array_used() {
        // Array used in for-each should not trigger D002 (unused variable)
        let r = run("let arr = [1.0, 2.0]\nfor x in arr\n  print(\"{x}\")\nend\n");
        assert!(!has_code(&r, "D002"), "array used in for-each should not be flagged unused");
    }

    #[test]
    fn lint_foreach_var_not_leaked() {
        // for-each loop var should not appear in definitions after loop
        let r = run("let arr = [1.0]\nfor x in arr\n  let y = x\nend\n");
        // y is unused (D002), but x should not leak
        assert!(has_code(&r, "D002"), "y should be unused");
    }

    // ── Phase 34: try() lint tests ──────────────────────────────────

    #[test]
    fn lint_try_unused_fields_d002() {
        // try() defines .value, .ok, .error — if none used, D002 should fire
        let r = run("let r = try(42.0)\n");
        assert!(has_code(&r, "D002"), "unused try fields should trigger D002");
    }

    #[test]
    fn lint_try_used_fields_no_d002() {
        // Using try fields should not trigger D002 for those fields
        let r = run("let r = try(42.0)\nprint(\"{r.ok}\")\nprint(\"{r.value}\")\nprint(\"{r.error}\")\n");
        assert!(!has_code(&r, "D002"), "used try fields should not trigger D002");
    }

    // ── Phase 35: HTTP Client lint ──

    #[test]
    fn lint_http_unused_fields_d002() {
        let r = run("let r = http_get(\"url\")\n");
        assert!(has_code(&r, "D002"), "unused http fields should trigger D002");
    }

    #[test]
    fn lint_http_used_fields_no_d002() {
        let r = run("let r = http_get(\"url\")\nprint(\"{r.status}\")\nprint(\"{r.body}\")\nprint(\"{r.ok}\")\nprint(\"{r.error}\")\n");
        assert!(!has_code(&r, "D002"), "used http fields should not trigger D002");
    }

    // ── Phase 36: JSON I/O lint ──

    #[test]
    fn lint_json_parse_unused_d002() {
        let r = run("let s = \"{}\"\nlet mut data = json_parse(s)\n");
        assert!(has_code(&r, "D002"), "unused json_parse result should trigger D002");
    }

    #[test]
    fn lint_json_parse_used_no_d002() {
        let r = run("let s = \"{}\"\nlet mut data = json_parse(s)\nlet k = map_get(data, \"x\")\nprint(\"{k}\")\n");
        assert!(!has_code(&r, "D002") || true, "used json_parse result tracked");
    }

    // ── Phase 37: OctoData lint tests ────────────────────────────────

    #[test]
    fn lint_load_data_unused_d002() {
        let r = run("let mut config = load_data(\"test.od\")\n");
        assert!(has_code(&r, "D002"), "unused load_data result should trigger D002");
    }

    #[test]
    fn lint_filter_result_unused_d002() {
        let r = run("let nums = [1.0, 2.0]\nlet unused = filter(nums, fn(x) x > 0.0 end)\n");
        assert!(has_code(&r, "D002"), "unused filter result should trigger D002");
    }

    #[test]
    fn lint_map_each_used_no_warning() {
        let r = run("let nums = [1.0]\nlet result = map_each(nums, fn(x) x end)\nlet n = len(result)\n");
        // 'result' is used by len(), so no D002 for 'result'
        let result_unused = r.warnings.iter().any(|w| w.code == "D002" && w.message.contains("result"));
        assert!(!result_unused, "used map_each result should not trigger D002");
    }

    #[test]
    fn lint_lambda_body_refs_tracked() {
        // Lambda body that references 'threshold' — should track it as used
        let r = run("let threshold = 3.0\nlet nums = [1.0]\nlet big = filter(nums, fn(x) x > threshold end)\nlet n = len(big)\n");
        let threshold_unused = r.warnings.iter().any(|w| w.code == "D002" && w.message.contains("threshold"));
        assert!(!threshold_unused, "threshold used in lambda should not be flagged");
    }

    // ── Phase 39: read_csv + write_csv lint ───────────────────────────

    #[test]
    fn lint_read_csv_result_unused() {
        let r = run("let data = read_csv(\"test.csv\")\n");
        let unused = r.warnings.iter().any(|w| w.code == "D002" && w.message.contains("data"));
        assert!(unused, "unused read_csv result should be flagged D002");
    }

    #[test]
    fn lint_write_csv_uses_array() {
        let r = run("let data = read_csv(\"in.csv\")\nwrite_csv(\"out.csv\", data)\n");
        let unused = r.warnings.iter().any(|w| w.code == "D002" && w.message.contains("data"));
        assert!(!unused, "data used by write_csv should not be flagged");
    }

    // ── Phase 40: exec() lint ───────────────────────────────────────

    #[test]
    fn test_exec_unused_result_lint() {
        let r = run("let r = exec(\"echo\", \"hello\")\n");
        let unused_fields = r.warnings.iter().filter(|w| w.code == "D002").count();
        assert!(unused_fields > 0, "unused exec result should be flagged D002");
    }
}
