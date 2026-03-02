//! Range Tracker — Pass 2 advisory lint.
//!
//! Propagates value range intervals [lo, hi] through pipeline stages
//! and flags numeric hazards. Never blocks — advisory only.

use std::collections::HashMap;
use std::fmt;

use octoflow_parser::ast::{Arg, Expr, Program, ScalarExpr, StageCall, Statement};

// ── Range type ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub lo: f64,
    pub hi: f64,
}

impl Range {
    const UNKNOWN: Range = Range { lo: f64::NEG_INFINITY, hi: f64::INFINITY };

    fn point(v: f64) -> Self { Range { lo: v, hi: v } }

    fn is_unknown(&self) -> bool {
        self.lo == f64::NEG_INFINITY && self.hi == f64::INFINITY
    }

    fn includes_zero(&self) -> bool { self.lo <= 0.0 && self.hi >= 0.0 }

    fn sanitize(self) -> Self {
        if self.lo.is_nan() || self.hi.is_nan() { Range::UNKNOWN } else { self }
    }
}

// ── Report types ────────────────────────────────────────────────────

pub struct RangeWarning {
    pub code: &'static str,
    pub message: String,
    pub context: String,
    pub suggestion: String,
}

pub struct StreamRange {
    pub name: String,
    pub range: Range,
    pub note: String,
}

pub struct RangeReport {
    pub stream_ranges: Vec<StreamRange>,
    pub warnings: Vec<RangeWarning>,
}

impl fmt::Display for RangeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for sr in &self.stream_ranges {
            let lo = if sr.range.lo == f64::NEG_INFINITY { "-inf".into() }
                     else { format!("{:.1}", sr.range.lo) };
            let hi = if sr.range.hi == f64::INFINITY { "+inf".into() }
                     else { format!("{:.1}", sr.range.hi) };
            writeln!(f, "  stream {:<12} [{}, {}]  ({})", sr.name, lo, hi, sr.note)?;
        }
        if !self.warnings.is_empty() {
            writeln!(f)?;
            for w in &self.warnings {
                writeln!(f, "  [{}] {}", w.code, w.message)?;
                writeln!(f, "         {}", w.context)?;
                writeln!(f, "         {}", w.suggestion)?;
            }
        }
        writeln!(f)?;
        write!(f, "  STATUS: {} warning{}", self.warnings.len(),
               if self.warnings.len() == 1 { "" } else { "s" })
    }
}

// ── Function definitions (mirrors compiler.rs pattern) ──────────────

#[derive(Debug, Clone)]
struct FnDef {
    params: Vec<String>,
    body: Vec<StageCall>,
}

// ── Core analysis ───────────────────────────────────────────────────

pub fn analyze(program: &Program, base_dir: &str) -> RangeReport {
    let mut streams: HashMap<String, Range> = HashMap::new();
    let mut scalars: HashMap<String, Range> = HashMap::new();
    let mut fn_defs: HashMap<String, FnDef> = HashMap::new();
    let mut warnings: Vec<RangeWarning> = Vec::new();
    let mut stream_ranges: Vec<StreamRange> = Vec::new();

    // Collect fn defs, struct defs, and imports
    let mut struct_defs: HashMap<String, Vec<String>> = HashMap::new();
    for (stmt, _span) in &program.statements {
        match stmt {
            Statement::FnDecl { name, params, body } => {
                fn_defs.insert(name.clone(), FnDef { params: params.clone(), body: body.clone() });
            }
            Statement::UseDecl { module } => {
                import_module_fns(base_dir, module, &mut fn_defs);
            }
            Statement::StructDef { name, fields } => {
                struct_defs.insert(name.clone(), fields.clone());
            }
            _ => {}
        }
    }

    // Propagate ranges
    for (stmt, _span) in &program.statements {
        match stmt {
            Statement::StreamDecl { name, expr } => {
                let range = eval_expr_range(expr, &streams, &scalars, &fn_defs, &mut warnings, name);
                streams.insert(name.clone(), range);
                stream_ranges.push(StreamRange {
                    name: name.clone(), range,
                    note: if range.is_unknown() { "input".into() } else { "bounded".into() },
                });
            }
            Statement::ArrayDecl { name, elements, .. } => {
                // Track array element ranges for index access
                if !elements.is_empty() {
                    let mut lo = f64::INFINITY;
                    let mut hi = f64::NEG_INFINITY;
                    for elem in elements {
                        let r = eval_scalar_range(elem, &streams, &scalars);
                        lo = lo.min(r.lo);
                        hi = hi.max(r.hi);
                    }
                    scalars.insert(name.clone(), Range { lo, hi });
                }
            }
            Statement::LetDecl { name, value, .. } => {
                // Check for vec or struct constructor: register component ranges
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    if matches!(fn_name.as_str(), "vec2" | "vec3" | "vec4") {
                        let components = ["x", "y", "z", "w"];
                        for (i, arg) in args.iter().enumerate() {
                            if i < 4 {
                                let range = eval_scalar_range(arg, &streams, &scalars);
                                scalars.insert(format!("{}.{}", name, components[i]), range);
                            }
                        }
                        continue;
                    }
                    if let Some(fields) = struct_defs.get(fn_name) {
                        let fields = fields.clone();
                        for (i, arg) in args.iter().enumerate() {
                            if i < fields.len() {
                                let range = eval_scalar_range(arg, &streams, &scalars);
                                scalars.insert(format!("{}.{}", name, &fields[i]), range);
                            }
                        }
                        continue;
                    }
                }
                let range = eval_scalar_range(value, &streams, &scalars);
                scalars.insert(name.clone(), range);
            }
            Statement::Assign { name, value } => {
                let range = eval_scalar_range(value, &streams, &scalars);
                scalars.insert(name.clone(), range);
            }
            Statement::WhileLoop { body, .. } | Statement::ForLoop { body, .. } => {
                // Propagate ranges through loop body (single pass — conservative)
                for (body_stmt, _) in body {
                    match body_stmt {
                        Statement::LetDecl { name: bname, value, .. } => {
                            if let ScalarExpr::FnCall { name: fn_name, args } = value {
                                if matches!(fn_name.as_str(), "vec2" | "vec3" | "vec4") {
                                    let components = ["x", "y", "z", "w"];
                                    for (i, arg) in args.iter().enumerate() {
                                        if i < 4 {
                                            let range = eval_scalar_range(arg, &streams, &scalars);
                                            scalars.insert(format!("{}.{}", bname, components[i]), range);
                                        }
                                    }
                                    continue;
                                }
                                if let Some(fields) = struct_defs.get(fn_name) {
                                    let fields = fields.clone();
                                    for (i, arg) in args.iter().enumerate() {
                                        if i < fields.len() {
                                            let range = eval_scalar_range(arg, &streams, &scalars);
                                            scalars.insert(format!("{}.{}", bname, &fields[i]), range);
                                        }
                                    }
                                    continue;
                                }
                            }
                            let range = eval_scalar_range(value, &streams, &scalars);
                            scalars.insert(bname.clone(), range);
                        }
                        Statement::Assign { name: aname, value } => {
                            let range = eval_scalar_range(value, &streams, &scalars);
                            scalars.insert(aname.clone(), range);
                        }
                        Statement::ArrayDecl { name: aname, elements, .. } => {
                            if !elements.is_empty() {
                                let mut lo = f64::INFINITY;
                                let mut hi = f64::NEG_INFINITY;
                                for elem in elements {
                                    let r = eval_scalar_range(elem, &streams, &scalars);
                                    lo = lo.min(r.lo);
                                    hi = hi.max(r.hi);
                                }
                                scalars.insert(aname.clone(), Range { lo, hi });
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    RangeReport { stream_ranges, warnings }
}

// ── Expression evaluation ───────────────────────────────────────────

fn eval_expr_range(
    expr: &Expr, streams: &HashMap<String, Range>, scalars: &HashMap<String, Range>,
    fn_defs: &HashMap<String, FnDef>, warnings: &mut Vec<RangeWarning>, ctx: &str,
) -> Range {
    match expr {
        Expr::Tap { .. } => Range::UNKNOWN,
        Expr::RandomStream { lo, hi, .. } => Range { lo: *lo as f64, hi: *hi as f64 },
        Expr::Cache { inner, .. } => eval_expr_range(inner, streams, scalars, fn_defs, warnings, ctx),
        Expr::Ref { name } => streams.get(name).copied().unwrap_or(Range::UNKNOWN),
        Expr::Pipe { input, stages } => {
            let mut r = eval_expr_range(input, streams, scalars, fn_defs, warnings, ctx);
            for stage in stages {
                r = propagate_stage(r, stage, scalars, fn_defs, warnings, ctx);
            }
            r
        }
    }
}

fn resolve_arg(arg: &Arg, scalars: &HashMap<String, Range>) -> Range {
    match arg {
        Arg::Literal(v) => Range::point(*v),
        Arg::IntLiteral(i) => Range::point(*i as f64),
        Arg::Ref(name) => scalars.get(name).copied().unwrap_or(Range::UNKNOWN),
    }
}

// ── Stage propagation (all 22 ops) ─────────────────────────────────

fn propagate_stage(
    input: Range, stage: &StageCall, scalars: &HashMap<String, Range>,
    fn_defs: &HashMap<String, FnDef>, warnings: &mut Vec<RangeWarning>, ctx: &str,
) -> Range {
    let op = stage.operation.as_str();

    // Function call?
    if let Some(fn_def) = fn_defs.get(op) {
        return propagate_fn_call(input, fn_def, &stage.args, scalars, fn_defs, warnings, ctx);
    }

    let k = || stage.args.first().map(|a| resolve_arg(a, scalars)).unwrap_or(Range::point(0.0));

    let result = match op {
        "add" => {
            let k = k();
            Range { lo: input.lo + k.lo, hi: input.hi + k.hi }
        }
        "subtract" => {
            let k = k();
            Range { lo: input.lo - k.hi, hi: input.hi - k.lo }
        }
        "multiply" => {
            let k = k();
            let p = [input.lo * k.lo, input.lo * k.hi, input.hi * k.lo, input.hi * k.hi];
            Range { lo: min4(p), hi: max4(p) }
        }
        "divide" => {
            let k = k();
            if k.includes_zero() {
                warnings.push(RangeWarning {
                    code: "R003", message: "divide(): divisor range includes zero".into(),
                    context: format!("stream {}", ctx),
                    suggestion: "Ensure divisor is never zero, or guard with clamp".into(),
                });
                Range::UNKNOWN
            } else {
                let q = [input.lo / k.lo, input.lo / k.hi, input.hi / k.lo, input.hi / k.hi];
                Range { lo: min4(q), hi: max4(q) }
            }
        }
        "negate" => Range { lo: -input.hi, hi: -input.lo },
        "abs" => {
            if input.lo >= 0.0 { input }
            else if input.hi <= 0.0 { Range { lo: -input.hi, hi: -input.lo } }
            else { Range { lo: 0.0, hi: input.lo.abs().max(input.hi.abs()) } }
        }
        "mod" => {
            let k = k();
            Range { lo: 0.0, hi: k.lo.abs().max(k.hi.abs()) }
        }
        "clamp" => {
            let a = stage.args.get(0).map(|a| resolve_arg(a, scalars)).unwrap_or(Range::UNKNOWN);
            let b = stage.args.get(1).map(|b| resolve_arg(b, scalars)).unwrap_or(Range::UNKNOWN);
            Range { lo: input.lo.max(a.lo), hi: input.hi.min(b.hi) }
        }
        "min" => { let k = k(); Range { lo: input.lo.min(k.lo), hi: input.hi.min(k.hi) } }
        "max" => { let k = k(); Range { lo: input.lo.max(k.lo), hi: input.hi.max(k.hi) } }
        "pow" => {
            let k = k();
            if input.lo >= 0.0 {
                let v = [input.lo.powf(k.lo), input.lo.powf(k.hi),
                         input.hi.powf(k.lo), input.hi.powf(k.hi)];
                if input.hi > 1.0 && k.hi > 20.0 {
                    warnings.push(RangeWarning {
                        code: "R007",
                        message: format!("pow({}): large exponent with base > 1 may overflow", k.hi),
                        context: format!("stream {}", ctx),
                        suggestion: "Consider clamping input or using a smaller exponent".into(),
                    });
                }
                Range { lo: min4(v), hi: max4(v) }
            } else { Range::UNKNOWN }
        }
        "sqrt" => {
            if input.lo >= 0.0 {
                Range { lo: input.lo.sqrt(), hi: input.hi.sqrt() }
            } else {
                if !input.is_unknown() {
                    warnings.push(RangeWarning {
                        code: "R004", message: "sqrt(): input range includes negative values".into(),
                        context: format!("stream {}", ctx),
                        suggestion: "Guard with clamp(0, ...) or abs() before sqrt()".into(),
                    });
                }
                Range { lo: 0.0, hi: if input.hi >= 0.0 { input.hi.sqrt() } else { 0.0 } }
            }
        }
        "exp" => {
            if input.hi > 88.0 && !input.is_unknown() {
                warnings.push(RangeWarning {
                    code: "R006", message: "exp(): input range exceeds safe threshold (> 88)".into(),
                    context: format!("stream {}", ctx),
                    suggestion: "Clamp input before exp() to prevent overflow".into(),
                });
            }
            Range { lo: input.lo.exp(), hi: input.hi.exp() }
        }
        "log" => {
            if input.lo > 0.0 {
                Range { lo: input.lo.ln(), hi: input.hi.ln() }
            } else {
                if !input.is_unknown() {
                    warnings.push(RangeWarning {
                        code: "R005",
                        message: "log(): input range includes zero or negative values".into(),
                        context: format!("stream {}", ctx),
                        suggestion: "Guard with clamp(epsilon, ...) before log()".into(),
                    });
                }
                Range::UNKNOWN
            }
        }
        "sin" | "cos" => Range { lo: -1.0, hi: 1.0 },
        "floor" => Range { lo: input.lo.floor(), hi: input.hi.floor() },
        "ceil" => Range { lo: input.lo.ceil(), hi: input.hi.ceil() },
        "round" => Range { lo: input.lo.round(), hi: input.hi.round() },
        "ema" | "decay" => input,
        "prefix_sum" => {
            if input.lo >= 0.0 { Range { lo: input.lo, hi: f64::INFINITY } }
            else { Range::UNKNOWN }
        }
        _ => Range::UNKNOWN,
    };
    result.sanitize()
}

// ── Scalar range evaluation ─────────────────────────────────────────

fn eval_scalar_range(
    expr: &ScalarExpr, streams: &HashMap<String, Range>, scalars: &HashMap<String, Range>,
) -> Range {
    match expr {
        ScalarExpr::Literal(v) => Range::point(*v),
        ScalarExpr::IntLiteral(i) => Range::point(*i as f64),
        ScalarExpr::Ref(name) => scalars.get(name).copied().unwrap_or(Range::UNKNOWN),
        ScalarExpr::Reduce { op, stream } => {
            let s = streams.get(stream).copied().unwrap_or(Range::UNKNOWN);
            match op.as_str() {
                "min" | "max" => s,
                "sum" => if s.lo >= 0.0 { Range { lo: 0.0, hi: f64::INFINITY } }
                         else { Range::UNKNOWN },
                "count" => Range { lo: 0.0, hi: f64::INFINITY },
                _ => Range::UNKNOWN,
            }
        }
        ScalarExpr::BinOp { left, op, right } => {
            let l = eval_scalar_range(left, streams, scalars);
            let r = eval_scalar_range(right, streams, scalars);
            match op {
                octoflow_parser::ast::BinOp::Add => Range { lo: l.lo + r.lo, hi: l.hi + r.hi },
                octoflow_parser::ast::BinOp::Sub => Range { lo: l.lo - r.hi, hi: l.hi - r.lo },
                octoflow_parser::ast::BinOp::Mul => {
                    let p = [l.lo * r.lo, l.lo * r.hi, l.hi * r.lo, l.hi * r.hi];
                    Range { lo: min4(p), hi: max4(p) }
                }
                octoflow_parser::ast::BinOp::Div => {
                    if r.includes_zero() { Range::UNKNOWN }
                    else {
                        let q = [l.lo / r.lo, l.lo / r.hi, l.hi / r.lo, l.hi / r.hi];
                        Range { lo: min4(q), hi: max4(q) }
                    }
                }
                octoflow_parser::ast::BinOp::Mod => {
                    // Modulo result is bounded by the divisor
                    if r.lo > 0.0 { Range { lo: 0.0, hi: r.hi } }
                    else { Range::UNKNOWN }
                }
                // Bitwise operations: range is unknown (depends on bit patterns)
                octoflow_parser::ast::BinOp::Shl
                | octoflow_parser::ast::BinOp::Shr
                | octoflow_parser::ast::BinOp::BitAnd
                | octoflow_parser::ast::BinOp::BitOr
                | octoflow_parser::ast::BinOp::BitXor => Range::UNKNOWN,
            }
        }
        // Comparisons and booleans return 0.0 or 1.0
        ScalarExpr::Compare { .. } | ScalarExpr::And { .. } | ScalarExpr::Or { .. } => {
            Range { lo: 0.0, hi: 1.0 }
        }
        ScalarExpr::Bool(_) => Range { lo: 0.0, hi: 1.0 },
        ScalarExpr::StringLiteral(_) | ScalarExpr::NoneLiteral => Range::UNKNOWN,
        // Conditionals: range is union of both branches
        ScalarExpr::If { then_expr, else_expr, .. } => {
            let t = eval_scalar_range(then_expr, streams, scalars);
            let e = eval_scalar_range(else_expr, streams, scalars);
            Range { lo: t.lo.min(e.lo), hi: t.hi.max(e.hi) }
        }
        ScalarExpr::FnCall { name, args } => {
            let arg_ranges: Vec<Range> = args.iter()
                .map(|a| eval_scalar_range(a, streams, scalars))
                .collect();
            match (name.as_str(), arg_ranges.as_slice()) {
                ("abs", [a]) => {
                    if a.lo >= 0.0 { *a }
                    else if a.hi <= 0.0 { Range { lo: -a.hi, hi: -a.lo } }
                    else { Range { lo: 0.0, hi: a.lo.abs().max(a.hi.abs()) } }
                }
                ("sqrt", [a]) => {
                    if a.lo >= 0.0 { Range { lo: a.lo.sqrt(), hi: a.hi.sqrt() } }
                    else { Range { lo: 0.0, hi: if a.hi >= 0.0 { a.hi.sqrt() } else { 0.0 } } }
                }
                ("exp", [a]) => Range { lo: a.lo.exp(), hi: a.hi.exp() }.sanitize(),
                ("log", [a]) => {
                    if a.lo > 0.0 { Range { lo: a.lo.ln(), hi: a.hi.ln() } }
                    else { Range::UNKNOWN }
                }
                ("sin", _) | ("cos", _) => Range { lo: -1.0, hi: 1.0 },
                ("floor", [a]) => Range { lo: a.lo.floor(), hi: a.hi.floor() },
                ("ceil", [a]) => Range { lo: a.lo.ceil(), hi: a.hi.ceil() },
                ("round", [a]) => Range { lo: a.lo.round(), hi: a.hi.round() },
                ("pow", [base, exp]) => {
                    if base.lo >= 0.0 {
                        let v = [base.lo.powf(exp.lo), base.lo.powf(exp.hi),
                                 base.hi.powf(exp.lo), base.hi.powf(exp.hi)];
                        Range { lo: min4(v), hi: max4(v) }.sanitize()
                    } else { Range::UNKNOWN }
                }
                ("clamp", [val, lo, hi]) => {
                    Range { lo: val.lo.max(lo.lo), hi: val.hi.min(hi.hi) }
                }
                ("random", []) => Range { lo: 0.0, hi: 1.0 },
                ("len", _) => Range { lo: 0.0, hi: f64::MAX },
                ("contains", _) => Range { lo: 0.0, hi: 1.0 },
                _ => Range::UNKNOWN,
            }
        }
        ScalarExpr::Index { array, .. } => {
            // Use the array's tracked range (union of all elements)
            scalars.get(array).copied().unwrap_or(Range::UNKNOWN)
        }
        ScalarExpr::Lambda { .. } => Range::UNKNOWN,
        ScalarExpr::ArrayLiteral(_) => Range::UNKNOWN,
    }
}

// ── Function inlining ───────────────────────────────────────────────

fn propagate_fn_call(
    input: Range, fn_def: &FnDef, call_args: &[Arg],
    scalars: &HashMap<String, Range>, fn_defs: &HashMap<String, FnDef>,
    warnings: &mut Vec<RangeWarning>, ctx: &str,
) -> Range {
    let mut local = scalars.clone();
    for (param, arg) in fn_def.params.iter().zip(call_args.iter()) {
        local.insert(param.clone(), resolve_arg(arg, scalars));
    }
    let mut r = input;
    for stage in &fn_def.body {
        r = propagate_stage(r, stage, &local, fn_defs, warnings, ctx);
    }
    r
}

fn import_module_fns(base_dir: &str, module: &str, fn_defs: &mut HashMap<String, FnDef>) {
    let path = std::path::Path::new(base_dir).join(format!("{}.flow", module));
    if let Ok(source) = std::fs::read_to_string(&path) {
        if let Ok(prog) = octoflow_parser::parse(&source) {
            for (stmt, _span) in &prog.statements {
                if let Statement::FnDecl { name, params, body } = stmt {
                    let def = FnDef { params: params.clone(), body: body.clone() };
                    fn_defs.insert(format!("{}.{}", module, name), def.clone());
                    fn_defs.insert(name.clone(), def);
                }
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn min4(v: [f64; 4]) -> f64 { v.iter().copied().fold(f64::INFINITY, f64::min) }
fn max4(v: [f64; 4]) -> f64 { v.iter().copied().fold(f64::NEG_INFINITY, f64::max) }

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> RangeReport {
        let prog = octoflow_parser::parse(src).unwrap();
        analyze(&prog, ".")
    }

    fn stream_range(r: &RangeReport, name: &str) -> Range {
        r.stream_ranges.iter().find(|s| s.name == name).unwrap().range
    }

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 0.001 }

    // ── Unit: range arithmetic (13 tests) ───────────────────────────

    #[test]
    fn range_add() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(0.0, 10.0) |> add(5.0)\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, 5.0) && approx(y.hi, 15.0));
    }

    #[test]
    fn range_subtract() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(0.0, 10.0) |> subtract(3.0)\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, -3.0) && approx(y.hi, 7.0));
    }

    #[test]
    fn range_multiply_positive() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(0.0, 10.0) |> multiply(3.0)\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, 0.0) && approx(y.hi, 30.0));
    }

    #[test]
    fn range_multiply_negative() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(0.0, 10.0) |> multiply(-2.0)\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, -20.0) && approx(y.hi, 0.0));
    }

    #[test]
    fn range_divide_positive() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(2.0, 10.0) |> divide(2.0)\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, 1.0) && approx(y.hi, 5.0));
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn range_negate() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(2.0, 8.0) |> negate()\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, -8.0) && approx(y.hi, -2.0));
    }

    #[test]
    fn range_abs_mixed() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(-3.0, 5.0) |> abs()\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, 0.0) && approx(y.hi, 5.0));
    }

    #[test]
    fn range_clamp() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(10.0, 20.0)\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, 10.0) && approx(y.hi, 20.0));
    }

    #[test]
    fn range_sqrt_positive() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(4.0, 16.0) |> sqrt()\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, 2.0) && approx(y.hi, 4.0));
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn range_sin() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> sin()\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, -1.0) && approx(y.hi, 1.0));
    }

    #[test]
    fn range_exp() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(0.0, 1.0) |> exp()\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, 1.0) && approx(y.hi, std::f64::consts::E));
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn range_log_positive() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(1.0, 10.0) |> log()\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, 0.0) && approx(y.hi, 10.0_f64.ln()));
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn range_floor() {
        let r = run("stream x = tap(\"i\")\nstream y = x |> clamp(2.3, 5.7) |> floor()\n");
        let y = stream_range(&r, "y");
        assert!(approx(y.lo, 2.0) && approx(y.hi, 5.0));
    }

    // ── Integration (9 tests) ───────────────────────────────────────

    #[test]
    fn normalize_pattern_r003() {
        let src = "stream data = tap(\"i\")\nlet mn = min(data)\nlet mx = max(data)\n\
                   let range = mx - mn\nstream result = data |> subtract(mn) |> divide(range)\n";
        let r = run(src);
        assert_eq!(r.warnings.len(), 1);
        assert_eq!(r.warnings[0].code, "R003");
    }

    #[test]
    fn cinematic_pattern_clean() {
        let src = "stream p = tap(\"i\")\nstream w = p |> add(15.0) |> clamp(0.0, 255.0)\n\
                   stream g = w |> divide(255.0) |> pow(0.95) |> multiply(255.0)\n\
                   stream f = g |> clamp(0.0, 255.0)\n";
        let r = run(src);
        assert!(r.warnings.is_empty());
        let f = stream_range(&r, "f");
        assert!(f.lo >= 0.0 && f.hi <= 255.0);
    }

    #[test]
    fn fn_inline_normalize() {
        let src = "fn clamp_unit(): clamp(0.0, 1.0)\nfn norm(): divide(255.0) |> clamp_unit()\n\
                   stream d = tap(\"i\")\nstream n = d |> norm()\n";
        let r = run(src);
        assert!(r.warnings.is_empty());
        let n = stream_range(&r, "n");
        assert!(approx(n.lo, 0.0) && approx(n.hi, 1.0));
    }

    #[test]
    fn divide_zero_literal_r003() {
        let r = run("stream d = tap(\"i\")\nstream out = d |> divide(0.0)\n");
        assert_eq!(r.warnings.len(), 1);
        assert_eq!(r.warnings[0].code, "R003");
    }

    #[test]
    fn clamp_then_sqrt_clean() {
        let r = run("stream d = tap(\"i\")\nstream out = d |> clamp(0.0, 100.0) |> sqrt()\n");
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn unknown_sqrt_suppressed() {
        let r = run("stream d = tap(\"i\")\nstream out = d |> sqrt()\n");
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn proven_negative_sqrt_r004() {
        let r = run("stream d = tap(\"i\")\nstream out = d |> clamp(1.0, 10.0) |> negate() |> sqrt()\n");
        assert_eq!(r.warnings.len(), 1);
        assert_eq!(r.warnings[0].code, "R004");
    }

    #[test]
    fn exp_overflow_r006() {
        let r = run("stream d = tap(\"i\")\nstream out = d |> clamp(0.0, 100.0) |> exp()\n");
        assert_eq!(r.warnings.len(), 1);
        assert_eq!(r.warnings[0].code, "R006");
    }

    #[test]
    fn scalar_range_tracking() {
        let src = "stream d = tap(\"i\")\nstream s = d |> clamp(1.0, 100.0)\n\
                   let mx = max(s)\nstream out = s |> divide(mx)\n";
        let r = run(src);
        assert!(r.warnings.is_empty()); // mx in [1, 100], no zero
    }
}
