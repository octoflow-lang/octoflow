//! AST types for the OctoFlow language.

/// Source location (line and column, 1-indexed).
#[derive(Debug, Clone, Copy, Default)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

/// A complete OctoFlow program — a sequence of statements with source locations.
#[derive(Debug, Clone)]
pub struct Program {
    pub statements: Vec<(Statement, Span)>,
}

/// A top-level statement.
#[derive(Debug, Clone)]
pub enum Statement {
    /// `stream <name> = <expr>`
    StreamDecl { name: String, expr: Expr },
    /// `let <name> = <scalar_expr>` or `let mut <name> = <scalar_expr>`
    LetDecl { name: String, value: ScalarExpr, mutable: bool },
    /// `<name> = <scalar_expr>` — reassign a mutable variable
    Assign { name: String, value: ScalarExpr },
    /// `emit(<expr>, "<path>")`
    Emit { expr: Expr, path: String },
    /// `print("text {scalar} more text")` — with interpolation
    Print { segments: Vec<PrintSegment> },
    /// `fn <name>(<params>): <stage> |> <stage> ...`
    FnDecl { name: String, params: Vec<String>, body: Vec<StageCall> },
    /// `use <module>` — import functions from another .flow file
    UseDecl { module: String },
    /// `struct <Name>(<field1>, <field2>, ...)` — define a named record type
    StructDef { name: String, fields: Vec<String> },
    /// `let <name> = [<expr1>, <expr2>, ...]` — array of scalar values
    ArrayDecl { name: String, elements: Vec<ScalarExpr>, mutable: bool },
    /// `arr[index] = value` — assign to array element (requires mutable array)
    ArrayAssign { array: String, index: ScalarExpr, value: ScalarExpr },
    /// `push(arr, value)` — append element to mutable array
    ArrayPush { array: String, value: ScalarExpr },
    /// `let <name> = map()` — declare an empty hashmap
    MapDecl { name: String, mutable: bool },
    /// `map_set(<map>, <key_expr>, <value_expr>)` — insert/update a key in a mutable map
    MapInsert { map: String, key: ScalarExpr, value: ScalarExpr },
    /// `write_file(<path_expr>, <content_expr>)` — write string to file (overwrite)
    WriteFile { path: ScalarExpr, content: ScalarExpr },
    /// `append_file(<path_expr>, <content_expr>)` — append string to file
    AppendFile { path: ScalarExpr, content: ScalarExpr },
    /// `write_bytes(<path_expr>, <array_name>)` — write array as raw bytes (f32→u8)
    WriteBytes { path: ScalarExpr, array_name: String },
    /// `save_data(<path_expr>, <map_name>)` — serialize hashmap to .od file
    SaveData { path: ScalarExpr, map_name: String },
    /// `write_csv(<path_expr>, <array_name>)` — write array of maps to CSV file
    WriteCsv { path: ScalarExpr, array_name: String },
    /// `while <condition>` ... `end` — loop while condition is non-zero
    WhileLoop { condition: ScalarExpr, body: Vec<(Statement, Span)> },
    /// `for <var> in range(<start>, <end>)` ... `end` — counted iteration
    ForLoop { var: String, start: ScalarExpr, end: ScalarExpr, body: Vec<(Statement, Span)> },
    /// `for <var> in <array>` ... `end` — iterate over array elements
    ForEachLoop { var: String, iterable: String, body: Vec<(Statement, Span)> },
    /// `break` — exit the innermost loop
    Break,
    /// `continue` — skip to the next iteration of the innermost loop
    Continue,
    /// `fn <name>(<params>)` ... `return <expr>` ... `end` — scalar function with imperative body
    ScalarFnDecl { name: String, params: Vec<String>, body: Vec<(Statement, Span)> },
    /// `return <expr>` — return a value from a scalar function
    Return { value: ScalarExpr },
    /// `if <cond>` ... `elif <cond>` ... `else` ... `end` — block conditional
    IfBlock {
        condition: ScalarExpr,
        body: Vec<(Statement, Span)>,
        elif_branches: Vec<(ScalarExpr, Vec<(Statement, Span)>)>,
        else_body: Vec<(Statement, Span)>,
    },
    /// `name(args...)` — bare function call as statement (result discarded)
    ExprStmt { expr: ScalarExpr },
    /// `extern "<library>" { fn name(p: type, ...) -> ret ... }` — FFI declarations
    ExternBlock { library: String, functions: Vec<ExternFn>, span: Span },
}

/// A function declaration inside an `extern` block.
#[derive(Debug, Clone)]
pub struct ExternFn {
    pub name: String,
    pub params: Vec<ExternParam>,
    pub return_type: Option<String>,
}

/// A parameter in an extern function declaration.
#[derive(Debug, Clone)]
pub struct ExternParam {
    pub name: String,
    pub type_name: String,
}

/// A segment of an interpolated print string.
#[derive(Debug, Clone)]
pub enum PrintSegment {
    /// Static text: `"hello "`
    Literal(String),
    /// Scalar reference: `{name}` or `{name:.N}` for precision
    Scalar { name: String, precision: Option<usize> },
    /// Expression: `print(expr)` — evaluate and print result
    Expr(ScalarExpr),
}

/// An expression that produces a data stream.
#[derive(Debug, Clone)]
pub enum Expr {
    /// `tap("<path>")` — read CSV input
    Tap { path: String },
    /// `random_stream(N)` or `random_stream(N, lo, hi)` — GPU-native random data source.
    /// Generates N float32 values in [lo, hi) using Wang hash PRNG directly on the GPU.
    /// Zero CPU data upload — numbers are generated in GPU registers.
    RandomStream { count: Box<ScalarExpr>, lo: f32, hi: f32 },
    /// `cache("key") <expr>` — memoize a stream pipeline result by key.
    /// On first evaluation: runs the inner expr, stores result in the session cache.
    /// On subsequent evaluations with the same key: returns cached data, skips GPU dispatch.
    Cache { key: String, inner: Box<Expr> },
    /// Reference to a previously declared stream
    Ref { name: String },
    /// `<input> |> stage1() |> stage2()` — pipeline with one or more stages
    Pipe {
        input: Box<Expr>,
        stages: Vec<StageCall>,
    },
}

/// A single stage in a pipeline: `operation(arg1, arg2, ...)`.
#[derive(Debug, Clone)]
pub struct StageCall {
    pub operation: String,
    pub args: Vec<Arg>,
}

/// A stage call argument — a literal number (float or int) or a scalar reference.
#[derive(Debug, Clone)]
pub enum Arg {
    Literal(f64),
    IntLiteral(i64),
    Ref(String),
}

/// A scalar expression (from reduce, arithmetic, comparison, or conditional).
#[derive(Debug, Clone)]
pub enum ScalarExpr {
    /// `min(data)`, `max(data)`, `sum(data)` — reduce a stream to scalar
    Reduce { op: String, stream: String },
    /// Binary arithmetic: `left op right`
    BinOp { left: Box<ScalarExpr>, op: BinOp, right: Box<ScalarExpr> },
    /// Reference to a let-bound scalar
    Ref(String),
    /// Float constant
    Literal(f64),
    /// Integer constant
    IntLiteral(i64),
    /// `if <cond> then <true_val> else <false_val>`
    If { condition: Box<ScalarExpr>, then_expr: Box<ScalarExpr>, else_expr: Box<ScalarExpr> },
    /// Comparison: `left op right` -> 1.0 (true) or 0.0 (false)
    Compare { left: Box<ScalarExpr>, op: CompareOp, right: Box<ScalarExpr> },
    /// Boolean AND: `left && right`
    And { left: Box<ScalarExpr>, right: Box<ScalarExpr> },
    /// Boolean OR: `left || right`
    Or { left: Box<ScalarExpr>, right: Box<ScalarExpr> },
    /// Boolean literal: `true` = 1.0, `false` = 0.0
    Bool(bool),
    /// None literal
    NoneLiteral,
    /// Scalar function call: `abs(x)`, `sqrt(range)`, `pow(x, 2.0)`
    FnCall { name: String, args: Vec<ScalarExpr> },
    /// Array index access: `arr[expr]`
    Index { array: String, index: Box<ScalarExpr> },
    /// String literal: `"hello"`
    StringLiteral(String),
    /// Lambda expression: `fn(params) expr end`
    Lambda { params: Vec<String>, body: Box<ScalarExpr> },
    /// Array literal expression: `[1, 2, 3]`
    ArrayLiteral(Vec<ScalarExpr>),
}

/// Binary arithmetic operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,      // % (modulo)
    // Bitwise operators (Phase 43)
    Shl,      // << (left shift)
    Shr,      // >> (right shift)
    BitAnd,   // & (bitwise AND)
    BitOr,    // | (bitwise OR)
    BitXor,   // ^ (bitwise XOR)
}

/// Comparison operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompareOp {
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    Equal,
    NotEqual,
}
