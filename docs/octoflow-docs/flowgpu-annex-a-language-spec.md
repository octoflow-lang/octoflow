# OctoFlow — Annex A: Layer 2 Language Specification

**Parent Document:** OctoFlow Blueprint & Architecture  
**Status:** Draft  
**Version:** 0.1  
**Date:** February 14, 2026  

---

## Table of Contents

1. Overview & Design Philosophy
2. Core Language
3. General-Purpose Language Additions
4. Type System
5. Control Flow & Dataflow Semantics
6. Vanilla Standard Library
7. Standard Modules (Ships with Language)
8. Extended Module Ecosystem (On-Demand)
9. Module System Architecture
10. Module Authoring
11. Module Registry & Community Contribution
12. Module Resolution
13. Failsafe System & Pre-flight Validation
14. LLM-Driven Module Composition & Ecosystem Flywheel
15. Future: Reactive Extensions & Application Platform
16. Open Questions

---

## 1. Overview & Design Philosophy

Layer 2 is the language itself — the contract between humans/LLMs (Layer 3) and the compiler (Layer 1). OctoFlow is a **general-purpose programming language** where computation is expressed as dataflow. GPU acceleration is its superpower, not its only capability. Any program that can be written in Python or Rust can be written in OctoFlow — with the added benefit that compute-heavy portions automatically run on GPU when profitable.

The language consists of three distinct tiers:

```
┌─────────────────────────────────────────┐
│  EXTENDED MODULES (On-Demand Ecosystem) │
│  App frameworks, GUI, database, network │
│  Desktop/mobile, web server, ML, crypto │
│  Community-driven, domain-specific      │
├─────────────────────────────────────────┤
│  STANDARD MODULES (Ships with Language) │
│  String, file I/O, console, datetime,  │
│  JSON/CSV, CLI args, formatting, OS,    │
│  regex, HTTP client, process, env       │
│  Makes OctoFlow practically useful       │
├─────────────────────────────────────────┤
│  VANILLA (Canonical Compute Library)    │
│  Math, statistics, array operations     │
│  GPU-accelerated reference impls        │
│  Correctness oracle for module testing  │
├─────────────────────────────────────────┤
│  CORE LANGUAGE (Primitives & Semantics) │
│  Streams, pipes, stages, taps, types    │
│  Variables, errors, control flow        │
│  What the compiler natively understands │
└─────────────────────────────────────────┘
```

### Design Principles

1. **General-purpose first, GPU-accelerated second.** OctoFlow can do everything Python can do — read files, manipulate strings, make HTTP requests, build applications. The difference is that when computation gets heavy, it automatically goes to GPU. The language is not limited to GPU workloads.

2. **Small core, rich ecosystem.** The core language is deliberately minimal — just enough to define dataflow structure, types, and control flow. All actual computation lives in vanilla, standard modules, or community modules.

3. **Parallel by default.** If two computations have no data dependency, they are parallel. No annotation, no thread management, no explicit parallelism. The compiler infers parallelism from the dataflow graph structure.

4. **Hardware-invisible.** Programmers never mention GPU, CPU, threads, warps, blocks, shared memory, or device transfers. The compiler handles all hardware decisions. String operations quietly run on CPU. Matrix operations quietly run on GPU. The programmer doesn't care.

5. **Familiar surface syntax.** Approachable to Python/JavaScript developers. The pipe operator (`|>`) is the primary composition mechanism. Code reads top-to-bottom as a data transformation pipeline.

6. **Escape hatches for experts.** Annotations like `@sequential`, `@parallel(dim)`, `@device(gpu)` exist for overriding compiler decisions when needed. Normal users never touch these.

7. **Vanilla is canonical.** The vanilla standard library defines what compute operations *mean* in this language. It is the reference implementation, the correctness benchmark, and the always-available fallback.

8. **Standard modules make the language practical.** Vanilla covers math and data. Standard modules cover everything else a working programmer needs — strings, files, network, system. They ship with the language and are always available.

9. **Extended modules make the language powerful.** Community modules expand OctoFlow into any domain — web servers, desktop apps, mobile apps, databases, machine learning, game engines. These are installed on demand, like pip packages or cargo crates.

---

## 2. Core Language

The core language defines the structural primitives that the compiler natively understands. These are not operations on data — they are the grammar for connecting operations.

### 2.1 Streams

A stream is a typed, potentially unbounded sequence of data. Streams are the primary data abstraction.

```
stream prices: float[N]              # N-dimensional float array
stream frames: image[1080, 1920, 3]  # typed multidimensional data
stream events: record{time: int, symbol: string, price: float}
```

Streams can be:
- **Finite**: A bounded dataset (file, array, query result)
- **Continuous**: An unbounded data source (market feed, camera, sensor)

The compiler uses this distinction for optimization — finite streams can be fully materialized in GPU memory; continuous streams require windowed/streaming execution.

### 2.2 Stages

A stage is a transformation function. Stages are the computational units of the dataflow graph.

```
stage double(x: float) -> float:
    return x * 2

stage normalize(data: float[N]) -> float[N]:
    mn = min(data)
    mx = max(data)
    return (data - mn) / (mx - mn)
```

Stages are:
- **Pure by default**: No side effects, no mutable external state. The compiler relies on purity for parallelization decisions. Impure stages must be explicitly marked.
- **Typed**: Input and output types are declared and enforced.
- **Composable**: Stages connect via pipes to form the dataflow graph.

### 2.3 Pipes

The pipe operator (`|>`) connects stages, creating edges in the dataflow graph.

```
stream result = input |> stage_a() |> stage_b() |> stage_c()
```

This is equivalent to `stage_c(stage_b(stage_a(input)))` but reads as a left-to-right data flow, which maps directly to the execution graph.

**Pipes are typed connections.** The output type of stage_a must be compatible with the input type of stage_b. The compiler enforces this statically.

**Pipes have implicit parallelism.** If stage_a and stage_b operate on independent data dimensions, the compiler can execute them concurrently or pipeline them.

### 2.4 Taps

Taps are the boundary between the dataflow system and the external world. Data enters through input taps and exits through output taps.

```
stream data = tap("source_name")     # data enters the system
emit(result, "destination_name")     # data exits the system
```

Taps are inherently CPU-side operations (I/O requires OS interaction). The compiler inserts CPU→GPU transfers after input taps and GPU→CPU transfers before output taps as needed.

Tap sources/destinations are abstract — they can be files, network sockets, databases, inter-process channels, or any registered I/O adapter.

### 2.5 Accumulators

Accumulators are reduction points where parallel data converges.

```
stream total = values |> reduce(sum)            # many → one
stream counts = records |> group_by(key) |> count()  # many → few
```

Accumulators are significant to the compiler because reductions require synchronization across parallel execution units. The compiler maps these to GPU parallel reduction patterns (tree reduction, warp shuffle, etc.).

### 2.6 Temporal Marker

The `temporal` keyword marks a dependency across time steps — iteration N depends on iteration N-1.

```
stream ema = prices |> temporal decay(0.94)
```

Without `temporal`, the compiler assumes iterations are independent and parallelizes across them. With `temporal`, the compiler pipelines across time (sequential on the time axis) but can still parallelize across other dimensions (e.g., instruments).

### 2.7 Annotations (Escape Hatches)

For expert users who need to override compiler decisions:

```
@sequential                    # force sequential execution
@parallel(dim=0)               # force parallelism on dimension 0
@device(gpu)                   # force GPU execution
@device(cpu)                   # force CPU execution
@inline                        # hint to fuse this stage with neighbors
```

These are never required. The compiler makes good decisions by default. Annotations exist for performance tuning and edge cases.

---

## 3. General-Purpose Language Additions

The core dataflow primitives (streams, stages, pipes) can express any computation. But for OctoFlow to feel like a practical general-purpose language, it needs a few additional constructs that make everyday programming natural.

### 3.1 Variables (Local Bindings)

Not everything is a stream. Sometimes you just need a value.

```flow
let threshold = 0.05
let max_retries = 3
let config = load_config("settings.toml")
let name = "OctoFlow"
```

`let` creates an immutable binding. The value is computed once and doesn't flow through a pipeline. Under the hood, `let` is a zero-element stream — the compiler optimizes it to a simple value.

For mutable local state within a stage:

```flow
stage process(data: float[N]) -> float[N]:
    var accumulator = 0.0
    var count = 0
    for item in data:
        accumulator = accumulator + item
        count = count + 1
    let mean = accumulator / count
    return data |> subtract(mean)
```

`var` creates a mutable binding scoped to the current stage. Mutable state is local only — it never escapes the stage boundary, so it doesn't affect parallelism analysis.

### 3.2 Error Handling

Programs fail. File not found. Network timeout. Division by zero. OctoFlow handles errors through a `Result` type inspired by Rust:

```flow
// Operations that can fail return Result<T, Error>
let file = open("data.csv")   // Result<FileHandle, IoError>

// Handle with match
match file:
    ok(handle) -> handle |> read_csv() |> process()
    err(e)     -> emit(format("Failed: {e}"), "stderr")

// Or propagate with ?
stage load_data(path: string) -> Result<DataFrame, Error>:
    let handle = open(path)?           // propagates error if open fails
    let raw = read_csv(handle)?        // propagates error if parse fails
    let cleaned = raw |> drop_null()
    return ok(cleaned)

// Or use default
let data = open("data.csv") |> unwrap_or(empty_dataframe())
```

Errors never silently crash. They propagate through the pipeline as typed values. The pre-flight system can analyze error propagation paths at compile time.

### 3.3 Print & Logging

Syntactic sugar for the most common I/O:

```flow
print("Hello, world!")                        // → emit to stdout
print("Processing {count} records...")        // string interpolation
log.info("Pipeline started")                  // structured logging
log.error("Failed to connect: {err}")
log.debug("Stage 3 received {n} elements")    // stripped in release mode
```

Under the hood, `print` is `emit(value, "stdout")` and `log` routes to a configurable logging sink. These are CPU-side operations — the compiler knows this and handles them appropriately.

### 3.4 String Interpolation

Strings are first-class, CPU-targeted:

```flow
let name = "OctoFlow"
let version = 1
let msg = "Welcome to {name} v{version}"      // string interpolation
let path = "{base_dir}/{filename}.csv"         // path construction
let query = "SELECT * FROM {table} WHERE id = {id}"
```

String operations always run on CPU. The compiler handles this transparently — if a GPU pipeline needs to produce a string (e.g., formatting output), the relevant data transfers to CPU, the string operation executes, and the result continues.

### 3.5 Collections

Beyond arrays (which are GPU-friendly), OctoFlow has CPU-native collections for general programming:

```flow
// Map (hash map / dictionary)
let config = map{
    "host": "localhost",
    "port": 8080,
    "debug": true
}
let host = config["host"]

// List (dynamic-length, CPU-native)
let items = list[1, 2, 3, 4, 5]
let extended = items |> append(6)

// Set
let unique_symbols = set["AAPL", "GOOG", "MSFT"]
let has_apple = unique_symbols |> contains("AAPL")

// Tuple
let pair = (42, "answer")
let (num, label) = pair    // destructuring
```

**Key distinction:** Arrays (`float[N]`) are GPU-friendly contiguous memory. Collections (`map`, `list`, `set`) are CPU-native data structures for general programming. The compiler knows the difference and routes accordingly.

### 3.6 Functions (Non-Streaming)

Stages transform streams. But sometimes you need a plain function:

```flow
fn calculate_risk(price: float, volatility: float, position: float) -> float:
    return price * volatility * position * 0.01

fn validate_email(email: string) -> bool:
    return email |> contains("@") |> and(email |> contains("."))

fn format_report(stats: Stats) -> string:
    return "Mean: {stats.mean}, Std: {stats.std}, N: {stats.count}"
```

`fn` defines a regular function (not a stream transformer). Functions can be called from within stages, from other functions, or at the top level. The compiler infers whether a function should run on GPU or CPU based on its contents — a math function with no strings may get GPU-inlined, while a string formatting function always goes to CPU.

### 3.7 Entry Point

Every OctoFlow program has an implicit or explicit entry point:

```flow
// Implicit: top-level statements execute in order
let data = tap("input.csv", format=csv)
let result = data |> process() |> analyze()
emit(result, "output.csv")

// Explicit: main function for structured programs
fn main(args: Args):
    let config = args |> parse_config()
    match config.command:
        "analyze" -> run_analysis(config)
        "serve"   -> start_server(config)
        "help"    -> print(help_text)
        _         -> print("Unknown command: {config.command}")
```

### 3.8 Modules & Imports

```flow
// Import standard modules
import std.io                   // file I/O
import std.net.http             // HTTP client
import std.json                 // JSON parsing
import std.datetime             // date/time operations

// Import community modules
import community.web.server     // web server framework
import community.ui.desktop     // desktop GUI framework

// Import specific items
from std.io import read_file, write_file
from std.json import parse, stringify

// Use
let data = read_file("input.json")? |> parse()?
let response = http.get("https://api.example.com/data")?
```

---

## 4. Type System

The type system serves two purposes: correctness (catching errors at compile time) and compiler intelligence (providing information for GPU/CPU classification).

### 3.1 Primitive Types

```
int          # integer (platform-width)
int8, int16, int32, int64
uint8, uint16, uint32, uint64
float        # float (defaults to float32)
float16, float32, float64
bool
string       # note: strings are CPU-native; GPU stages avoid strings
byte         # raw byte
```

### 3.2 Compound Types

```
float[N]                    # 1D array, dimension N
float[M, N]                 # 2D array, dimensions M × N
image[H, W, C]              # semantic alias for 3D array (height, width, channels)

record{name: string, value: float}   # named record / struct
option<float>               # nullable type
either<result, error>       # sum type for error handling
```

### 3.3 Stream Types

```
stream<float[N]>            # stream of float arrays
stream<record{...}>         # stream of records
stream<image[H, W, 3]>      # stream of images
```

### 3.4 Dimensional Types

Dimensions are a first-class concept. They tell the compiler which axes of data a stage operates over, enabling automatic parallelism inference.

```
# This function operates per-element — compiler parallelizes across all dimensions
stage scale(x: float) -> float:
    return x * 2.0

# This function operates per-row — compiler parallelizes across rows
stage row_normalize(row: float[N]) -> float[N]:
    return row / sum(row)

# This function operates on the full array — limited parallelism
stage global_mean(data: float[M, N]) -> float:
    return sum(data) / (M * N)
```

The compiler infers from type signatures:
- `scale` takes a scalar → can be mapped across any array dimension → embarrassingly parallel
- `row_normalize` takes a 1D slice → parallelizable across the other dimension
- `global_mean` takes the full array → reduction, limited parallelism

### 3.5 Purity Tracking

The type system tracks side effects:

```
pure stage double(x: float) -> float       # default: pure
effect stage write_log(msg: string) -> void  # explicitly impure
```

Pure stages can be freely parallelized, reordered, and memoized. Impure stages must maintain execution order and are typically CPU-bound.

---

## 5. Control Flow & Dataflow Semantics

### 4.1 Conditionals

```
stream result = data |> branch(
    condition: is_positive,
    true_path:  |> scale(2.0),
    false_path: |> scale(-1.0)
) |> merge()
```

The compiler sees a dataflow split/merge pattern. If the condition is data-independent (same condition for all elements), both paths can be compiled as GPU kernels with masking. If data-dependent, the compiler evaluates whether GPU predication or CPU branching is more efficient.

### 4.2 Iteration

```
# Bounded iteration — compiler knows trip count
stream result = data |> repeat(100, |> refine())

# Convergence iteration — runs until condition met
stream result = data |> until(converged, |> iterate_step())
```

Bounded iteration with independent iterations is parallelized. Bounded iteration with dependencies is pipelined. Convergence iteration is inherently sequential on the convergence check but may parallelize within each iteration.

### 4.3 Recursion

```
stage tree_sum(node):
    if is_leaf(node):
        return node.value
    else:
        left = tree_sum(node.left)
        right = tree_sum(node.right)
        return left + right
```

The compiler analyzes recursion structure. Divide-and-conquer recursion (where branches are independent) can be parallelized — each recursive branch maps to parallel GPU work. Linear recursion (each call depends on the previous) is sequential.

### 4.4 Pipeline Composition

Pipelines are first-class — they can be named, stored, and composed:

```
pipeline preprocess = |> drop_null() |> cast_types() |> normalize()
pipeline analyze = |> group_by("symbol") |> agg(mean, std)

stream result = raw_data |> preprocess |> analyze
```

This enables reusable pipeline fragments as the building blocks of module composition.

---

## 6. Vanilla Standard Library

Vanilla is the canonical standard library. It ships with the language and is always available. Vanilla serves three roles:

1. **Reference implementations** — correct, portable, not necessarily fastest
2. **Interface definitions** — defines the type signatures and behavioral contracts that modules must conform to
3. **Fallback** — if no module is installed for an operation, vanilla handles it

### 5.1 Update Policy

Vanilla is updated only when:
- GPU architecture changes require new primitive mappings (e.g., new memory hierarchy, new instruction sets)
- Correctness bugs are discovered
- New fundamental operations are added to the language

Vanilla does NOT update for:
- Performance improvements (that's what modules are for)
- Domain-specific operations (those are modules from the start)
- Stylistic or API design changes (stability is paramount)

### 5.2 Vanilla Operations Catalog

#### Numeric Operations
```
add, subtract, multiply, divide, modulo
power, sqrt, abs, negate
floor, ceil, round, truncate
min, max, clamp
```

#### Trigonometric & Transcendental
```
sin, cos, tan, asin, acos, atan, atan2
exp, log, log2, log10
sinh, cosh, tanh
```

#### Comparison & Logic
```
equal, not_equal, less, greater, less_equal, greater_equal
and, or, not, xor
```

#### Array Operations
```
map          # apply function per element
filter       # select elements by predicate
reduce       # many → one (sum, product, etc.)
scan         # prefix sum / running accumulation
sort         # sort by key or value
unique       # deduplicate
reshape      # change array dimensions
transpose    # swap dimensions
concat       # join arrays
slice        # extract sub-array
zip          # combine multiple arrays element-wise
enumerate    # attach index to each element
flatten      # reduce dimensionality
chunk        # split into fixed-size pieces
```

#### Aggregation
```
sum, mean, median, std, variance
min, max, count
skew, kurtosis
quantile, percentile
group_by     # partition by key
agg          # apply multiple aggregations
```

#### Data Handling
```
drop_null    # remove null/missing values
fill_null    # replace nulls with value/strategy
cast         # type conversion
merge        # combine streams by key
split        # divide stream by condition
sample       # random sampling
```

#### Temporal Operations
```
window       # sliding/tumbling window
lag          # shift by N time steps
lead         # forward shift by N time steps
diff         # difference between consecutive elements
cumsum       # cumulative sum
cumprod      # cumulative product
decay        # exponential decay (EMA-style)
rolling      # rolling window aggregation
```

#### I/O Boundary
```
tap          # data input (defined in core language)
emit         # data output (defined in core language)
```

### 5.3 Vanilla Interface Definitions

Each vanilla operation defines an **interface** — a typed behavioral contract that modules can implement. An interface specifies:

```yaml
interface: smoother
description: "Smooths a time series by applying a decay/averaging function"
signature:
  inputs:
    data: "stream<float[N]>"
    factor: "float"
  outputs:
    result: "stream<float[N]>"
properties:
  purity: pure
  temporal: true
  dimensions_parallel: [N]       # parallelizable across N
  dimensions_sequential: [time]  # sequential across time
correctness_tests:
  - input: [1.0, 2.0, 3.0, 4.0, 5.0], factor: 0.5
    output: [1.0, 1.5, 2.25, 3.125, 4.0625]
  - input: [10.0, 10.0, 10.0], factor: 1.0
    output: [10.0, 10.0, 10.0]
```

Any module claiming to implement `smoother` must:
- Accept and return the declared types
- Pass all correctness tests
- Respect the declared purity and dependency properties

---

## 7. Standard Modules (Ships with Language)

Standard modules are not vanilla — they don't define GPU-accelerated compute operations. Instead, they provide the general-purpose programming infrastructure that makes OctoFlow a practical language for everyday use. These ship with the language installer and are always available via `import std.*`.

### 7.1 std.io — File System & I/O

```flow
import std.io

// Reading
let text = read_file("data.txt")?                    // read entire file as string
let bytes = read_bytes("image.png")?                  // read as raw bytes
let lines = read_lines("log.txt")?                    // read as list of strings
stream rows = tap("large_file.csv", format=csv)       // stream for large files

// Writing
write_file("output.txt", content)?
write_bytes("output.bin", data)?
append_file("log.txt", "New entry\n")?

// File system
let exists = file_exists("config.toml")
let files = list_dir("/data/inputs")?
let info = file_info("data.csv")?                     // size, modified, permissions
create_dir("output/reports")?
copy_file("src.txt", "dst.txt")?
move_file("old.txt", "new.txt")?
delete_file("temp.txt")?

// Path manipulation
let full = path.join(base_dir, "subdir", "file.txt")
let ext = path.extension("report.pdf")                // "pdf"
let stem = path.stem("report.pdf")                    // "report"
let parent = path.parent("/data/inputs/file.csv")     // "/data/inputs"
```

### 7.2 std.string — String Operations

```flow
import std.string

let upper = "hello" |> to_upper()                     // "HELLO"
let trimmed = "  spaces  " |> trim()                  // "spaces"
let parts = "a,b,c" |> split(",")                     // ["a", "b", "c"]
let joined = ["a", "b", "c"] |> join(", ")            // "a, b, c"
let found = "hello world" |> contains("world")        // true
let replaced = "foo bar" |> replace("foo", "baz")     // "baz bar"
let sub = "hello" |> substring(0, 3)                  // "hel"
let len = "hello" |> length()                         // 5
let starts = "hello" |> starts_with("hel")            // true
let padded = "42" |> pad_left(5, '0')                 // "00042"
```

All string operations are CPU-targeted. The compiler routes them to CPU automatically.

### 7.3 std.json — JSON Processing

```flow
import std.json

// Parse
let data = json.parse('{"name": "OctoFlow", "version": 1}')?
let name = data["name"]                               // "OctoFlow"

// Generate
let obj = map{"name": "OctoFlow", "features": list["gpu", "dataflow"]}
let text = json.stringify(obj, pretty=true)

// Stream processing
stream records = tap("data.jsonl") |> json.parse_lines()
```

### 7.4 std.csv — CSV Processing

```flow
import std.csv

// Read with headers
let table = csv.read("data.csv", headers=true)?
let prices = table["price"] |> cast(float)            // column access → GPU array

// Write
csv.write("output.csv", result_table, headers=true)?

// Streaming for large files
stream rows = tap("huge.csv") |> csv.parse_stream(headers=true)
```

### 7.5 std.datetime — Date & Time

```flow
import std.datetime

let now = datetime.now()
let today = datetime.today()
let ts = datetime.parse("2026-02-15T10:30:00Z", format=iso8601)?
let formatted = ts |> datetime.format("YYYY-MM-DD")
let diff = datetime.between(start, end)               // Duration
let tomorrow = today |> datetime.add(days=1)
let weekday = today |> datetime.day_of_week()         // "Saturday"

// Timezone
let manila = datetime.now(tz="Asia/Manila")
let utc = manila |> datetime.to_utc()
```

### 7.6 std.net.http — HTTP Client

```flow
import std.net.http

// Simple requests
let response = http.get("https://api.example.com/data")?
let body = response.body |> json.parse()?

// With headers and parameters
let response = http.get("https://api.example.com/search",
    params=map{"q": "flowgpu", "limit": 10},
    headers=map{"Authorization": "Bearer {token}"}
)?

// POST
let response = http.post("https://api.example.com/submit",
    body=json.stringify(payload),
    content_type="application/json"
)?

// Download
http.download("https://example.com/data.zip", "local/data.zip")?
```

### 7.7 std.os — Operating System Interface

```flow
import std.os

// Environment
let home = os.env("HOME")?
let all_env = os.env_all()
os.set_env("MY_VAR", "value")

// Process
let result = os.exec("ls", args=["-la"])?
let output = result.stdout
let code = result.exit_code

// System info
let platform = os.platform()          // "linux", "macos", "windows"
let arch = os.arch()                  // "x86_64", "aarch64"
let cpus = os.cpu_count()
let mem = os.total_memory()

// Current working directory
let cwd = os.cwd()
os.chdir("/data")?
```

### 7.8 std.args — CLI Argument Parsing

```flow
import std.args

// Simple
let args = args.parse()
let input_file = args.positional(0)?
let verbose = args.flag("verbose", short="v", default=false)
let output = args.option("output", short="o", default="result.csv")

// Structured (with auto-generated help)
let cli = args.program("flowgpu-analyzer")
    |> args.description("Analyze trading data")
    |> args.positional("input", help="Input data file")
    |> args.option("output", short="o", help="Output file", default="out.csv")
    |> args.option("window", short="w", help="Window size", type=int, default=50)
    |> args.flag("verbose", short="v", help="Verbose output")
    |> args.parse()?
```

### 7.9 std.fmt — Formatting

```flow
import std.fmt

let s = fmt.format("{:.2f}", 3.14159)                 // "3.14"
let table = fmt.table(data, columns=["Name", "Price", "Change"])
let bar = fmt.progress_bar(current, total)
let colored = fmt.color("Error!", "red")               // terminal color
let size = fmt.bytes(1048576)                          // "1.0 MB"
let num = fmt.comma(1000000)                           // "1,000,000"
```

### 7.10 std.regex — Regular Expressions

```flow
import std.regex

let pattern = regex.compile(r"\d{4}-\d{2}-\d{2}")?
let matches = "Date: 2026-02-15" |> regex.find(pattern)
let valid = "test@email.com" |> regex.matches(r"^[\w.]+@[\w.]+\.\w+$")
let replaced = text |> regex.replace(r"\s+", " ")
let groups = "John Doe, 42" |> regex.capture(r"(\w+) (\w+), (\d+)")
```

### 7.11 std.math — Extended Math (beyond vanilla)

```flow
import std.math

// Constants
let pi = math.PI
let e = math.E

// Functions beyond vanilla's GPU-optimized set
let result = math.factorial(10)
let combo = math.choose(52, 5)
let gcd = math.gcd(48, 18)
let prime = math.is_prime(997)
let fib = math.fibonacci(20)
```

### 7.12 std.random — Random Number Generation

```flow
import std.random

let n = random.int(1, 100)                            // random integer
let f = random.float(0.0, 1.0)                        // random float
let choice = random.choice(list["red", "blue", "green"])
let shuffled = items |> random.shuffle()
let sample = data |> random.sample(100)               // 100 random elements

// Seeded for reproducibility
let rng = random.seed(42)
let reproducible = rng.float(0.0, 1.0)
```

### 7.13 std.crypto — Hashing & Basic Crypto

```flow
import std.crypto

let hash = crypto.sha256("hello world")
let md5 = crypto.md5(file_bytes)
let uuid = crypto.uuid_v4()
let encoded = crypto.base64_encode(data)
let decoded = crypto.base64_decode(encoded)?
```

### 7.14 std.log — Structured Logging

```flow
import std.log

log.configure(level="info", output="app.log", format=json)

log.debug("Processing batch {batch_id}")
log.info("Loaded {count} records from {source}")
log.warn("Missing data for {symbol}, using fallback")
log.error("Failed to connect to {host}: {err}")
```

---

## 8. Extended Module Ecosystem (On-Demand)

Extended modules are installed on demand via the OctoFlow package manager. They cover every domain a general-purpose language needs. This section defines the module categories and representative packages that the community (or core team) can build.

**Key principle**: Every extended module uses the same module system defined in Section 9. They conform to interfaces, pass correctness tests, get auto-benchmarked. The only difference from vanilla modules is that they don't ship pre-installed.

### 8.1 Domain Map

```
flowgpu add <module>              # install a module

TIER 1: PRACTICAL TOOLS (first year priorities)
─────────────────────────────────────────────────
  Database & Storage
    ext.db.sqlite                 # SQLite embedded database
    ext.db.postgres               # PostgreSQL client
    ext.db.redis                  # Redis client
    ext.db.mongo                  # MongoDB client
    ext.storage.s3                # AWS S3 / compatible object storage

  Web & Network
    ext.web.server                # HTTP server framework
    ext.web.websocket             # WebSocket client/server
    ext.web.graphql               # GraphQL client
    ext.net.tcp                   # raw TCP sockets
    ext.net.udp                   # raw UDP sockets
    ext.net.smtp                  # email sending
    ext.net.ssh                   # SSH client

  Data Formats
    ext.format.toml               # TOML parsing/writing
    ext.format.yaml               # YAML parsing/writing
    ext.format.xml                # XML parsing
    ext.format.parquet            # Apache Parquet (columnar, GPU-friendly)
    ext.format.arrow              # Apache Arrow (in-memory columnar)
    ext.format.msgpack            # MessagePack binary serialization
    ext.format.protobuf           # Protocol Buffers
    ext.format.excel              # XLSX read/write

  Testing & Development
    ext.test                      # test framework (assertions, suites, mocks)
    ext.bench                     # benchmarking framework
    ext.doc                       # documentation generator

TIER 2: APPLICATION FRAMEWORKS (year 1-2)
─────────────────────────────────────────────────
  Desktop GUI
    ext.ui.desktop                # native desktop GUI framework
    ext.ui.widgets                # common widgets (button, text, slider, table)
    ext.ui.layout                 # layout engine (flexbox-like)
    ext.ui.canvas                 # GPU-accelerated 2D drawing
    ext.ui.charts                 # chart/graph widgets (GPU-rendered)

  Mobile
    ext.mobile.android            # Android app framework
    ext.mobile.ios                # iOS app framework
    ext.mobile.cross              # cross-platform mobile (single codebase)

  Web Frontend
    ext.web.ui                    # web UI framework (compiles to WebGPU/WASM)
    ext.web.components            # reusable UI components
    ext.web.router                # client-side routing
    ext.web.state                 # state management

  CLI Applications
    ext.cli.tui                   # terminal UI (tables, progress bars, colors)
    ext.cli.interactive           # interactive prompts and menus
    ext.cli.repl                  # REPL/shell builder

TIER 3: DOMAIN-SPECIFIC (year 2+, largely community)
─────────────────────────────────────────────────
  Machine Learning & AI
    ext.ml.tensor                 # tensor operations (GPU-accelerated)
    ext.ml.autograd               # automatic differentiation
    ext.ml.nn                     # neural network layers
    ext.ml.train                  # training loops and optimizers
    ext.ml.onnx                   # ONNX model import/export
    ext.ml.llm                    # LLM inference (local models)

  Scientific Computing
    ext.sci.linalg                # advanced linear algebra (GPU)
    ext.sci.fft                   # Fast Fourier Transform (GPU)
    ext.sci.ode                   # ODE solvers
    ext.sci.optimize              # optimization algorithms
    ext.sci.signal                # signal processing

  Finance
    ext.fin.market                # market data feeds
    ext.fin.backtest              # backtesting framework
    ext.fin.risk                  # risk calculations (VaR, CVaR)
    ext.fin.options               # options pricing (Black-Scholes, MC)
    ext.fin.orderbook             # order book management

  Image & Video
    ext.media.image               # image processing (GPU-accelerated)
    ext.media.video               # video encoding/decoding
    ext.media.audio               # audio processing
    ext.media.camera              # camera capture

  Game Development
    ext.game.engine               # 2D/3D game engine
    ext.game.physics              # physics simulation (GPU)
    ext.game.render               # GPU rendering pipeline
    ext.game.input                # input handling

  System & DevOps
    ext.sys.process               # advanced process management
    ext.sys.thread                # thread pool for CPU-bound work
    ext.sys.ffi                   # foreign function interface (call C/Rust)
    ext.sys.wasm                  # WebAssembly compilation target
    ext.sys.docker                # Docker container management
    ext.sys.cron                  # scheduled task execution

BRIDGES & INTEROP (core team + community)
─────────────────────────────────────────────────
  Compilation Targets
    ext.target.wasm               # compile to WebAssembly (browser, serverless)
    ext.target.js                 # transpile to JavaScript
    ext.target.py                 # transpile to Python module
    ext.target.rs                 # transpile to Rust library
    ext.target.c                  # transpile to C library
    ext.target.lib                # compile to shared library (.so/.dylib/.dll)

  Language Bridges (call external code from OctoFlow)
    ext.bridge.python             # call Python functions and libraries
    ext.bridge.rust               # call Rust functions via FFI
    ext.bridge.c                  # call C functions via FFI
    ext.bridge.js                 # call JavaScript (in WASM context)

  Protocol Bridges (expose OctoFlow to external systems)
    ext.bridge.rest               # expose as REST API (built on ext.web.server)
    ext.bridge.grpc               # expose as gRPC service
    ext.bridge.graphql            # expose as GraphQL endpoint
    ext.bridge.arrow_ipc          # expose via Apache Arrow IPC (zero-copy)
    ext.bridge.mqtt               # expose/consume via MQTT (IoT)
```

### 8.2 Compilation Targets & Bridge Architecture

OctoFlow's default output is a native binary (`.fgb`) with GPU acceleration. But real-world software lives in ecosystems — browsers, Python ML pipelines, Rust backends, mobile apps. Bridges let OctoFlow participate in these ecosystems without requiring everything to be rewritten.

**Compilation target matrix:**

```
flow build app.flow                          # native binary (default)
flow build app.flow --target wasm -o app.wasm  # WebAssembly
flow build app.flow --target js -o app.js      # JavaScript
flow build app.flow --target py -o app_module   # Python package
flow build app.flow --target rs -o app.rs      # Rust source
flow build app.flow --target c -o app.c        # C source
flow build app.flow --target lib -o libapp.so  # shared library
```

**What each target produces and when to use it:**

| Target | Output | GPU? | Use Case |
|--------|--------|------|----------|
| `fgb` (default) | Self-contained binary | SPIR-V/Vulkan | Production servers, CLI tools, desktop apps |
| `wasm` | `.wasm` + `.js` loader | WebGPU | Browser apps, dashboards, interactive tools |
| `js` | `.js` module | ❌ (CPU only) | Embed in existing web projects, Node.js |
| `py` | Python package with `.so` | Via Vulkan | Plug into ML pipelines, Jupyter, data science |
| `rs` | Rust source + Cargo.toml | Via Vulkan | Embed in Rust services, system programming |
| `c` | C source + header | Via Vulkan | Embed anywhere, maximum portability |
| `lib` | Shared library + C header | Via Vulkan | Any language with FFI (Java, Go, C#, Ruby) |

**Not every OctoFlow feature maps to every target.** The compiler warns when targeting a limited platform:

```
$ flow build gpu_heavy.flow --target js
WARNING: 4 stages use GPU acceleration which is not available 
         in the JavaScript target. These will run on CPU.
         Stages: normalize, detect_anomaly, compute_stats, sort_large
         Consider --target wasm for browser GPU via WebGPU.
```

### 8.3 Bridge Examples

#### WASM Target: Interactive Browser Dashboard

The most immediately useful bridge. OctoFlow computation runs in the browser with WebGPU for GPU access.

```flow
// dashboard.flow — compiles to WASM, runs in browser
import ext.target.wasm as wasm
import ext.web.ui as ui

// This function runs on the user's GPU via WebGPU
stage analyze(prices: stream<float[N]>) -> AnalysisResult:
    let sma_50 = prices |> rolling(50) |> mean()
    let sma_200 = prices |> rolling(200) |> mean()
    let vol = prices |> rolling(20) |> std()
    let sharpe = prices |> diff() |> mean() |> divide(vol)
    return AnalysisResult{
        sma_50: sma_50,
        sma_200: sma_200,
        volatility: vol,
        sharpe: sharpe
    }

// UI renders in browser
pub fn main():
    let data = wasm.fetch_json("/api/prices")?
    let prices = data["close"] |> to_array()
    let result = analyze(prices)

    ui.render(
        ui.column(list[
            ui.heading("Market Dashboard"),
            ui.chart(prices, type="candlestick"),
            ui.chart(list[result.sma_50, result.sma_200], 
                type="line", labels=list["SMA 50", "SMA 200"]),
            ui.stat_row(list[
                ui.stat("Volatility", "{result.volatility |> last():.2f}%"),
                ui.stat("Sharpe", "{result.sharpe |> last():.2f}")
            ])
        ])
    )
```

```bash
$ flow build dashboard.flow --target wasm -o dist/
# Produces: dist/dashboard.wasm, dist/dashboard.js, dist/index.html
# Open index.html in browser — GPU computation via WebGPU
```

#### Python Bridge: Plug OctoFlow into ML Pipeline

Use OctoFlow's GPU-accelerated stages inside a Python/Jupyter workflow.

```flow
// fast_features.flow — compiles to Python module
module fast_features

/// GPU-accelerated feature engineering for ML pipelines.
/// 10-100x faster than pandas for large datasets.

pub stage compute_features(
    prices: stream<float[N]>,
    volumes: stream<float[N]>
) -> stream<FeatureSet>:
    let returns = prices |> diff() |> divide(prices |> lag(1))
    let vol_20 = returns |> rolling(20) |> std()
    let vol_60 = returns |> rolling(60) |> std()
    let vwap = prices |> multiply(volumes) |> cumsum() 
        |> divide(volumes |> cumsum())
    let rsi = returns |> rolling(14) |> rsi_calc()
    let momentum = prices |> diff(periods=10)
    
    return FeatureSet{
        returns: returns,
        vol_20: vol_20,
        vol_60: vol_60,
        vol_ratio: vol_20 |> divide(vol_60),
        vwap: vwap,
        rsi: rsi,
        momentum: momentum
    }
```

```bash
$ flow build fast_features.flow --target py -o fast_features/
# Produces: fast_features/__init__.py, fast_features/_native.so
```

```python
# In Python / Jupyter notebook
import fast_features as ff
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data with pandas (Python ecosystem)
df = pd.read_csv("market_data.csv")

# Feature engineering with OctoFlow (GPU-accelerated, 100x faster)
features = ff.compute_features(
    prices=df["close"].values,
    volumes=df["volume"].values
)

# Back to Python for ML (scikit-learn ecosystem)
X = features.to_dataframe()
y = (df["close"].shift(-1) > df["close"]).astype(int)
model = RandomForestClassifier()
model.fit(X, y)
```

The Python bridge uses numpy arrays as the interchange format. OctoFlow stages run on GPU via Vulkan, results come back as numpy arrays. Zero-copy where possible via Arrow IPC.

#### Shared Library: Embed in Any Language

```flow
// analytics_lib.flow — compiles to shared library
module analytics_lib

pub stage moving_average(data: stream<float[N]>, window: int) -> stream<float[N]>:
    return data |> rolling(window) |> mean()

pub stage bollinger(data: stream<float[N]>, window: int, width: float) -> stream<BollingerResult>:
    let mid = data |> rolling(window) |> mean()
    let vol = data |> rolling(window) |> std()
    return BollingerResult{
        upper: mid |> add(vol |> multiply(width)),
        middle: mid,
        lower: mid |> subtract(vol |> multiply(width))
    }
```

```bash
$ flow build analytics_lib.flow --target lib -o libanalytics.so
# Also produces: analytics.h (C header)
```

```c
// C header (auto-generated)
#include "analytics.h"

float* flowgpu_moving_average(float* data, int n, int window);
BollingerResult* flowgpu_bollinger(float* data, int n, int window, float width);
void flowgpu_free(void* ptr);
```

Now callable from C, C++, Go, Java (JNI), C# (P/Invoke), Ruby (FFI), Swift, Kotlin — any language with C FFI support.

#### REST/WebSocket Bridge: OctoFlow as Microservice

```flow
// service.flow — runs as a standalone computation service
import std.json
import ext.web.server as web
import ext.bridge.arrow_ipc as arrow

let app = web.app()

// JSON API (simple, universal)
app |> web.post("/api/analyze", fn(req):
    let data = req.body |> json.parse()?
    let prices = data["prices"] |> to_array()
    let result = prices |> full_analysis()
    return web.response(200, json.stringify(result))
)

// Arrow IPC API (zero-copy, for high-performance clients)
app |> web.post("/api/analyze/arrow", fn(req):
    let table = arrow.deserialize(req.body)?
    let prices = table |> column("price")
    let result = prices |> full_analysis()
    return web.response(200, arrow.serialize(result))
)

// WebSocket for streaming (real-time)
app |> web.websocket("/ws/stream", fn(conn):
    stream feed = tap("market_feed")
    stream signals = feed |> live_analysis()
    for signal in signals:
        conn |> web.send(json.stringify(signal))
)

web.listen(app, port=9000)?
```

Any client in any language connects via HTTP or WebSocket. The OctoFlow service does the GPU-heavy computation. The client just consumes JSON or Arrow.

### 8.4 Bridge Priority for v1

Not all bridges ship at once. Priority based on immediate utility:

```
v1.0 — Ships with language:
  ✅ ext.bridge.rest          REST API (already part of ext.web.server)
  ✅ ext.target.lib           Shared library output (C ABI)
  
v1.x — First extensions:
  [SOON] ext.target.wasm          Browser deployment via WebGPU
  [SOON] ext.bridge.python        Python interop (numpy arrays)
  [SOON] ext.bridge.arrow_ipc     Zero-copy data exchange

v2.0 — Full ecosystem:
  [PLANNED] ext.target.js            JavaScript transpilation
  [PLANNED] ext.target.py            Python package generation
  [PLANNED] ext.bridge.rust          Rust FFI
  [PLANNED] ext.bridge.c             C FFI  
  [PLANNED] ext.bridge.grpc          gRPC services
  [PLANNED] ext.target.rs            Rust source output
  [PLANNED] ext.target.c             C source output
```

The community will build additional bridges — Go, Java, C#, Ruby, Swift, Kotlin. The shared library target (`.so` with C header) makes this possible from day one, since every language has C FFI.

### 8.5 Example: Full CRUD Web Application

With the right extended modules installed, OctoFlow can build a complete web application:

```flow
import std.json
import std.log
import ext.web.server as web
import ext.db.sqlite as db

// Initialize
let database = db.open("app.db")?
db.exec(database, "CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL
)")?

// Define routes
let app = web.app()

// CREATE
app |> web.post("/users", stage handle_create(req):
    let body = req.body |> json.parse()?
    let result = db.exec(database,
        "INSERT INTO users (name, email) VALUES (?, ?)",
        params=[body["name"], body["email"]]
    )?
    return web.response(201, json.stringify(map{"id": result.last_id}))
)

// READ
app |> web.get("/users", stage handle_list(req):
    let users = db.query(database, "SELECT * FROM users")?
    return web.response(200, json.stringify(users))
)

app |> web.get("/users/:id", stage handle_get(req):
    let user = db.query_one(database,
        "SELECT * FROM users WHERE id = ?",
        params=[req.params["id"]]
    )?
    match user:
        some(u) -> web.response(200, json.stringify(u))
        none    -> web.response(404, json.stringify(map{"error": "Not found"}))
)

// UPDATE
app |> web.put("/users/:id", stage handle_update(req):
    let body = req.body |> json.parse()?
    db.exec(database,
        "UPDATE users SET name = ?, email = ? WHERE id = ?",
        params=[body["name"], body["email"], req.params["id"]]
    )?
    return web.response(200, json.stringify(map{"status": "updated"}))
)

// DELETE
app |> web.delete("/users/:id", stage handle_delete(req):
    db.exec(database, "DELETE FROM users WHERE id = ?",
        params=[req.params["id"]]
    )?
    return web.response(200, json.stringify(map{"status": "deleted"}))
)

// Start server
log.info("Starting server on :8080")
web.listen(app, port=8080)?
```

This is fully functional OctoFlow. No HTML, no CSS, no JavaScript. The web server is a OctoFlow extended module. The database is a OctoFlow extended module. The language handles everything.

### 8.6 Example: Desktop GUI Application

```flow
import std.io
import ext.ui.desktop as ui
import ext.ui.charts as charts

// Application state
let app = ui.app("Data Analyzer")

// Load data
var data = list[]
var stats = map{}

// Layout
let window = ui.window(title="Data Analyzer", size=(800, 600))

let sidebar = ui.panel(layout="vertical", width=200):
    ui.button("Load CSV", on_click=load_data)
    ui.button("Analyze", on_click=run_analysis)
    ui.separator()
    ui.label("Records: {data |> length()}")
    ui.label("Mean: {stats.get('mean', 'N/A')}")
    ui.label("Std: {stats.get('std', 'N/A')}")

let main_area = ui.panel(layout="vertical"):
    ui.tab_view():
        ui.tab("Chart"):
            charts.line(data, x="date", y="price", gpu=true)
        ui.tab("Table"):
            ui.data_table(data, sortable=true, filterable=true)

window |> ui.layout("horizontal", children=[sidebar, main_area])

// Event handlers
stage load_data():
    let path = ui.file_dialog(filter="*.csv")?
    data = std.csv.read(path, headers=true)?
    ui.refresh()

stage run_analysis():
    // This part runs on GPU automatically — it's array math
    let prices = data["price"] |> cast(float)
    stats = map{
        "mean": prices |> mean(),
        "std": prices |> std(),
        "min": prices |> min(),
        "max": prices |> max(),
        "sharpe": prices |> diff() |> mean() |> divide(prices |> diff() |> std())
    }
    ui.refresh()

// Run
ui.run(app, window)
```

The chart rendering is GPU-accelerated. The statistical computations are GPU-accelerated. The GUI framework is a native desktop module (not HTML). OctoFlow's compiler knows that button clicks are CPU events and array math is GPU work — it handles the routing automatically.

### 8.7 Example: Mobile App

```flow
import ext.mobile.cross as mobile
import std.net.http
import std.json

let app = mobile.app("Price Tracker")

// Screen definitions
mobile.screen("home"):
    mobile.list(items=watchlist, render=stage render_item(item):
        mobile.row():
            mobile.text(item.symbol, style="bold")
            mobile.text("{item.price:.2f}", style="mono")
            mobile.text("{item.change:+.1f}%",
                color=if item.change >= 0 then "green" else "red"
            )
    )
    mobile.fab(icon="add", on_tap=add_symbol)

mobile.screen("detail", params=["symbol"]):
    let history = fetch_history(params.symbol)?
    // GPU-accelerated chart rendering on mobile GPU
    mobile.chart(history, type="candlestick")
    mobile.stats_card(history |> compute_stats())

// Data
stage fetch_prices() -> list[PriceData]:
    let response = http.get("https://api.prices.com/batch",
        params=map{"symbols": watchlist |> join(",")}
    )?
    return response.body |> json.parse()?

// Refresh every 5 seconds
mobile.timer(interval=5000, action=fetch_prices)

// Run
mobile.run(app)
```

### 8.8 Example: System Automation Script

OctoFlow as a scripting language, replacing Python/Bash:

```flow
#!/usr/bin/env flowgpu
import std.io
import std.os
import std.fmt
import std.datetime

// Clean up old log files
let log_dir = os.env("LOG_DIR") |> unwrap_or("/var/log/myapp")
let cutoff = datetime.now() |> datetime.subtract(days=30)

let old_files = list_dir(log_dir)?
    |> filter(stage(f): f.extension == "log")
    |> filter(stage(f): f.modified < cutoff)

print("Found {old_files |> length()} files older than 30 days")

for file in old_files:
    print("  Deleting: {file.name} ({fmt.bytes(file.size)})")
    delete_file(file.path)?

print("Cleanup complete. Freed {old_files |> map(stage(f): f.size) |> sum() |> fmt.bytes()}")
```

### 8.9 The GPU Advantage in General-Purpose Use

The point of OctoFlow isn't that every program uses GPU. Most of the examples above are CPU-heavy (web servers, file I/O, string manipulation). The point is that **when computation gets heavy, GPU acceleration is automatic and invisible**.

A Python web server that needs to compute statistics on 10 million records will block for seconds. A OctoFlow web server does the exact same I/O on CPU, then silently offloads the statistics to GPU and returns in milliseconds.

```flow
// This web endpoint processes 10M records
// The server framework runs on CPU
// The computation automatically runs on GPU
// The programmer doesn't think about it
app |> web.get("/analytics", stage handle_analytics(req):
    let data = db.query(database, "SELECT * FROM trades")?  // CPU: database
    let prices = data["price"] |> cast(float)                // GPU: type conversion
    let stats = prices |> group_by(data["symbol"])           // CPU: grouping
        |> agg(mean, std, skew, kurtosis)                    // GPU: heavy math
        |> sort("sharpe", descending=true)                   // GPU: parallel sort
    return web.response(200, json.stringify(stats))           // CPU: serialization
)
```

The compiler draws the GPU/CPU line. The programmer writes normal code.

---

## 9. Module System Architecture

### 6.1 What Is a Module?

A module is a package that provides one or more stage implementations. Modules are:

- **Optional** — no module is ever required; vanilla handles everything
- **Installable** — added to a project explicitly by the user
- **Versioned** — semantic versioning, dependency resolution
- **Typed** — must conform to declared interfaces
- **Benchmarked** — performance data is machine-generated, not self-reported
- **Competitive** — multiple modules can implement the same interface; users/compiler choose

### 6.2 Module Structure

A module is a directory with a defined structure:

```
fast_ema/
├── flow.manifest.yaml    # metadata, interface declarations, dependencies
├── src/
│   └── ema.flow          # implementation source code
├── tests/
│   └── test_ema.flow     # additional tests beyond vanilla conformance
└── bench/
    └── bench_ema.flow    # benchmark definitions
```

### 6.3 Module Manifest

```yaml
# flow.manifest.yaml

name: "fast_ema"
version: "1.2.0"
description: "GPU-optimized EMA using fused temporal reduction"
author: "mike_d"
license: "MIT"
repository: "https://github.com/octoflow/fast_ema"

# What vanilla interface(s) this module implements
implements:
  - interface: "vanilla.smoother"
    operation: "ema"

# Authoring level
level: "custom_stage"   # composition | custom_stage | native

# Type signature (must match interface)
signature:
  inputs:
    data: "stream<float[N]>"
    decay: "float"
  outputs:
    result: "stream<float[N]>"

# Parallel profile (used by compiler for GPU/CPU decisions)
parallel_profile:
  parallelizable_dimensions: [N]
  temporal_dependency: true
  purity: pure
  arithmetic_intensity: "medium"
  memory_access_pattern: "coalesced"

# Dependencies
dependencies:
  flowgpu: ">= 0.1.0"
  vanilla: ">= 0.1.0"

# Optional: hardware-specific builds
targets:
  - nvidia_sm80+    # Ampere and later
  - amd_gfx1100+    # RDNA 3 and later
  - cpu_io           # CPU path for I/O-only targets
```

---

## 10. Module Authoring

### 7.1 Three Levels of Module Authoring

Modules can be authored at three levels of complexity and control. Each level uses progressively more language features but all produce modules with identical standing in the ecosystem.

---

### Level 1: Composition Modules

**Who:** Any language user. Zero extra knowledge required.

**What:** Compose existing vanilla or module operations into reusable pipeline fragments. No new computational logic.

**Example: Golden Cross signal detector**

```
# src/golden_cross.flow
# Implements: vanilla.signal_generator

module golden_cross

import vanilla.decay
import vanilla.crossover

export stage detect(prices: stream<float[N]>) -> stream<bool[N]>:
    fast = prices |> decay(period=50)
    slow = prices |> decay(period=200)
    return fast |> crossover(slow)
```

The compiler handles all GPU/CPU decisions. The module author just connects existing operations. This is the 80% path — most community contributions are compositions.

---

### Level 2: Custom Stage Modules

**Who:** Developers comfortable writing algorithmic logic.

**What:** Write new stage functions using vanilla primitives internally. Custom computation, but still in the dataflow language.

**Example: Volume-weighted EMA**

```
# src/vwema.flow
# Implements: vanilla.smoother

module volume_weighted_ema

import vanilla.{scale, sum, multiply, decay}

export stage vwema(
    prices: stream<float[N]>,
    volumes: stream<float[N]>,
    factor: float
) -> stream<float[N]>:
    # Normalize volumes to weights
    weights = volumes |> scale(1.0 / sum(volumes))
    # Apply weights to prices
    weighted = prices |> multiply(weights)
    # Apply temporal decay
    return weighted |> temporal decay(factor)
```

The author writes computation logic but stays within the dataflow language. The compiler still handles all GPU/CPU mapping. More skill than Level 1 but no hardware knowledge needed.

---

### Level 3: Native Modules

**Who:** Performance engineers, GPU specialists.

**What:** Drop down to a low-level API for hand-optimized GPU kernels. This is the escape hatch for when auto-compilation isn't fast enough.

**Example: Tiled matrix multiply with shared memory**

```
# src/fast_matmul.flow
# Implements: vanilla.matmul

module ultra_matmul

@native(gpu)
export stage matmul(
    a: matrix<float32>[M, K],
    b: matrix<float32>[K, N]
) -> matrix<float32>[M, N]:

    TILE = 32

    # Explicit shared memory allocation
    shared a_tile: float32[TILE, TILE]
    shared b_tile: float32[TILE, TILE]

    # Explicit thread block mapping
    for tile_idx in tiles(K, TILE):
        a_tile = load_tile(a, block_row, tile_idx, TILE)
        b_tile = load_tile(b, tile_idx, block_col, TILE)
        sync_threads()

        result += a_tile @ b_tile
        sync_threads()

    return result
```

This is raw GPU programming within the OctoFlow ecosystem. The module still conforms to the `vanilla.matmul` interface, still has the same type signature, still passes correctness tests — but the implementation is hand-tuned.

**Native modules must provide a CPU path for headless/no-GPU targets** — either a separate CPU implementation or a declaration that vanilla should be used:

```yaml
# In flow.manifest.yaml
cpu_fallback: "vanilla.matmul"   # use vanilla when GPU not available
```

---

### 7.2 Module Authoring Workflow

```
1. Initialize module project
   $ flow module init my_module

2. Edit flow.manifest.yaml
   - Declare interface, types, parallel profile

3. Write implementation in src/
   - Level 1: compose existing stages
   - Level 2: write custom stages
   - Level 3: write native GPU code

4. Test locally
   $ flow test
   - Runs module's own tests
   - Runs vanilla conformance tests (if implementing interface)
   - Reports correctness pass/fail

5. Benchmark locally
   $ flow bench
   - Runs against vanilla baseline
   - Reports relative performance on local hardware

6. Publish
   $ flow publish
   - Uploads to registry
   - Triggers automated validation pipeline
```

---

## 11. Module Registry & Community Contribution

### 8.1 Registry Architecture

The module registry is the central (or federated) repository where modules are published, discovered, and installed. Analogous to crates.io (Rust) or PyPI (Python).

### 8.2 Publishing Pipeline

When a module is published, the registry runs an automated validation pipeline:

```
Module uploaded
    ↓
Step 1: Manifest Validation
  - Is flow.manifest.yaml well-formed?
  - Are all required fields present?
  - Is the version number valid (semver)?
  - Are declared dependencies available?
    ↓
Step 2: Compilation Check
  - Does the module compile on reference targets?
    - NVIDIA reference GPU (e.g., RTX 4060 or equivalent)
    - AMD reference GPU (e.g., RX 7600 or equivalent)
    - CPU-only (x86_64)
  - Are there compilation errors or warnings?
    ↓
Step 3: Type Conformance
  - Does the module's type signature match the declared interface?
  - Are input/output types compatible?
  - Is the parallel profile consistent with the implementation?
    ↓
Step 4: Correctness Testing
  - Run vanilla conformance test suite
  - All correctness tests must pass
  - Output must be numerically equivalent to vanilla
    (within declared tolerance for floating-point)
  - Edge cases: empty input, single element, maximum size
    ↓
Step 5: Automated Benchmarking
  - Run on registry's reference hardware
  - Compare throughput vs. vanilla baseline
  - Measure across multiple data sizes (100, 10K, 1M, 100M elements)
  - Record GPU memory usage
  - Results are machine-generated, not self-reported
    ↓
Step 6: Publication
  - Module is live and discoverable
  - Benchmark results are attached to the listing
  - Conformance badge: "vanilla-conformant" ✓
```

### 8.3 Module Discovery

Users and LLM frontends discover modules through:

**CLI search:**
```
$ flow search ema
  vanilla.decay          (built-in)  canonical EMA implementation
  fast_ema v1.2.0        (module)    1.8x faster on GPU, vanilla-conformant ✓
  parallel_ema v0.9.1    (module)    optimized for > 100K instruments ✓
  approx_ema v2.0.0      (module)    approximate, 5x faster, ±0.1% accuracy ✓
```

**Registry web interface** with:
- Search by operation name, interface, tag, domain
- Sort by performance, downloads, rating, recency
- Filter by target hardware, conformance status, authoring level
- Benchmark comparison charts across modules

**Programmatic API** (for LLM frontends):
```json
GET /api/search?interface=vanilla.smoother&min_speedup=1.5&target=nvidia

[
  {
    "name": "fast_ema",
    "version": "1.2.0",
    "speedup_vs_vanilla": 1.8,
    "conformant": true,
    "gpu_memory_mb": 12,
    "downloads": 4521
  }
]
```

The LLM frontend uses this API during intent generation to recommend optimal modules for the user's workload.

### 8.4 Module Installation

```
$ flow add fast_ema                    # install latest
$ flow add fast_ema@1.2.0             # install specific version
$ flow add fast_ema --target nvidia    # install with NVIDIA-optimized build
$ flow remove fast_ema                 # uninstall
```

Installed modules are recorded in the project's dependency file:

```yaml
# flow.lock.yaml
dependencies:
  fast_ema: "1.2.0"
  advanced_stats: "3.1.0"
```

### 8.5 Trust & Security

Since modules compile to GPU/CPU code that runs on the user's machine:

- **Source-available required**: All published modules must include source code. No binary-only modules.
- **Compilation is local**: The registry stores source; compilation happens on the user's machine (or in a sandboxed build environment). Users never download pre-compiled GPU binaries from untrusted sources.
- **Audit trail**: Every published version is immutable. Versions cannot be silently updated. Yanking a version is possible but leaves a record.
- **Community review**: Optional but encouraged. High-trust modules can earn a "reviewed" badge through community code review.
- **Sandboxed execution**: Native (Level 3) modules are the highest risk. The registry flags native modules clearly, and the compiler can optionally restrict what native code can access.

---

## 12. Module Resolution

When a user writes `data |> ema(0.94)` and multiple EMA implementations are available, the compiler resolves which to use.

### 9.1 Resolution Priority (highest to lowest)

```
1. Explicit import
   import fast_ema.ema          # user chose explicitly → use this

2. Project configuration
   # flow.config.yaml
   prefer:
     smoother: "fast_ema"       # project-level preference → use this

3. Compiler auto-selection
   Based on:
     - Target hardware (which module has best benchmarks for this GPU?)
     - Data size (which module is optimized for this scale?)
     - Vanilla conformance (only consider conformant modules)
   
4. Vanilla fallback
   If no module is installed or all resolution fails → use vanilla
```

### 9.2 Auto-Selection Strategy

When the compiler auto-selects, it uses the module's declared parallel profile and registry benchmarks:

```
For each candidate module:
  score = benchmark_speedup(module, target_hardware, estimated_data_size)
        × conformance_weight(1.0 if conformant, 0.0 if not)
        × compatibility_weight(1.0 if target supported, 0.5 if fallback)

Select module with highest score.
If best score ≤ 1.0 (no module beats vanilla): use vanilla.
```

### 9.3 Frontend Integration

The LLM frontend participates in module resolution during intent generation:

```
Human: "smooth my price data, I have 50,000 instruments"
    ↓
LLM pre-prompt includes operation catalog + module registry API
    ↓
LLM queries: GET /api/search?interface=smoother&data_scale=50000&target=nvidia
    ↓
Registry returns: fast_ema (1.8x), parallel_ema (2.3x for >10K)
    ↓
LLM generates intent with: operation: "ema", module_hint: "parallel_ema"
    ↓
Transpiler generates: import parallel_ema.ema
                      stream result = prices |> ema(0.94)
```

The LLM acts as a module discovery and recommendation engine. The user gets optimized code without knowing the module ecosystem exists. But vanilla is always the fallback — if the user is offline, if no modules are installed, if the LLM doesn't suggest one — vanilla handles it.

---

## 13. Failsafe System & Pre-flight Validation

### 10.1 Design Philosophy

The language promises "you don't think about hardware." That promise extends to failure modes. If the user never manages GPU memory, they should also never experience GPU OOM crashes. If the user never manages numeric types, they should also never experience silent overflow corruption.

**The safety guarantee is architectural, not optional.** It is not a linter you can turn off. It is a pre-compiler layer that every module definition passes through before it can compile or execute. The user never sees a crash — they see a pre-flight report with actionable suggestions.

**Safety flows upward from vanilla:**

```
Vanilla (proven safe, pre-verified before shipping)
    ↓ composition inherits safety
Level 1 Composition Modules (safe by construction)
    ↓ vanilla primitives used internally
Level 2 Custom Stage Modules (safe if using vanilla primitives)
    ↓ validated at registry
Level 3 Native Modules (author's responsibility + registry validation)
```

A Level 1 composition module that only connects vanilla operations is **safe by construction**. It cannot overflow because vanilla's `multiply` already handles overflow. It cannot OOM because vanilla's `reduce` already estimates memory. Safety propagates through composition automatically. The module author does not think about safety — they inherit it.

### 10.2 Threat Model

| Failure Mode | Severity | Where Caught |
|-------------|----------|-------------|
| GPU out-of-memory | Critical | Pre-flight memory estimation + runtime guard |
| CPU out-of-memory | Critical | Pre-flight memory estimation + runtime guard |
| Numeric overflow/underflow | High | Pre-flight numeric analysis + runtime detection |
| NaN propagation | High | Pre-flight NaN source analysis + runtime tracking |
| Silent precision loss | Medium | Pre-flight type chain analysis + compile-time warning |
| Unbounded stream growth | Critical | Pre-flight bounds analysis |
| Circular dependency / deadlock | Critical | Pre-flight dependency analysis |
| Backpressure stall | High | Pre-flight producer/consumer rate analysis + runtime buffering |
| Infinite accumulation | Critical | Pre-flight bounds analysis |
| Division by zero | High | Vanilla guards on all division ops + pre-flight analysis |
| Index out of bounds | High | Vanilla bounds checking on all array access |
| Data transfer stall | Medium | Runtime timeout + fallback |

### 10.3 Pre-flight Validation Pipeline

Every module definition — whether written by hand, composed in a GUI, generated by an LLM, or written in markdown — passes through the pre-flight pipeline before compilation.

```
Module definition (markdown / GUI / code / LLM-generated)
    ↓
┌─────────────────────────────────────────────────────┐
│  PRE-FLIGHT VALIDATION                               │
│                                                       │
│  Check 1: Operation Existence                         │
│    Do all referenced operations exist in vanilla      │
│    or installed modules?                              │
│                                                       │
│  Check 2: Type Chain Validation                       │
│    Do types flow correctly through the pipeline?      │
│    Are all pipe connections type-compatible?           │
│    Are implicit casts safe (no precision loss)?        │
│                                                       │
│  Check 3: Memory Estimation                           │
│    For known/estimated input sizes:                   │
│    - Calculate per-stage memory requirements           │
│    - Calculate intermediate stream buffer sizes        │
│    - Calculate peak GPU memory usage                   │
│    - Calculate peak CPU memory usage                   │
│    - Compare against available hardware                │
│                                                       │
│  Check 4: Numeric Safety Analysis                     │
│    - Identify operations that can produce NaN          │
│      (division, sqrt of negative, log of zero)        │
│    - Trace overflow potential through arithmetic chains │
│    - Flag precision loss from type narrowing           │
│    - Identify accumulation operations that could       │
│      exceed type range over many iterations            │
│                                                       │
│  Check 5: Bounds Analysis                             │
│    - Are all streams bounded or properly windowed?     │
│    - Do accumulations on infinite streams have         │
│      emission points?                                 │
│    - Are buffer sizes bounded?                         │
│                                                       │
│  Check 6: Dependency Analysis                         │
│    - Is the dataflow graph acyclic (except temporal)?  │
│    - Are temporal dependencies properly marked?        │
│    - Are there implicit ordering constraints?          │
│                                                       │
│  Check 7: Backpressure Analysis                       │
│    - Can any producer outrun its consumer?             │
│    - Are buffer overflow policies defined?             │
│    - Is there risk of memory growth from unbounded     │
│      buffering?                                       │
│                                                       │
└─────────────────────────────────────────────────────┘
    ↓
Pre-flight Report
    ↓
Only on PASS → Transpiler → Compiler → Execution
```

### 10.4 Pre-flight Report

The pre-flight report is human-readable (and LLM-readable) output that tells the user exactly what was checked and what the results are.

**Passing report:**

```
╔══════════════════════════════════════════════════╗
║  PRE-FLIGHT REPORT: volatility_band module       ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  ✓ Operations    All 6 operations found in       ║
║                  vanilla                         ║
║  ✓ Type chain    float[N] throughout, consistent ║
║  ✓ Memory        Est. 16MB GPU for N=50,000      ║
║                  (within 8GB budget)             ║
║  ✓ Numeric       No overflow risk at float32     ║
║                  NaN guarded on window edges     ║
║  ✓ Bounds        All streams bounded             ║
║  ✓ Dependencies  Acyclic, 1 temporal dependency  ║
║  ✓ Backpressure  Balanced pipeline               ║
║                                                  ║
║  STATUS: READY TO COMPILE                        ║
╚══════════════════════════════════════════════════╝
```

**Failing report:**

```
╔══════════════════════════════════════════════════╗
║  PRE-FLIGHT REPORT: huge_analysis module         ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  ✓ Operations    All operations found            ║
║  ✓ Type chain    Valid                           ║
║  ✗ Memory        Est. 48GB GPU required          ║
║                  Stage 3: rolling(window=2000)   ║
║                  on 1M instruments = 8B floats   ║
║                  EXCEEDS 8GB available           ║
║                                                  ║
║  SUGGESTIONS:                                    ║
║  → Reduce window size to < 250 for GPU fit       ║
║  → Add chunking: process 125K instruments/batch  ║
║  → Force CPU: @device(cpu) on stage 3            ║
║  → The compiler can auto-chunk if you add:       ║
║    @max_memory(4GB) to the pipeline              ║
║                                                  ║
║  ✓ Numeric       No issues                       ║
║  ✓ Bounds        Bounded                         ║
║  ✓ Dependencies  Valid                           ║
║  ⚠ Backpressure  Stage 2 may outrun stage 3     ║
║                  (auto-buffered, max 100MB)       ║
║                                                  ║
║  STATUS: BLOCKED — fix memory issue              ║
╚══════════════════════════════════════════════════╝
```

The report includes **actionable suggestions**, not just error messages. The user (or LLM) can read the suggestions and fix the issue without understanding GPU memory architecture.

### 10.5 Vanilla Safety Guarantees

Every vanilla operation ships with built-in safety properties. These are not optional — they are part of the vanilla implementation.

**Memory safety:**
- Every vanilla operation declares its memory footprint formula as a function of input size
- `reduce(sum)` on `float32[N]` requires: `N × 4 bytes` input + `4 bytes` output + `~1KB` working memory
- `rolling(window=W)` on `float32[N]` requires: `N × W × 4 bytes` intermediate
- The pre-flight system uses these formulas for memory estimation

**Numeric safety:**
- `divide(a, b)`: guards against `b == 0`, returns configurable default (NaN, 0, infinity, or error)
- `sqrt(x)`: guards against `x < 0`, returns NaN with source tracking
- `log(x)`: guards against `x <= 0`
- `cast(float64 → float32)`: warns at pre-flight if value range exceeds float32 precision
- All integer arithmetic: overflow detection enabled by default

**NaN tracking:**
- When a vanilla operation produces a NaN, it records which stage and which input element caused it
- NaN tracking data propagates through the pipeline
- At the output tap, the runtime can report: "3 NaN values in output, originated at stage 'std()' on elements [0, 1, 2] due to insufficient window size"

**Bounds safety:**
- All array access in vanilla is bounds-checked
- All windowing operations handle edge cases (partial windows at stream start/end)
- All accumulations on infinite streams require an emission policy (emit every N elements, emit on condition, etc.)

### 10.6 Execution Modes

While the pre-flight system is always active, runtime checking has configurable intensity:

**Strict Mode (default for development):**
- All pre-flight checks active
- Runtime overflow detection on every arithmetic operation
- Runtime NaN tracking with full provenance
- Runtime memory monitoring with proactive warnings
- Runtime bounds checking on all array access
- Performance cost: ~10-30% slower than release mode

**Release Mode (for production):**
- All pre-flight checks active (always, non-negotiable)
- Runtime overflow detection: sampling-based (check every 1000th operation)
- Runtime NaN tracking: enabled but lightweight (stage-level, not element-level)
- Runtime memory monitoring: active (OOM prevention is never disabled)
- Runtime bounds checking: disabled for vanilla operations (already proven safe), enabled for native modules
- Performance cost: ~2-5% slower than unsafe mode

**Unsafe Mode (explicit opt-in for benchmarking):**
- Pre-flight checks still active (never disabled)
- All runtime checks disabled except memory monitoring
- Memory monitoring remains active: OOM crashes are never acceptable regardless of mode
- Requires explicit `@unsafe` annotation on the pipeline
- Performance: maximum, equivalent to hand-written GPU code

```
@unsafe  # required to enable unsafe mode
stream result = data |> heavy_computation()
```

**The key invariant across all modes:** Pre-flight validation is never skipped. Memory monitoring is never disabled. The user can trade numeric safety for performance, but they can never produce an OOM crash or an unvalidated pipeline.

### 10.7 Markdown-Based Module Definition

Modules can be defined in markdown with special syntax blocks. This is the most accessible authoring format — suitable for GUI editors, documentation, and LLM generation.

The markdown IS the module source. It is parsed by the pre-flight system directly.

**Example:**

```markdown
# Module: Volatility Band

## Metadata
- **implements**: vanilla.band_indicator
- **author**: mike_d
- **version**: 1.0.0
- **license**: MIT

## Inputs
| Name    | Type            | Description              |
|---------|-----------------|--------------------------|
| prices  | stream<float[N]>| Price data per instrument |
| window  | int             | Lookback period          |
| width   | float           | Band width multiplier    |

## Pipeline

```flow
windowed  = prices |> rolling(window)
vol       = windowed |> std()
mid       = windowed |> mean()
upper     = mid |> add(vol |> multiply(width))
lower     = mid |> subtract(vol |> multiply(width))
```

## Outputs
| Name  | Type            | Description    |
|-------|-----------------|----------------|
| upper | stream<float[N]>| Upper band     |
| lower | stream<float[N]>| Lower band     |
| mid   | stream<float[N]>| Middle band    |

## Notes
Standard volatility bands using rolling standard deviation.
Width parameter controls band distance from mean.
```

The pre-flight system parses this markdown, extracts the flow block, validates all operations against vanilla, runs the full pre-flight pipeline, and produces the report — all before any compilation.

**This same markdown can be:**
- Written by hand in any text editor
- Generated by an LLM from natural language
- Produced by a GUI drag-and-drop editor
- Stored in a git repository for version control
- Rendered as documentation on the registry website

One format, many interfaces, same safety guarantees.

### 10.8 GUI Module Builder (Future)

The GUI module builder is a visual interface for composing modules. It renders vanilla operations as blocks that can be connected with wires (pipes). The GUI:

- Shows available vanilla operations as a palette
- Shows installed module operations as an extended palette
- Allows drag-and-drop connection of blocks
- **Runs pre-flight validation in real-time** as the user builds
- Shows memory estimates updating live as the pipeline grows
- Highlights type mismatches immediately on connection
- Generates the markdown module definition as output
- Can import existing markdown modules for visual editing

The GUI is a frontend to the markdown format — not a separate system. Every GUI action maps to a markdown edit. The user can switch between GUI and markdown at any time.

---

## 14. LLM-Driven Module Composition & Ecosystem Flywheel

### 11.1 The Core Insight

Because Level 1 composition modules are safe by construction (they only compose vanilla-safe operations), and because the pre-flight system validates automatically (no human review needed), and because the structured intent format is simple enough for small LLMs to generate reliably — **the barrier to module contribution drops to near zero.**

A user describes what they want in natural language. An LLM composes it from vanilla operations. The failsafe validates it. It's published. Done.

The user just contributed a module without writing a single line of code.

### 11.2 The Flywheel

```
┌──────────────────────────────────────────────────────────┐
│                                                            │
│   Vanilla (safe primitives, proven, canonical)             │
│       ↓                                                    │
│   LLM composes vanilla blocks into module                  │
│   (natural language → structured intent → flow code)       │
│       ↓                                                    │
│   Failsafe pre-flight validates automatically              │
│   (no human review needed for Level 1)                     │
│       ↓                                                    │
│   Module published to registry                             │
│   (auto-benchmarked, auto-tested, auto-documented)         │
│       ↓                                                    │
│   Registry grows → operations catalog expands              │
│       ↓                                                    │
│   Pre-prompt for LLMs updates with new operations          │
│       ↓                                                    │
│   LLMs can now compose using community modules too         │
│       ↓                                                    │
│   More sophisticated compositions become possible          │
│       ↓                                                    │
│   More users contribute → more modules → richer ecosystem  │
│       ↓                                                    │
│   (cycle repeats, accelerating)                            │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

### 11.3 LLM Composition Flow

**Step 1: User describes intent**
```
User: "I want a module that detects mean reversion setups.
       When price is more than 2 standard deviations from 
       the 50-period moving average, generate a signal in 
       the opposite direction."
```

**Step 2: LLM generates structured intent**

The LLM, guided by the pre-prompt containing vanilla operations and module registry, generates:

```yaml
module:
  name: "mean_reversion_signal"
  implements: "vanilla.signal_generator"
  
  inputs:
    prices: "stream<float[N]>"
  
  pipeline:
    - stage: "rolling"
      args: { window: 50 }
      input: "prices"
      output: "windowed"
    
    - stage: "mean"
      input: "windowed"
      output: "sma"
    
    - stage: "std"
      input: "windowed"
      output: "volatility"
    
    - stage: "subtract"
      args: { a: "prices", b: "sma" }
      output: "deviation"
    
    - stage: "divide"
      args: { a: "deviation", b: "volatility" }
      output: "z_score"
    
    - stage: "branch"
      condition: "greater(z_score, 2.0)"
      true_value: -1
      false_condition: "less(z_score, -2.0)"
      false_true_value: 1
      default_value: 0
      output: "signal"
  
  outputs:
    signal: "stream<int[N]>"
```

**Step 3: Validation gates check the intent**

```
Gate 1 (Schema):    ✓ Valid YAML, all required fields present
Gate 2 (Types):     ✓ float[N] → rolling → float[N,50] → mean → float[N] → ... → int[N]
Gate 3 (Dependencies): ✓ Acyclic, no temporal (all operations are windowed)
Gate 4 (Operations):   ✓ All operations exist in vanilla
```

**Step 4: Transpiler generates flow code**

```flow
module mean_reversion_signal

import vanilla.{rolling, mean, std, subtract, divide, branch}

export stage detect(prices: stream<float[N]>) -> stream<int[N]>:
    windowed   = prices |> rolling(50)
    sma        = windowed |> mean()
    volatility = windowed |> std()
    deviation  = prices |> subtract(sma)
    z_score    = deviation |> divide(volatility)
    signal     = z_score |> branch(
        greater(2.0)  → -1,
        less(-2.0)    → 1,
        otherwise     → 0
    )
    return signal
```

**Step 5: Pre-flight validates the generated code**

```
✓ Operations     All found in vanilla
✓ Type chain     Valid: float[N] → ... → int[N]
✓ Memory         Est. 24MB for N=50,000
✓ Numeric        Division by volatility could produce NaN 
                  if volatility=0 (auto-guarded by vanilla.divide)
✓ Bounds         Windowed, bounded
✓ Dependencies   Acyclic
✓ Backpressure   Balanced

STATUS: READY TO COMPILE
```

**Step 6: Published to registry**

The module is live. Auto-benchmarked. Auto-documented (the LLM's original natural language description becomes the module documentation). Searchable. Installable.

### 11.4 Community Scaling Dynamics

**Why this scales exponentially:**

1. **Zero-barrier contribution**: Non-programmers contribute through natural language. A forex trader who has never coded can describe their proprietary indicator and it becomes a module.

2. **Safety by construction**: Level 1 modules composed from vanilla don't need human code review. The automated pipeline is the reviewer. This removes the bottleneck that kills most open-source ecosystems — maintainer review bandwidth.

3. **Self-improving ecosystem**: Every new module becomes available in the LLM's operations catalog. The LLM can compose new modules from existing community modules, not just vanilla. Complexity grows without cognitive burden.

4. **Competition drives quality**: When 5 people publish EMA modules, benchmarks are public. The best implementation naturally surfaces. Users benefit without evaluating alternatives — the compiler's auto-selection picks the fastest conformant module.

5. **Domain explosion**: Financial computing modules attract forex traders. Image processing modules attract photographers. Scientific computing modules attract researchers. Each domain community builds on shared vanilla primitives but creates domain-specific vocabularies.

6. **LLM-agnostic means maximum reach**: Someone using Claude, someone using Qwen, someone using a local Llama — all contribute to and benefit from the same ecosystem. No vendor lock-in means maximum community size.

### 11.5 Quality Control at Scale

As the ecosystem grows, quality control becomes critical:

**Automated (always active):**
- Vanilla conformance tests (correctness)
- Automated benchmarks on reference hardware (performance)
- Pre-flight validation (safety)
- Type signature verification (compatibility)

**Community-driven:**
- Download counts (popularity signal)
- User ratings (quality signal)
- Bug reports (reliability signal)
- Community code review (for Level 2 and 3 modules)

**Registry curation:**
- "Verified" badge for modules that pass extended test suites
- "Featured" status for top-performing modules per interface
- Deprecation warnings for modules that fall behind or become unmaintained
- Automatic compatibility testing against new vanilla versions

### 11.6 The Module as Documentation

Because modules can be defined in markdown, and because LLMs generate natural language descriptions alongside the structured intent, **every module is self-documenting by default**.

The registry listing for a module includes:
- The original natural language description (from the user's intent)
- The generated flow code (readable pipeline)
- The pre-flight report (safety verification)
- Benchmark results (performance data)
- Usage examples (generated from type signatures)
- Vanilla conformance status

This means the ecosystem has rich, searchable documentation from day one — not as an afterthought, but as a structural consequence of how modules are created.

---

## 15. Future: Reactive Extensions & Application Platform

The current language (v1) is a general-purpose programming language with GPU acceleration. It handles computation, I/O, networking, databases, and application building through its module ecosystem. However, one paradigm is intentionally deferred: **reactive/interactive programming**.

### 15.1 What v1 Cannot Do Well

v1 pipelines are fundamentally **input → process → output**. They handle:
- Batch processing (load file, compute, save results)
- Streaming (continuous data flow from tap to emit)
- Request/response (web server handles request, computes, returns)

What they don't handle natively:
- **Bidirectional state**: user changes a slider → chart updates → user clicks a data point → detail panel opens
- **Persistent interactive state**: application is "running" indefinitely, responding to events
- **Real-time rendering loops**: game engine updating 60 frames per second based on accumulated state

The desktop GUI and mobile examples in Section 8 work through the `ext.ui.*` modules using callback patterns (like `on_click`), but these are module-level abstractions, not language primitives.

### 15.2 Reactive Extensions (v2 Roadmap)

The dataflow model naturally extends to reactive programming. The key additions:

```flow
// New core primitives for v2

signal mouse_pos: (int, int)         // a value that changes over time (reactive stream)
signal click: Event                  // event signal (discrete occurrences)

state counter: int = 0               // persistent mutable state (survives across events)
state theme: Theme = Theme.dark      // application-level state

// Reactions: pipelines that re-execute when inputs change
reaction on click:
    counter = counter + 1

reaction when counter > 100:
    emit("Achievement unlocked!", "notifications")

// Reactive pipelines: output updates automatically when input changes
reactive stream filtered_data = data
    |> filter(stage(x): x.price > price_threshold.value)
    |> sort("volume", descending=true)

// Views: declarative UI descriptions bound to reactive state
view dashboard:
    chart(filtered_data, type="scatter")   // re-renders when filtered_data changes
    slider(price_threshold, min=0, max=100) // modifying slider triggers re-filter
    text("Showing {filtered_data |> length()} items")
```

**Why this works as a natural extension:**
- `signal` is a stream with exactly one current value (special case of `stream`)
- `state` is an accumulator that persists across pipeline executions
- `reaction` is a pipeline that re-triggers when its input signals change
- `reactive stream` is a memoized pipeline that only recomputes when dependencies change
- `view` is an emit target that renders to screen instead of file

The compiler already knows how to:
- Track dependencies (dataflow graph analysis)
- Determine what's parallel (static analysis)
- Route GPU vs CPU (cost model)

Reactive extensions add one new thing: **event-driven re-execution** of subgraphs. When a signal changes, the compiler knows which downstream stages are affected and re-executes only those — on GPU where profitable, on CPU where necessary.

### 15.3 Architecture Decisions to Preserve Now

To ensure v2 reactive extensions are possible, v1 must:

1. **Keep `tap`/`emit` generic** — they must support interactive sources (keyboard, mouse, touch, microphone) not just file/network
2. **Support persistent pipelines** — pipelines that stay alive, not just batch run-and-exit
3. **Allow mutable state in controlled contexts** — `var` within stages is already designed; `state` at application level is the next step
4. **Not assume batch execution** — the scheduler must support event-driven re-execution, not just sequential walk-through

None of these require building the reactive system now. They're constraints on v1's design that keep the door open.

---

## 16. Open Questions

The following design decisions require further exploration:

### 16.1 Language Syntax

- **Final syntax for temporal marking**: `temporal` keyword prefix? Special operator? Annotation?
- **Lambda/anonymous stages**: Can you write inline transformations, or must all stages be named?
- **Pattern matching**: Should the language support full pattern matching on stream elements?
- **For loops**: How do imperative `for` loops interact with the dataflow model? Are they sugar for `map`?
- **String interpolation syntax**: `{expr}` vs `${expr}` vs `f"..."` syntax?

### 16.2 Type System

- **Generic stages**: Can stages be polymorphic over types? (`stage double<T: Numeric>(x: T) -> T`)
- **Dependent types**: Should array dimensions be part of the type? (`float[1000]` vs `float[N]` vs `float[]`)
- **Effect system depth**: How granularly should side effects be tracked? Just pure/impure, or a full effect system?
- **Null safety**: Is `null` a valid value? Or do we use `Option<T>` exclusively?
- **Union types**: Can a value be `string | int`? Or must all types be concrete?

### 16.3 General-Purpose Features

- **Closures**: Can stages capture variables from their enclosing scope?
- **Iterators**: Should there be a lazy iteration protocol alongside streams?
- **Concurrency model**: Beyond dataflow parallelism, how do you express concurrent tasks (e.g., fetch from 3 APIs simultaneously)?
- **Package manager**: `flowgpu add` — how does dependency resolution work? Lock files? Semver ranges?
- **Build system**: Do OctoFlow projects need a build configuration file? What format?

### 16.4 Module System

- **Interface evolution**: How do vanilla interfaces change across language versions without breaking existing modules?
- **Approximate modules**: Some modules trade accuracy for speed (e.g., approximate median). How is accuracy tolerance declared and verified?
- **Cross-module composition**: Can modules depend on other modules? If so, how deep can the dependency tree go?
- **Private/internal modules**: Can organizations have private registries for proprietary modules?

### 16.5 Compiler Integration

- **Module-specific compiler hints**: Can a module provide hints to the compiler beyond the parallel profile? (e.g., "this implementation works best with tile size 64")
- **JIT compilation**: Should modules support just-in-time compilation for data-size-dependent optimization?
- **Profiling feedback**: Should the runtime feed actual performance data back to influence future module selection?
- **Incremental compilation**: Can individual stages be recompiled without rebuilding the full graph?

### 16.6 Ecosystem

- **Governance**: Who maintains vanilla and standard modules? How are new interfaces proposed and accepted?
- **Namespacing**: How are module name conflicts resolved?
- **Deprecation policy**: How are modules or vanilla interfaces retired?
- **Standard module ownership**: Are standard modules maintained by core team or community?
- **Security**: How are malicious modules detected? Code signing? Sandboxing?

### 16.7 Failsafe System

- **Memory estimation accuracy**: How precise can static memory estimates be for dynamic/data-dependent pipelines?
- **Numeric tolerance**: What's the default NaN/overflow policy? Error, warn, substitute, or configurable per-module?
- **Pre-flight for native modules**: How deeply can the pre-flight system analyze Level 3 native code vs treating it as a black box?
- **Runtime recovery**: When a runtime guard triggers (e.g., approaching OOM), should the system auto-degrade (fall back to CPU) or stop and report?

### 16.8 Platform Support

- **macOS**: MoltenVK translation layer or native Metal SPIR-V path?
- **Windows**: Vulkan or also Direct3D 12 via SPIR-V (when Shader Model 7 ships)?
- **Browser**: WebGPU target for running OctoFlow in browsers?
- **Mobile GPU**: Vulkan Compute on Android, Metal on iOS — unified or separate paths?
- **Cloud GPU**: Server-side execution without display — headless Vulkan?

---

*This annex is a living document. Open questions will be resolved as design work continues in subsequent annexes.*
