# OctoFlow — Annex B: Programming Model & Language Semantics

**Parent Document:** OctoFlow Blueprint & Architecture  
**Supersedes:** Annex A, Section 3 (General-Purpose Language Additions)  
**Status:** Draft  
**Version:** 0.1  
**Date:** February 15, 2026  

---

## Table of Contents

1. Design Philosophy: Safety Over Expressiveness
2. The Complete Concept Map
3. Data: Records & Enums
4. Values: let & var
5. Computation: stage & fn
6. Flow: Pipes, Match, If, For
7. Structure: Modules & Visibility
8. Safety: Result, Option, and the ? Operator
9. Type System (Finalized)
10. Collections
11. Strings & Interpolation
12. I/O Model: tap, emit, print
13. Error Propagation Through Pipelines
14. Output & Compilation Model
15. Project Structure
16. REPL & Script Mode
17. What OctoFlow Does NOT Have
18. Complete Examples
19. LLM & GUI Builder Implications

---

## 1. Design Philosophy: Safety Over Expressiveness

### 1.1 The Core Constraint

OctoFlow code is primarily written by:
- **LLMs** translating natural language to code
- **GUI block builders** connecting visual blocks with wires
- **Humans** doing light editing and review

It is NOT primarily written by:
- Expert programmers who want maximum control
- People who enjoy clever abstractions
- Teams that need deep OOP hierarchies

This changes everything about language design. The language should optimize for:

| Priority | Meaning |
|----------|---------|
| **Unambiguity** | One way to do everything. LLMs never choose between equivalent patterns. |
| **Safety** | Every error caught at compile time or pre-flight. Runtime crashes are architecture failures. |
| **Inferability** | The compiler infers what it can. The author specifies the minimum. |
| **Readability** | Anyone (human or LLM) reading the code understands it immediately. |
| **Composability** | Small pieces connect into large programs through pipes and modules. |

The language deliberately sacrifices:

| Sacrificed | Why |
|-----------|-----|
| OOP (classes, methods, inheritance) | Two ways to call behavior (method vs function) = ambiguity |
| Generics / type parameters | LLMs frequently get generic signatures wrong |
| Operator overloading | `a + b` meaning different things for different types = hidden behavior |
| Implicit conversions | Silent type coercion = silent bugs |
| Exceptions / throw | Non-local control flow = hard to reason about |
| Null | Billion dollar mistake. Option type instead. |
| Global mutable state | Breaks parallelism analysis. Breaks everything. |
| Multiple inheritance / mixins | Complexity explosion with no safety benefit |

### 1.2 The One-Way Principle

For every task, OctoFlow has exactly ONE way to do it:

| Task | The One Way | NOT This |
|------|------------|----------|
| Define data | `record` | ❌ class, struct, dataclass, namedtuple |
| Define alternatives | `enum` | ❌ union, variant, abstract class hierarchy |
| Transform streams | `stage` | ❌ method on stream, operator overload |
| Regular computation | `fn` | ❌ method, lambda, static method, classmethod |
| Handle errors | `Result<T,E>` + `?` | ❌ try/catch, exceptions, error codes |
| Handle absence | `Option<T>` | ❌ null, nil, None, undefined |
| Organize code | `module` + `pub` | ❌ class, namespace, package, crate |
| Connect stages | `\|>` pipe | ❌ method chaining, nested calls |

When an LLM generates OctoFlow code, it never has to choose between patterns. There's only one pattern.

---

## 2. The Complete Concept Map

```
┌─────────────────────────────────────────────────────────┐
│                 FLOWGPU LANGUAGE CONCEPTS                 │
│                                                           │
│  DATA DEFINITION                                          │
│    record ──── Named fields, no behavior                  │
│    enum ────── Tagged alternatives, exhaustive match      │
│    type ────── Alias for existing type                    │
│                                                           │
│  VALUES                                                   │
│    let ─────── Immutable binding (default, safe)          │
│    var ─────── Mutable binding (stage-local only)         │
│                                                           │
│  COMPUTATION                                              │
│    stage ───── Stream transformer (GPU/CPU unit)          │
│    fn ──────── Regular function (not stream-aware)        │
│                                                           │
│  FLOW                                                     │
│    |> ──────── Pipe (connect stages, left-to-right)       │
│    match ───── Pattern match (exhaustive, enforced)       │
│    if/else ─── Conditional (expression, returns value)    │
│    for ─────── Iteration (sugar for map on collections)   │
│                                                           │
│  STRUCTURE                                                │
│    module ──── Namespace + encapsulation                   │
│    pub ─────── Public visibility marker                   │
│    import ──── Bring modules into scope                   │
│    from ────── Import specific items                      │
│                                                           │
│  SAFETY                                                   │
│    Result<T,E> Operation that can fail                    │
│    Option<T> ─ Value that may not exist                   │
│    ? ────────── Propagate error to caller                 │
│    match ────── Exhaustive handling enforced               │
│                                                           │
│  I/O                                                      │
│    tap ─────── Data enters the system                     │
│    emit ────── Data exits the system                      │
│    print ───── Shorthand for emit to stdout               │
│                                                           │
│  ANNOTATIONS (escape hatches, rarely used)                │
│    @device(gpu|cpu)   Force device assignment             │
│    @parallel(dim)     Force parallelism on dimension      │
│    @sequential        Force sequential execution          │
│    @unsafe            Disable runtime safety checks       │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

Total concepts: **23** (including annotations). Compare: Python has 35 keywords plus classes, decorators, metaclasses, descriptors, generators, context managers, comprehensions, async/await. Rust has 39 keywords plus traits, lifetimes, macros, closures, iterators. OctoFlow is intentionally smaller.

---

## 3. Data: Records & Enums

### 3.1 Records

A record is a named collection of typed fields. Records are DATA ONLY — no methods, no behavior, no hidden state.

```flow
record User:
    id: int
    name: string
    email: string

record Trade:
    symbol: string
    price: float
    quantity: int
    side: Direction
    timestamp: datetime

record Config:
    host: string
    port: int
    debug: bool
    max_connections: int
```

**Creating records:**
```flow
let user = User{id: 1, name: "Mike", email: "mike@fx.com"}
let trade = Trade{
    symbol: "XAUUSD",
    price: 2045.50,
    quantity: 100,
    side: Direction.Long,
    timestamp: datetime.now()
}
```

**Accessing fields:**
```flow
let name = user.name
let total = trade.price * trade.quantity
```

**Updating records (immutable — creates a new record):**
```flow
let updated_user = user with {email: "new@fx.com"}
let filled_trade = trade with {price: 2046.00, timestamp: datetime.now()}
```

The `with` keyword creates a copy with specified fields changed. The original is not modified. This is safe — no aliasing bugs, no unexpected mutation.

**Destructuring:**
```flow
let User{name, email, ..} = user
print("Name: {name}, Email: {email}")

// In function parameters
fn format_trade(Trade{symbol, price, quantity, ..}: Trade) -> string:
    return "{symbol}: {quantity} @ {price}"
```

**Records are GPU-friendly.** A `stream<Trade>` is a struct-of-arrays in GPU memory — the compiler lays out all `price` values contiguously, all `quantity` values contiguously, etc. This is optimal for GPU memory coalescing. The programmer doesn't think about this.

### 3.2 Enums

An enum defines a type with a fixed set of alternatives. Each alternative can carry data.

```flow
// Simple enum (no data)
enum Direction:
    Long
    Short
    Flat

// Enum with data
enum OrderResult:
    Filled(price: float, time: datetime)
    PartialFill(filled_qty: int, remaining: int)
    Rejected(reason: string)
    Cancelled

// The built-in Result and Option are enums
enum Result<T, E>:
    Ok(T)
    Err(E)

enum Option<T>:
    Some(T)
    None
```

**Using enums with match (EXHAUSTIVE — compiler enforces all cases handled):**

```flow
let result = submit_order(order)

match result:
    Filled(price, time) ->
        print("Filled at {price} on {time}")
    PartialFill(filled, remaining) ->
        print("Partial: {filled} filled, {remaining} remaining")
    Rejected(reason) ->
        print("Rejected: {reason}")
    Cancelled ->
        print("Order was cancelled")
```

If you forget a case, the compiler refuses to compile:

```
ERROR: Non-exhaustive match on OrderResult
  Missing cases: PartialFill, Cancelled
  at src/trading.flow:42
```

This is a **hard safety guarantee**. No runtime "unhandled case" crashes.

### 3.3 Type Aliases

```flow
type Price = float
type Symbol = string
type TimeSeries = stream<float[N]>
type PriceMap = map<Symbol, Price>
type Validator = fn(User) -> Result<User, string>
```

Type aliases are documentation — they make code readable without adding complexity. The compiler treats `Price` and `float` identically.

---

## 4. Values: let & var

### 4.1 let — Immutable Binding (Default)

```flow
let threshold = 0.05
let name = "OctoFlow"
let config = load_config("settings.toml")?
let prices = tap("market_feed")
```

`let` bindings are **immutable**. Once assigned, they cannot be changed. This is the default because immutability is safe:
- No accidental mutation
- No aliasing bugs
- No thread-safety issues
- The compiler can freely reorder, parallelize, and optimize

### 4.2 var — Mutable Binding (Restricted)

```flow
stage compute_running_mean(data: stream<float>) -> stream<float>:
    var sum = 0.0
    var count = 0
    for value in data:
        sum = sum + value
        count = count + 1
    return sum / count
```

`var` creates a mutable binding with **strict restrictions**:

| Rule | Why |
|------|-----|
| `var` is scoped to the current stage or fn | Mutation never leaks across boundaries |
| `var` cannot be passed to other stages | Prevents aliased mutation |
| `var` cannot be captured in closures | Prevents escape |
| `var` cannot be global | Prevents shared mutable state |

If you try to misuse `var`:

```flow
let shared_counter = var 0   // ERROR: var cannot be at module level

stage bad_example(data: stream<float>) -> stream<float>:
    var local = 0
    return data |> map(stage(x):
        local = local + 1    // ERROR: cannot capture var from outer scope
        return x
    )
```

The compiler catches every unsafe use. `var` exists for local accumulation patterns within a single stage — nowhere else.

---

## 5. Computation: stage & fn

### 5.1 stage — Stream Transformer

A stage is the fundamental computation unit. It transforms streams of data and is the primary target for GPU/CPU classification.

```flow
stage normalize(data: stream<float[N]>) -> stream<float[N]>:
    let mn = data |> min()
    let mx = data |> max()
    let range = mx - mn
    return data |> subtract(mn) |> divide(range)

stage detect_anomaly(values: stream<float[N]>, threshold: float) -> stream<bool[N]>:
    let mu = values |> mean()
    let sigma = values |> std()
    return values |> subtract(mu) |> abs() |> greater(sigma * threshold)
```

**Stages are pure by default.** They take inputs, produce outputs, and have no side effects. The compiler relies on purity for GPU parallelization. If a stage needs side effects (I/O, logging), it must be marked:

```flow
@impure
stage save_results(data: stream<float[N]>, path: string) -> stream<float[N]>:
    emit(data, path)
    return data   // pass-through
```

**Stages compose with pipes:**

```flow
stream result = raw_data
    |> normalize()
    |> detect_anomaly(threshold=2.0)
    |> filter(stage(is_anomaly): is_anomaly)
```

### 5.2 fn — Regular Function

A regular function is for non-streaming computation — utility code, data construction, formatting, validation.

```flow
fn calculate_risk(price: float, volatility: float, position: float) -> float:
    return price * volatility * position * 0.01

fn validate_user(user: User) -> Result<User, string>:
    if user.name |> length() == 0:
        return err("Name cannot be empty")
    if user.email |> contains("@") == false:
        return err("Invalid email")
    return ok(user)

fn format_trade(trade: Trade) -> string:
    return "{trade.symbol}: {trade.quantity} @ {trade.price:.2f}"

fn build_config(args: Args) -> Config:
    return Config{
        host: args |> get_option("host") |> unwrap_or("localhost"),
        port: args |> get_option("port") |> unwrap_or(8080),
        debug: args |> has_flag("debug"),
        max_connections: 100
    }
```

**Key difference from stage:**
- `stage` operates on streams — it's the GPU/CPU classification unit
- `fn` operates on single values — it's a regular function
- `fn` can be called inside a `stage` (e.g., for per-element computation)
- `fn` can also be called at the top level (program setup, configuration)

**Both are free functions.** Neither is attached to a record. There is no `user.validate()` — only `validate_user(user)`. This is the One-Way Principle.

### 5.3 How the Compiler Distinguishes stage vs fn

The programmer chooses based on intent:
- "I'm transforming a data pipeline" → `stage`
- "I'm doing a regular computation" → `fn`

The compiler uses this distinction for GPU/CPU routing:
- `stage` → analyzed for parallelism, may run on GPU
- `fn` called inside a `stage` → gets inlined into the stage's GPU/CPU code
- `fn` called at top level → always CPU (setup code)

If a programmer accidentally writes a `fn` that should be a `stage`, the code still works — it just won't get GPU-accelerated. The pre-flight report can suggest: "Consider changing `fn process` to `stage process` — it operates on arrays and could benefit from GPU execution."

---

## 6. Flow: Pipes, Match, If, For

### 6.1 Pipe Operator |>

The pipe is the primary composition mechanism:

```flow
let result = data
    |> clean()
    |> normalize()
    |> analyze()
    |> format()
```

**Pipe rules:**
- Left side's output type must match right side's input type
- Compiler enforces at compile time
- Pipes create edges in the dataflow graph
- The compiler decides parallel/sequential/GPU/CPU from the graph

**Multi-input stages use named arguments:**

```flow
let result = prices |> crossover(fast=ema_12, slow=ema_26)
let merged = data_a |> merge(data_b, on="symbol")
```

### 6.2 match — Pattern Matching (Exhaustive)

```flow
// Match on enum
match order_result:
    Filled(price, time) -> log.info("Filled at {price}")
    Rejected(reason) -> log.error("Rejected: {reason}")
    Cancelled -> log.warn("Cancelled")

// Match on value
match status_code:
    200 -> "OK"
    404 -> "Not Found"
    500 -> "Server Error"
    _   -> "Unknown: {status_code}"

// Match with guards
match temperature:
    t if t > 100 -> "Boiling"
    t if t > 0   -> "Liquid"
    t if t == 0  -> "Freezing"
    _            -> "Frozen"

// Match as expression (returns a value)
let label = match direction:
    Direction.Long  -> "BUY"
    Direction.Short -> "SELL"
    Direction.Flat  -> "HOLD"
```

**Exhaustiveness is enforced.** Every `match` on an enum must handle every variant. This is checked at compile time. No runtime surprises.

### 6.3 if/else — Conditional Expression

```flow
// if/else always returns a value (it's an expression)
let status = if balance > 0 then "positive" else "zero or negative"

// Multi-line
let category = if score > 90:
    "excellent"
else if score > 70:
    "good"
else if score > 50:
    "average"
else:
    "needs improvement"

// In pipelines (branch pattern)
stream signals = z_score
    |> if_then(greater(2.0), -1)
    |> if_then(less(-2.0), 1)
    |> otherwise(0)
```

### 6.4 for — Iteration

```flow
// Iterate over collections
for file in list_dir("data/")?:
    let data = read_file(file.path)?
    print("Loaded {file.name}: {data |> length()} bytes")

// Iterate with index
for (i, item) in items |> enumerate():
    print("{i}: {item}")

// For is sugar for map when used in a pipeline context
let doubled = for x in numbers: x * 2
// equivalent to: numbers |> map(stage(x): x * 2)
```

**for is always bounded.** There is no `while true` or unbounded loop. If you need continuous processing, use a continuous stream with a tap. This prevents infinite loops — a safety guarantee.

```flow
// NOT allowed:
while condition:    // ERROR: while does not exist in OctoFlow
    do_something()

// Instead, use stream processing:
stream events = tap("event_source")    // continuous stream
stream processed = events |> handle()  // processes until source closes
```

---

## 7. Structure: Modules & Visibility

### 7.1 Modules

A module is a namespace and encapsulation boundary. Every `.flow` file is implicitly a module.

```flow
// file: src/trading.flow
module trading

// Public — visible to importers
pub record Order:
    symbol: string
    price: float
    quantity: int
    side: Direction

pub fn create_order(symbol: string, price: float, qty: int, side: Direction) -> Order:
    return Order{symbol: symbol, price: price, quantity: qty, side: side}

pub stage execute_orders(orders: stream<Order>) -> stream<OrderResult>:
    return orders |> validate_order() |> submit_order()

// Private — only this module uses
fn validate_order(order: Order) -> Result<Order, string>:
    if order.quantity <= 0:
        return err("Quantity must be positive")
    if order.price <= 0.0:
        return err("Price must be positive")
    return ok(order)

fn submit_order(order: Order) -> OrderResult:
    // internal implementation
    ...
```

### 7.2 Visibility Rules

| Marker | Visibility | Default? |
|--------|-----------|----------|
| (none) | Private — only within this module | ✅ Yes |
| `pub` | Public — visible to anyone who imports this module | Must be explicit |

Two states. That's it. No `protected`, no `internal`, no `friend`, no `pub(crate)`.

### 7.3 Imports

```flow
// Import entire module
import std.io
import std.json
import ext.db.sqlite as db

// Import specific items
from std.io import read_file, write_file
from trading import Order, create_order

// Usage
let data = std.io.read_file("data.txt")?     // fully qualified
let data = read_file("data.txt")?             // after specific import
let conn = db.open("app.db")?                 // aliased import
```

**Import rules:**
- Imports are explicit — no implicit global scope
- No circular imports (compiler error)
- Standard modules (`std.*`) are always available
- Extended modules (`ext.*`) require installation via `flow add`
- Aliasing with `as` prevents name collisions

### 7.4 Nested Modules

```flow
// file: src/analytics/signals.flow
module analytics.signals

pub stage golden_cross(prices: stream<float[N]>) -> stream<int[N]>:
    let fast = prices |> rolling(50) |> mean()
    let slow = prices |> rolling(200) |> mean()
    return fast |> subtract(slow) |> sign()
```

```flow
// usage
import analytics.signals
let signals = prices |> analytics.signals.golden_cross()

// or
from analytics.signals import golden_cross
let signals = prices |> golden_cross()
```

---

## 8. Safety: Result, Option, and the ? Operator

### 8.1 Result<T, E> — Handling Failure

Any operation that can fail returns `Result`:

```flow
fn read_config(path: string) -> Result<Config, Error>:
    let text = read_file(path)?                // ? propagates error
    let parsed = json.parse(text)?             // ? propagates error
    let config = Config{
        host: parsed["host"] |> as_string()?,
        port: parsed["port"] |> as_int()?,
        debug: parsed["debug"] |> as_bool() |> unwrap_or(false)
    }
    return ok(config)
```

**Handling Results:**
```flow
// Option 1: match (explicit, handles both cases)
match read_config("settings.json"):
    ok(config) -> start_app(config)
    err(e)     -> print("Config error: {e}")

// Option 2: ? operator (propagate error to caller)
let config = read_config("settings.json")?

// Option 3: unwrap_or (provide default)
let config = read_config("settings.json") |> unwrap_or(default_config())

// Option 4: unwrap (CRASH if error — only for prototyping)
let config = read_config("settings.json") |> unwrap()
// Pre-flight WARNING: unwrap() will crash on error. Consider ? or unwrap_or.
```

### 8.2 Option<T> — Handling Absence

OctoFlow has **no null**. Values that might not exist use `Option`:

```flow
fn find_user(email: string) -> Option<User>:
    let results = query("SELECT * FROM users WHERE email = ?", email)?
    if results |> length() == 0:
        return none
    return some(results |> first())

// Handling Options
match find_user("mike@fx.com"):
    some(user) -> print("Found: {user.name}")
    none       -> print("User not found")

// Or with unwrap_or
let user = find_user("mike@fx.com") |> unwrap_or(default_user())

// Or with ? in functions returning Option
fn get_user_email(id: int) -> Option<string>:
    let user = find_user_by_id(id)?     // returns none if not found
    return some(user.email)
```

### 8.3 Error Propagation in Pipelines

When a stage in a pipeline returns `Result`, the pipeline handles it:

```flow
stream results = raw_data
    |> parse_csv()?          // if parse fails, pipeline stops with error
    |> validate()?           // if validation fails, pipeline stops with error  
    |> transform()           // pure transformation, cannot fail
    |> save("output.csv")?   // if save fails, pipeline stops with error
```

The `?` at each stage means: "if this stage produces an error for any element, stop the pipeline and propagate the error." This is different from element-level error handling:

```flow
// Element-level: skip bad elements, continue pipeline
stream results = raw_data
    |> parse_csv()
    |> filter_ok()           // drops Err elements, keeps Ok elements
    |> transform()
    |> save("output.csv")?

// Element-level: replace bad elements with defaults
stream results = raw_data
    |> parse_csv()
    |> map(stage(r): r |> unwrap_or(default_record()))
    |> transform()
```

---

## 9. Type System (Finalized)

### 9.1 Primitive Types

```flow
int                  // platform-width integer (64-bit)
int8, int16, int32, int64
uint8, uint16, uint32, uint64
float                // defaults to float32 (GPU-native)
float16, float32, float64
bool                 // true or false
string               // UTF-8 text (CPU-native)
byte                 // raw byte (uint8 alias)
```

### 9.2 Compound Types

```flow
// Arrays (GPU-friendly, contiguous memory)
float[N]             // 1D array, N elements
float[M, N]          // 2D array, M × N elements
int[H, W, C]         // 3D array

// Collections (CPU-native)
list<T>              // dynamic-length list
map<K, V>            // hash map
set<T>               // unique set

// Tuple
(int, string)        // fixed-size heterogeneous pair
(float, float, float) // triple

// Streams
stream<T>            // streaming data of type T
stream<float[N]>     // stream of float arrays
stream<Trade>        // stream of records

// Functions (as values)
fn(int, int) -> int  // function type
```

### 9.3 Type Inference

The compiler infers types when possible. The programmer only annotates when necessary:

```flow
// Compiler infers types from context
let x = 42                    // inferred: int
let name = "OctoFlow"         // inferred: string
let prices = list[1.0, 2.0]  // inferred: list<float>

// Stage signatures should be explicit (they're the interface)
stage normalize(data: stream<float[N]>) -> stream<float[N]>:
    let mn = data |> min()    // inferred: float
    let range = max(data) - mn // inferred: float
    return data |> subtract(mn) |> divide(range)

// fn signatures should be explicit for pub functions
pub fn calculate_risk(price: float, vol: float) -> float:
    return price * vol * 0.01

// fn signatures can be inferred for private functions
fn helper(x, y):              // types inferred from usage
    return x + y
```

### 9.4 No Implicit Conversions

Type conversions must be explicit:

```flow
let x: int = 42
let y: float = x              // ERROR: cannot implicitly convert int to float
let y: float = x |> to_float() // OK: explicit conversion

let a: float32 = 3.14
let b: float64 = a            // ERROR: cannot implicitly widen
let b: float64 = a |> to_f64() // OK: explicit
```

This prevents silent precision loss — a critical safety guarantee for numerical computing.

### 9.5 The Compiler Infers Module Metadata

When a module author writes a stage, the compiler infers:

| Property | Inferred From |
|----------|--------------|
| Type signature | Stage declaration |
| Purity | Analysis of stage body (side effects?) |
| Temporal dependency | Presence of `temporal` keyword |
| Parallelizable dimensions | Which array dimensions are operated on independently |
| Arithmetic intensity | Operation count per element |
| Memory access pattern | Array access patterns in stage body |
| Memory estimate | Data types × operation memory formulas |
| Module level | Whether only vanilla ops are used (composition vs custom) |
| Interface conformance | Matching against vanilla interface signatures |

**The author provides ONLY:** module name, doc comment, and the code itself. Everything else is inferred. This is what makes LLM-generated modules possible — the LLM writes code, the compiler does the rest.

---

## 10. Collections

### 10.1 Arrays (GPU-Friendly)

Arrays are fixed-layout, contiguous memory — optimal for GPU:

```flow
let prices = float[5]{1.0, 2.0, 3.0, 4.0, 5.0}
let matrix = float[3, 3]{
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0}
}

// Array operations (GPU-accelerated via vanilla)
let doubled = prices |> multiply(2.0)
let normalized = prices |> normalize()
let total = prices |> sum()
```

### 10.2 Lists (CPU-Native, Dynamic)

```flow
let items = list[1, 2, 3]
let extended = items |> append(4)         // list[1, 2, 3, 4]
let first = items |> first()              // some(1)
let length = items |> length()            // 3
let filtered = items |> filter(fn(x): x > 1)  // list[2, 3]
let mapped = items |> map(fn(x): x * 2)  // list[2, 4, 6]
```

### 10.3 Maps (CPU-Native)

```flow
let config = map{
    "host": "localhost",
    "port": 8080
}
let host = config |> get("host")          // some("localhost")
let updated = config |> insert("debug", true)
let removed = config |> remove("port")
let keys = config |> keys()               // list["host", "port"]
let has = config |> contains_key("host")  // true
```

### 10.4 Sets (CPU-Native)

```flow
let symbols = set["AAPL", "GOOG", "MSFT"]
let has = symbols |> contains("AAPL")     // true
let added = symbols |> insert("TSLA")
let union = symbols |> union(other_symbols)
let inter = symbols |> intersection(other_symbols)
```

### 10.5 Converting Between Arrays and Collections

```flow
// List → Array (for GPU processing)
let prices_list = list[1.0, 2.0, 3.0]
let prices_array = prices_list |> to_array()    // float[3]
let stats = prices_array |> mean()              // GPU-accelerated

// Array → List (for CPU manipulation)
let result_array = data |> analyze()
let result_list = result_array |> to_list()     // CPU-native
```

The compiler auto-inserts these conversions at GPU↔CPU boundaries when needed. But explicit conversion makes intent clear.

---

## 11. Strings & Interpolation

Strings are always CPU-native. No GPU string operations exist (or are planned).

```flow
let name = "OctoFlow"
let version = 1

// Interpolation with {}
let msg = "Welcome to {name} v{version}"
let path = "{base_dir}/{filename}.csv"

// Expression interpolation
let report = "Mean: {data |> mean():.2f}, Std: {data |> std():.2f}"
let status = "Processed {count} records in {elapsed:.1f}s"

// Multi-line strings
let query = """
    SELECT symbol, price, volume
    FROM trades
    WHERE date > '{start_date}'
    ORDER BY volume DESC
    LIMIT 100
"""

// String operations (all CPU via std.string)
let upper = name |> to_upper()
let parts = "a,b,c" |> split(",")
let joined = parts |> join(", ")
let trimmed = "  hello  " |> trim()
let contains = name |> contains("GPU")
```

---

## 12. I/O Model: tap, emit, print

### 12.1 tap — Data Enters

```flow
// File sources
stream data = tap("data.csv", format=csv)
stream bytes = tap("image.png", format=binary)
stream lines = tap("log.txt", format=lines)

// Network sources
stream feed = tap("wss://market.data/stream", format=json)
stream requests = tap("http://0.0.0.0:8080", format=http)

// System sources
stream input = tap("stdin")
stream env = tap("env")

// The tap is abstract — what it connects to is determined by:
// 1. The URI/path (file, URL, special name)
// 2. The format hint (csv, json, binary, lines, etc.)
// 3. Registered tap adapters (installed modules can register new tap types)
```

### 12.2 emit — Data Exits

```flow
emit(result, "output.csv")           // write to file
emit(response, "stdout")             // write to console
emit(data, "http://api.com/submit")  // send to URL
emit(records, "postgres://db/table") // write to database (with adapter)
```

### 12.3 print — Convenience

```flow
print("Hello, world!")               // → emit("Hello, world!\n", "stdout")
print("Count: {n}")                  // → string interpolation + emit to stdout
```

`print` exists because `emit(format(...), "stdout")` is verbose for the most common operation.

---

## 13. Error Propagation Through Pipelines

This section clarifies how errors flow through dataflow pipelines — a critical safety design.

### 13.1 Stage-Level Errors

When a stage returns `Result`, the pipeline caller decides what happens:

```flow
// Pipeline STOPS on first error (? operator)
stream output = input |> parse()? |> transform()? |> save()?
// If parse fails on any element → entire pipeline returns Err

// Pipeline SKIPS errors (filter_ok)
stream output = input |> parse() |> filter_ok() |> transform()
// Bad elements are dropped, good elements continue

// Pipeline REPLACES errors (unwrap_or)
stream output = input |> parse() |> map(fn(r): r |> unwrap_or(default)) |> transform()
// Bad elements become defaults, pipeline continues
```

### 13.2 Pre-flight Error Analysis

The pre-flight system analyzes error paths at compile time:

```
╔══════════════════════════════════════════════════╗
║  PRE-FLIGHT: ERROR PATH ANALYSIS                 ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  Stage 1: parse()     → can produce Err          ║
║  Stage 2: transform() → pure, no errors          ║
║  Stage 3: save()      → can produce Err (I/O)    ║
║                                                  ║
║  Unhandled error paths:                          ║
║    ⚠ parse() errors not handled                  ║
║      Suggestion: add ? or filter_ok() after      ║
║    ⚠ save() errors not handled                   ║
║      Suggestion: add ? or wrap in match          ║
║                                                  ║
║  STATUS: WARNING — errors may crash at runtime   ║
╚══════════════════════════════════════════════════╝
```

---

## 14. Output & Compilation Model

### 14.1 File Types

| Extension | What | Contents |
|-----------|------|----------|
| `.flow` | Source file | Human/LLM readable OctoFlow code |
| `.fgb` | Compiled binary | SPIR-V + CPU native code + runtime + metadata |
| `flow.project` | Project config | Name, version, dependencies, build settings |
| `flow.lock` | Lock file | Exact dependency versions for reproducibility |

### 14.2 Execution Modes

```bash
# Script mode — compile in memory, execute immediately
$ flow run script.flow
# Best for: quick tasks, scripting, development

# Build mode — compile to binary
$ flow build src/main.flow -o myapp
$ ./myapp
# Best for: production, distribution, performance
# The binary is self-contained — no OctoFlow install needed to run it

# Check mode — pre-flight only, no execution
$ flow check src/main.flow
# Best for: CI/CD, validation, LLM-generated code review

# REPL mode — interactive
$ flow repl
>>> let x = list[1, 2, 3, 4, 5]
>>> x |> mean()
3.0
>>> x |> std()
1.4142135623730951
# Best for: exploration, learning, quick calculations

# Shebang mode — executable scripts
#!/usr/bin/env flow run
import std.io
print("Hello from OctoFlow!")
# Best for: system scripts replacing Python/Bash
```

### 14.3 What the .fgb Binary Contains

```
┌─────────────────────────────────────┐
│  .fgb Binary Structure               │
├─────────────────────────────────────┤
│  Header                              │
│    Magic number, version, target     │
│    Hardware requirements (GPU arch)  │
├─────────────────────────────────────┤
│  SPIR-V Modules                      │
│    Pre-compiled GPU compute shaders  │
│    One per GPU subgraph              │
├─────────────────────────────────────┤
│  CPU Native Code                     │
│    Pre-compiled CPU functions        │
│    Via Cranelift → x86-64 / ARM64   │
├─────────────────────────────────────┤
│  Decision Functions                  │
│    Runtime GPU/CPU routing logic     │
│    Based on data size + hardware     │
├─────────────────────────────────────┤
│  Pipeline Metadata                   │
│    Dataflow graph structure          │
│    Type information                  │
│    Memory estimates                  │
│    Pre-flight report (embedded)      │
├─────────────────────────────────────┤
│  Embedded Runtime                    │
│    Scheduler                         │
│    GPU memory pool manager           │
│    Vulkan dispatch interface         │
│    OOM prevention monitor            │
│    Transfer coordinator              │
│    Hardware detection                │
├─────────────────────────────────────┤
│  Module Cache                        │
│    Compiled standard modules used    │
│    Compiled ext modules used         │
└─────────────────────────────────────┘
```

The `.fgb` is a **self-contained executable**. Copy it to any machine with a compatible GPU driver and it runs. No OctoFlow installation needed. Like Go or Rust binaries.

### 14.4 Package Manager

```bash
# Initialize a project
$ flow init my-project
# Creates: flow.project, src/main.flow, tests/

# Add dependencies
$ flow add ext.web.server
$ flow add ext.db.sqlite
$ flow add ext.ui.charts

# Remove dependencies
$ flow remove ext.ui.charts

# Update dependencies
$ flow update

# Install dependencies from lock file
$ flow install

# Publish a module
$ flow publish
# → Runs pre-flight, compiles, tests, benchmarks, publishes to registry
```

---

## 15. Project Structure

### 15.1 Standard Layout

```
my_project/
├── flow.project               # project configuration
├── src/
│   ├── main.flow              # entry point
│   ├── models.flow            # record and enum definitions
│   ├── services.flow          # business logic stages and fns
│   └── utils.flow             # helper functions
├── modules/                   # local module definitions (not published)
│   └── custom_indicator.flow
├── tests/
│   ├── test_services.flow     # unit tests
│   └── test_integration.flow  # integration tests
├── flow.lock                  # dependency lock (auto-generated)
└── .flowgpu/                  # local cache (gitignored)
    ├── hardware_profile.json  # cached hardware detection
    └── compiled/              # compilation cache
```

### 15.2 Project Configuration

```yaml
# flow.project
name: trading-platform
version: 0.1.0
description: GPU-accelerated trading analytics platform
entry: src/main.flow
license: MIT

[build]
target: auto                   # auto-detect, or "linux-x86_64", "macos-arm64"
gpu: auto                      # auto, nvidia, amd, cpu-only
mode: strict                   # strict (dev), release (prod), unsafe (benchmark)

[dependencies]
ext.web.server: ^1.0
ext.db.sqlite: ^1.0

[dev-dependencies]
ext.test: ^1.0
ext.bench: ^1.0
```

### 15.3 Testing

```flow
// tests/test_services.flow
import ext.test

test "normalize produces values between 0 and 1":
    let data = float[5]{10.0, 20.0, 30.0, 40.0, 50.0}
    let result = data |> normalize()
    assert(result |> min() == 0.0)
    assert(result |> max() == 1.0)

test "validate_user rejects empty name":
    let user = User{id: 1, name: "", email: "test@test.com"}
    let result = validate_user(user)
    assert(result |> is_err())

test "golden_cross signal generation":
    let prices = load_test_data("fixtures/prices.csv")?
    let signals = prices |> golden_cross()
    assert(signals |> length() == prices |> length())
    assert(signals |> unique() |> sort() == list[-1, 0, 1])
```

```bash
$ flow test
Running 3 tests...
  ✓ normalize produces values between 0 and 1     (0.02s)
  ✓ validate_user rejects empty name               (0.01s)
  ✓ golden_cross signal generation                  (0.15s, GPU)

3 tests passed, 0 failed
```

---

## 16. REPL & Script Mode

### 16.1 REPL

```
$ flow repl
OctoFlow v0.1.0 | GPU: NVIDIA RTX 4090 | Mode: strict

>>> let data = float[5]{1.0, 2.0, 3.0, 4.0, 5.0}
>>> data |> mean()
3.0

>>> data |> std()
1.4142135623730951

>>> record Point: x: float, y: float
>>> let p = Point{x: 3.0, y: 4.0}
>>> fn distance(p: Point) -> float: return (p.x * p.x + p.y * p.y) |> sqrt()
>>> distance(p)
5.0

>>> let big = float[1_000_000]{...random...}
>>> big |> mean()    // automatically on GPU
2500127.5            // (0.001s, GPU)

>>> :quit
```

### 16.2 Script Mode

```flow
#!/usr/bin/env flow run

// This is a OctoFlow script — runs directly like Python
import std.io
import std.os
import std.fmt

let args = os.args()
let dir = args |> get(1) |> unwrap_or(".")

let files = list_dir(dir)?
let total_size = files |> map(fn(f): f.size) |> sum()

print("Directory: {dir}")
print("Files: {files |> length()}")
print("Total size: {fmt.bytes(total_size)}")
```

```bash
$ chmod +x myscript.flow
$ ./myscript.flow /data
Directory: /data
Files: 42
Total size: 1.3 GB
```

---

## 17. What OctoFlow Does NOT Have

This section explicitly documents what is excluded from the language and why. This is as important as what IS in the language.

| Excluded Feature | Why |
|-----------------|-----|
| **Classes** | Data (records) and behavior (functions) are separate. One way to organize code. |
| **Methods on records** | `validate_user(user)` not `user.validate()`. One calling convention. |
| **Inheritance** | Composition via modules instead. No class hierarchies. |
| **Traits / Interfaces** | Compiler infers compatibility from type signatures. No explicit trait declarations. |
| **Generics / Type parameters** | Compiler infers types automatically. No `<T>` syntax. |
| **Operator overloading** | `+` always means numeric addition. No hidden behavior. |
| **Exceptions / throw / try-catch** | `Result<T,E>` with `?` operator instead. Errors are values, not control flow. |
| **Null / None / nil** | `Option<T>` instead. Absence is explicit, never a crash. |
| **Global mutable state** | All mutation is stage-local (`var`). No shared mutable state. |
| **Implicit type conversions** | `int` to `float` requires explicit `to_float()`. No silent precision loss. |
| **while loops** | Streams handle continuous processing. `for` handles bounded iteration. No infinite loops. |
| **Decorators / Annotations (beyond @device, etc.)** | Minimal escape hatches only. No annotation-driven programming. |
| **Macros** | Code generation is done by LLMs at the frontend layer, not by the language. |
| **async / await** | The dataflow graph IS the concurrency model. Async is implicit. |
| **Multiple return values** | Return a record or tuple instead. |
| **Variadic arguments** | Fixed-arity functions only. More predictable for LLMs. |
| **Default arguments** | Use function overloads or record-based config instead. Explicit is safer. |

**The guiding principle:** every excluded feature is one fewer decision an LLM has to make, one fewer pattern a GUI builder has to support, and one fewer source of bugs.

---

## 18. Complete Examples

### 18.1 Hello World

```flow
print("Hello, world!")
```

### 18.2 Script: Process CSV

```flow
#!/usr/bin/env flow run
import std.io
import std.csv

let data = csv.read("sales.csv", headers=true)?
let prices = data["price"] |> to_array()

print("Records: {prices |> length()}")
print("Mean: {prices |> mean():.2f}")
print("Max: {prices |> max():.2f}")
print("Std: {prices |> std():.2f}")
```

### 18.3 Data Pipeline (GPU-Accelerated)

```flow
import std.csv

stream trades = tap("trades.csv", format=csv)

stream analysis = trades
    |> cast_columns(map{"price": float, "volume": int})
    |> group_by("symbol")
    |> agg(
        avg_price = mean("price"),
        total_vol = sum("volume"),
        volatility = std("price"),
        trade_count = count()
    )
    |> sort("volatility", descending=true)

emit(analysis, "report.csv")
print("Analysis complete: {analysis |> length()} symbols processed")
```

### 18.4 Web API Server

```flow
import std.json
import std.log
import ext.web.server as web
import ext.db.sqlite as db

let database = db.open("app.db")?

let app = web.app()

app |> web.get("/health", fn(req):
    return web.response(200, json.stringify(map{"status": "ok"}))
)

app |> web.get("/users", fn(req):
    let users = db.query(database, "SELECT * FROM users")?
    return web.response(200, json.stringify(users))
)

app |> web.post("/users", fn(req):
    let body = req.body |> json.parse()?
    let result = db.exec(database,
        "INSERT INTO users (name, email) VALUES (?, ?)",
        params=list[body["name"], body["email"]]
    )?
    return web.response(201,
        json.stringify(map{"id": result.last_id})
    )
)

log.info("Starting server on :8080")
web.listen(app, port=8080)?
```

### 18.5 Module Definition (What LLMs Generate)

```flow
module bollinger_bands

/// Calculates Bollinger Bands: a middle band (SMA) with upper
/// and lower bands at a specified number of standard deviations.

record BollingerResult:
    upper: float[N]
    middle: float[N]
    lower: float[N]

pub stage calculate(
    prices: stream<float[N]>,
    window: int,
    num_std: float
) -> stream<BollingerResult>:
    let windowed = prices |> rolling(window)
    let middle = windowed |> mean()
    let vol = windowed |> std()
    let upper = middle |> add(vol |> multiply(num_std))
    let lower = middle |> subtract(vol |> multiply(num_std))
    return BollingerResult{upper: upper, middle: middle, lower: lower}
```

That's the entire module. The compiler infers everything else.

### 18.6 System Automation Script

```flow
#!/usr/bin/env flow run
import std.io
import std.os
import std.datetime
import std.fmt

let log_dir = os.env("LOG_DIR") |> unwrap_or("/var/log/myapp")
let cutoff = datetime.now() |> datetime.subtract(days=30)

let old_files = list_dir(log_dir)?
    |> filter(fn(f): f.extension == "log")
    |> filter(fn(f): f.modified < cutoff)

print("Found {old_files |> length()} files older than 30 days")

let freed = 0
for file in old_files:
    print("  Deleting: {file.name} ({fmt.bytes(file.size)})")
    delete_file(file.path)?

let total = old_files |> map(fn(f): f.size) |> sum()
print("Cleanup complete. Freed {fmt.bytes(total)}")
```

---

## 19. LLM & GUI Builder Implications

### 19.1 Why This Design Works for LLMs

| Language Property | LLM Benefit |
|------------------|-------------|
| One way to do everything | LLM never chooses between equivalent patterns |
| No methods on records | Always `fn_name(record)`, never ambiguous |
| No inheritance/traits | No "which abstraction to use" decisions |
| No generics | No complex type parameter reasoning |
| Exhaustive match | Compiler catches missing cases even if LLM forgets |
| Result/Option instead of exceptions | Error handling is explicit and mechanical |
| No implicit conversions | Type errors caught immediately, LLM can fix |
| No null | LLM never produces null pointer bugs |
| No global state | Every function's behavior is determined by its inputs |

### 19.2 Why This Design Works for GUI Block Builders

The entire language maps to visual blocks:

| Block Type | Language Concept |
|-----------|-----------------|
| Data block (blue) | `record` definition |
| Function block (green) | `fn` with inputs → outputs |
| Stage block (orange) | `stage` with stream in → stream out |
| Pipe wire (line) | `\|>` connecting output to input |
| Value block (gray) | `let` binding |
| Branch block (yellow) | `match` or `if/else` |
| I/O block (red) | `tap` / `emit` / `print` |

No block needs to represent methods, inheritance, trait bounds, generic parameters, or exception handlers. The GUI vocabulary is small and complete.

### 19.3 Module Authoring for LLMs

The LLM workflow for creating a module:

```
1. Human says: "I want Bollinger Bands"

2. LLM writes:
   - Module name
   - Doc comment (/// description)
   - Record for output data (if needed)
   - One pub stage with the pipeline

3. Compiler does:
   - Type inference
   - Purity analysis
   - Parallel profile
   - Memory estimation
   - Interface matching
   - Manifest generation
   - Pre-flight validation

4. Result: publishable module
```

The LLM's job is steps 1-2. The compiler does steps 3-4. The human reviews step 4's output (the pre-flight report) and approves publication.

---

*This annex defines the complete programming model for OctoFlow v1. It supersedes the general-purpose language additions in Annex A, Section 3. Annex A retains the dataflow-specific primitives (streams, pipes, stages, taps, accumulators, temporal markers) and the module ecosystem design.*
