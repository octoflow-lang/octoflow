# OctoFlow Coding Bible

**The definitive guide to writing OctoFlow code.**
**For humans and LLMs alike.**

Version: 0.1 — Aligned to Phase 4 (61 tests passing)
Date: February 16, 2026

---

## 1. The Prime Directive

**OctoFlow is dataflow. Data flows through stages. Stages transform data. The GPU executes stages in parallel.**

Every line of OctoFlow should answer one question: *"What happens to the data?"*

If you're not transforming data, you're doing it wrong.

```flow
// GOOD — data flows, transforms, arrives
stream prices = tap("market.csv")
stream normalized = prices |> subtract(mn) |> divide(range)
emit(normalized, "output.csv")

// BAD — imperative thinking, no flow
let x = read_file("market.csv")
let y = []
for i in x:
    y.append((i - mn) / range)
write_file(y, "output.csv")
```

OctoFlow has no `for` loops. No `while` loops. No mutation. No variable reassignment. Data flows forward. Always forward.

---

## 2. The Eight Arms

Every operation belongs to exactly one arm. Know which arm you're using.

```
Arm 1: ARITHMETIC     Numbers in, numbers out. Element-wise.
Arm 2: COMPARISON     Numbers in, decisions out. Element-wise.
Arm 3: REDUCTION      Stream in, scalar out. Aggregation.
Arm 4: TEMPORAL       Stream in, stream out. Time-aware.
Arm 5: SHAPE          Stream in, different shape out. Structural.
Arm 6: LOGIC          Booleans in, booleans out. Element-wise.
Arm 7: MATH           Numbers in, numbers out. Transcendental.
Arm 8: TRANSFORM      Type in, different type out. Conversion.
```

**Rule: Name your intent by arm, not by implementation.**

```flow
// GOOD — reader knows the intent
stream clamped = data |> clamp(0.0, 255.0)       // Arm 2: comparison
stream average = data |> mean()                    // Arm 3: reduction
stream smooth = data |> ema(0.1)                   // Arm 4: temporal

// BAD — hides intent behind manual math
stream clamped = data |> max(0.0) |> min(255.0)   // Works but less clear
stream average = data |> sum() |> divide(n)         // Manual reduction
```

Use the highest-level vanilla op available. The compiler optimizes better when it knows your intent.

---

## 3. Pipeline Composition

### 3.1 One Stage Per Transform

Each pipe stage should do ONE thing. The compiler fuses them.

```flow
// GOOD — each stage is one clear transform
stream result = pixels
    |> subtract(128.0)           // center around zero
    |> multiply(contrast)        // apply contrast
    |> add(128.0)                // shift back
    |> clamp(0.0, 255.0)        // keep in range

// BAD — trying to be clever in one stage
stream result = pixels |> multiply(contrast) |> add(128.0 * (1.0 - contrast))
```

The compiler fuses `subtract → multiply → add → clamp` into ONE GPU kernel. You don't pay for readability. Write clearly, let the compiler optimize.

### 3.2 Pipeline Length

Keep pipelines between 3-10 stages. Under 3 is trivial. Over 10, break into named functions.

```flow
// GOOD — complex pipeline broken into named functions
fn color_correct(amount: float):
    subtract(128.0) |> multiply(amount) |> add(128.0) |> clamp(0.0, 255.0)

fn warm_grade():
    add(10.0) |> clamp(0.0, 255.0)

stream final = pixels
    |> color_correct(1.3)
    |> warm_grade()
    |> gamma(0.95)

// BAD — 15 stages in one pipeline, unreadable
stream final = pixels |> subtract(128.0) |> multiply(1.3) |> add(128.0)
    |> clamp(0.0, 255.0) |> add(10.0) |> clamp(0.0, 255.0) |> divide(255.0)
    |> pow(0.95) |> multiply(255.0) |> floor() |> ...
```

### 3.3 Pipe Alignment

Align pipe operators vertically. One stage per line.

```flow
// GOOD — vertical alignment, scannable
stream processed = data
    |> normalize()
    |> ema(0.1)
    |> multiply(scale)
    |> add(offset)
    |> round()

// BAD — horizontal chaining, hard to read
stream processed = data |> normalize() |> ema(0.1) |> multiply(scale) |> add(offset) |> round()
```

Exception: trivial 2-stage pipelines can be one line.

```flow
// OK — short enough to be clear on one line
stream clamped = data |> clamp(0.0, 1.0)
stream doubled = data |> multiply(2.0)
```

---

## 4. Naming Conventions

### 4.1 Streams

Streams are nouns. They describe WHAT the data IS, not what was done to it.

```flow
// GOOD — describes the data
stream prices = tap("prices.csv")
stream normalized_prices = prices |> normalize()
stream smooth_prices = normalized_prices |> ema(0.1)
stream signals = smooth_prices |> subtract(threshold)

// BAD — describes the action
stream read_data = tap("prices.csv")
stream after_normalize = read_data |> normalize()
stream smoothed = after_normalize |> ema(0.1)
stream subtracted = smoothed |> subtract(threshold)
```

### 4.2 Scalars

Scalars are short, mathematical. They're intermediate values, not entities.

```flow
// GOOD
let mn = data |> min()
let mx = data |> max()
let range = mx - mn
let avg = data |> mean()
let n = data |> count()

// BAD — too verbose for intermediate math
let minimum_value = data |> min()
let maximum_value = data |> max()
let data_range = maximum_value - minimum_value
```

### 4.3 Functions

Functions are verbs or adjective phrases. They describe the transformation.

```flow
// GOOD
fn brightness(amount):
fn contrast(factor):
fn gamma(g):
fn normalize():
fn warm_grade():
fn denoise_temporal(strength):

// BAD
fn do_brightness(amount):          // redundant "do"
fn apply_contrast(factor):         // redundant "apply"
fn process_gamma(g):               // redundant "process"
fn my_normalize():                 // unnecessary prefix
```

### 4.4 Module Files

Module files are singular nouns describing the domain.

```
GOOD:                   BAD:
  filters.flow            my_filters.flow
  color.flow              color_stuff.flow
  temporal.flow           time_based_operations.flow
  transform.flow          utils.flow
  audio.flow              helpers.flow
```

Never name a file `utils.flow` or `helpers.flow`. If you can't name the domain, the functions don't belong together.

---

## 5. Functions

### 5.1 Functions Are Pipeline Fragments

A function defines a reusable section of pipeline. It receives the stream implicitly through the pipe operator.

```flow
// Function definition — a named pipeline fragment
fn brightness(amount):
    add(amount) |> clamp(0.0, 255.0)

fn contrast(factor):
    subtract(128.0) |> multiply(factor) |> add(128.0) |> clamp(0.0, 255.0)

fn gamma(g):
    divide(255.0) |> pow(1.0 / g) |> multiply(255.0)

// Usage — composes naturally with pipe
stream output = pixels
    |> brightness(15.0)
    |> contrast(1.2)
    |> gamma(2.2)
```

### 5.2 Keep Functions Small

A function should be 1-6 pipeline stages. If it's longer, decompose.

```flow
// GOOD — small, composable functions
fn center():
    subtract(128.0)

fn uncenter():
    add(128.0)

fn contrast(factor):
    center() |> multiply(factor) |> uncenter() |> clamp(0.0, 255.0)

// BAD — monolithic function
fn full_color_pipeline(brightness, contrast, gamma, saturation):
    add(brightness) |> clamp(0.0, 255.0) |> subtract(128.0)
    |> multiply(contrast) |> add(128.0) |> clamp(0.0, 255.0)
    |> divide(255.0) |> pow(1.0 / gamma) |> multiply(255.0)
    |> ... 12 more stages ...
```

### 5.3 Functions Can Call Functions

Composition is the primary abstraction mechanism.

```flow
fn brightness(amount):
    add(amount) |> clamp(0.0, 255.0)

fn contrast(factor):
    subtract(128.0) |> multiply(factor) |> add(128.0) |> clamp(0.0, 255.0)

fn enhance(bright, cont):
    brightness(bright) |> contrast(cont)

// "enhance" composes "brightness" and "contrast"
// Compiler sees through all layers and fuses into one kernel
stream output = pixels |> enhance(15.0, 1.3)
```

---

## 6. Modules

### 6.1 One Domain Per File

Each `.flow` file is a module. Each module covers one domain.

```
PROJECT STRUCTURE:
  pipeline.flow          Main pipeline (entry point)
  filters.flow           Image filter functions
  color.flow             Color grading functions
  temporal.flow          Temporal processing functions
  transform.flow         Geometric transform functions
  audio.flow             Audio processing functions
```

### 6.2 Import With `use`

```flow
// Import a module
use filters
use color

// Use with dotted syntax
stream output = pixels
    |> filters.brightness(15.0)
    |> color.warm_grade()
    |> filters.contrast(1.2)
```

### 6.3 Module Design Rules

**Rule 1: Every function in a module should use the same arm(s).**

```flow
// GOOD — filters.flow: all Arm 1 + Arm 2 (arithmetic + comparison)
fn brightness(amount):
    add(amount) |> clamp(0.0, 255.0)
fn contrast(factor):
    subtract(128.0) |> multiply(factor) |> add(128.0) |> clamp(0.0, 255.0)
fn exposure(stops):
    multiply(pow(2.0, stops)) |> clamp(0.0, 255.0)

// BAD — mixing unrelated concerns in one module
fn brightness(amount):     // Arm 1: arithmetic
    add(amount)
fn sort_data():            // Arm 5: shape — doesn't belong here
    sort()
fn log_message():          // Side effect — definitely doesn't belong
    print("hello")
```

**Rule 2: Modules should be small.** 5-15 functions per module. If a module has 30+ functions, split it.

**Rule 3: No circular dependencies.** If `A` uses `B`, `B` cannot use `A`.

```
GOOD:                      BAD:
  pipeline uses filters      filters uses color
  pipeline uses color        color uses filters (circular!)
  filters is independent
  color is independent
```

---

## 7. Data Flow Patterns

### 7.1 The Standard Pipeline

Most OctoFlow programs follow this pattern:

```flow
// 1. SOURCE — where data comes from
stream input = tap("data.csv")

// 2. PREPARE — get data ready
let mn = input |> min()
let mx = input |> max()
stream clean = input |> subtract(mn) |> divide(mx - mn)

// 3. PROCESS — the actual work
stream processed = clean
    |> ema(0.1)
    |> multiply(scale)

// 4. FINALIZE — prepare for output
stream final = processed |> clamp(0.0, 1.0) |> round()

// 5. SINK — where data goes
emit(final, "output.csv")
```

Source -> Prepare -> Process -> Finalize -> Sink. Five sections. Always in this order.

### 7.2 The Reduce-Then-Map Pattern

When you need statistics to drive element-wise operations:

```flow
// Reduce: extract scalars from the stream
let mn = data |> min()             // GPU reduce -> scalar
let mx = data |> max()             // GPU reduce -> scalar
let range = mx - mn                // CPU scalar math

// Map: use scalars in element-wise operations
stream normalized = data
    |> subtract(mn)                 // GPU map: uses scalar
    |> divide(range)                // GPU map: uses scalar

// The compiler:
//   1. Dispatches two GPU reduce kernels (min, max)
//   2. Computes range on CPU
//   3. Dispatches one FUSED GPU map kernel (subtract + divide)
```

This is the most common pattern in data processing. Statistics first, then transform.

### 7.3 The Multi-Stream Pattern

When multiple streams need the same preparation:

```flow
stream raw = tap("data.csv")

// Shared preparation
let avg = raw |> mean()
stream centered = raw |> subtract(avg)

// Branch into different analyses
stream smooth = centered |> ema(0.1)
stream volatile = centered |> abs() |> ema(0.05)
stream momentum = centered |> cumsum()

// Each branch can emit independently
emit(smooth, "smooth.csv")
emit(volatile, "volatility.csv")
emit(momentum, "momentum.csv")
```

### 7.4 The Filter Library Pattern (for OctoMedia)

```flow
// filters.flow — a library of composable filters

fn brightness(amount):
    add(amount) |> clamp(0.0, 255.0)

fn contrast(factor):
    subtract(128.0) |> multiply(factor) |> add(128.0) |> clamp(0.0, 255.0)

fn gamma(g):
    divide(255.0) |> pow(1.0 / g) |> multiply(255.0)

fn invert():
    multiply(-1.0) |> add(255.0)

fn threshold(t):
    // Values above t become 255, below become 0
    subtract(t) |> clamp(0.0, 255.0) |> ceil() |> multiply(255.0)

fn exposure(stops):
    multiply(pow(2.0, stops)) |> clamp(0.0, 255.0)

fn solarize(t):
    // Above threshold: invert. Below: keep.
    // Approximation using math ops
    subtract(t) |> abs() |> add(t) |> clamp(0.0, 255.0)
```

```flow
// pipeline.flow — uses the filter library
use filters

stream pixels = tap("image.csv")
stream graded = pixels
    |> filters.brightness(10.0)
    |> filters.contrast(1.2)
    |> filters.gamma(2.2)
emit(graded, "output.csv")
```

---

## 8. Scalar Expressions

### 8.1 Scalars Come From Reductions

```flow
let mn = data |> min()          // Arm 3 reduce -> scalar
let mx = data |> max()          // Arm 3 reduce -> scalar
let avg = data |> mean()        // Arm 3 reduce -> scalar
let total = data |> sum()       // Arm 3 reduce -> scalar
```

### 8.2 Scalar Math Is CPU

Scalar-to-scalar operations run on CPU. This is correct — one value doesn't benefit from GPU parallelism.

```flow
let range = mx - mn             // CPU: one subtraction
let midpoint = (mn + mx) / 2.0  // CPU: add + divide
let scale = 1.0 / range         // CPU: one division
```

### 8.3 Scalars Feed Into Maps

The power of scalars is using them in GPU element-wise operations:

```flow
// The pipeline: GPU reduce -> CPU scalar -> GPU map
let mn = data |> min()                          // GPU: parallel reduce
let mx = data |> max()                          // GPU: parallel reduce
let range = mx - mn                             // CPU: scalar math
stream normalized = data |> subtract(mn) |> divide(range)  // GPU: parallel map
```

### 8.4 Print Scalars for Debugging

```flow
let avg = data |> mean()
print avg                        // Prints scalar value during execution

let mn = data |> min()
let mx = data |> max()
print mn
print mx
```

Use `print` for debugging. Remove before production.

---

## 9. Performance Guidelines

### 9.1 The Compiler Fuses — Trust It

```flow
// These three stages become ONE GPU kernel:
data |> subtract(mn) |> divide(range) |> clamp(0.0, 1.0)

// The compiler sees: Normalize(mn, range) + Clamp(0, 1) -> FUSED
// One memory read, three operations, one memory write.
// You cannot beat this by hand.
```

**Never try to manually fuse operations.** Write clear, separated stages. The compiler's fusion engine combines them optimally.

### 9.2 Reduce Before Map

When you need statistics, always reduce FIRST, then map. Don't interleave.

```flow
// GOOD — all reduces together, then all maps
let mn = data |> min()
let mx = data |> max()
let avg = data |> mean()
let std = data |> std()

stream result = data
    |> subtract(avg)
    |> divide(std)
    |> clamp(-3.0, 3.0)

// BAD — reduces scattered between maps
let mn = data |> min()
stream step1 = data |> subtract(mn)
let mx = step1 |> max()              // reduces on intermediate stream
stream step2 = step1 |> divide(mx)
let avg = step2 |> mean()            // reduces on another intermediate
stream result = step2 |> subtract(avg)
```

Scattered reduces force unnecessary GPU dispatches and intermediate storage.

### 9.3 Clamp After Arithmetic

Always clamp after operations that could produce out-of-range values:

```flow
// GOOD — clamp after each operation that could overflow
stream result = pixels
    |> multiply(2.0)           // could exceed 255
    |> clamp(0.0, 255.0)      // safety
    |> add(50.0)               // could exceed 255 again
    |> clamp(0.0, 255.0)      // safety

// BETTER — clamp once at the end if all ops are bounded
stream result = pixels
    |> subtract(128.0)         // range: -128 to 127
    |> multiply(1.2)           // range: -153.6 to 152.4
    |> add(128.0)              // range: -25.6 to 280.4
    |> clamp(0.0, 255.0)      // one clamp at the end
```

The compiler fuses the chain including the clamp. One clamp at the end is cleaner and equally fast.

### 9.4 Avoid Unnecessary Intermediate Streams

```flow
// GOOD — continuous pipeline, compiler optimizes as one unit
stream output = data
    |> normalize()
    |> ema(0.1)
    |> multiply(100.0)
    |> round()

// BAD — unnecessary intermediate variables
stream a = data |> normalize()
stream b = a |> ema(0.1)
stream c = b |> multiply(100.0)
stream d = c |> round()
```

Named intermediates are only useful when the intermediate is used more than once or when it aids readability at a logical boundary.

```flow
// GOOD USE of intermediate — used twice
stream centered = data |> subtract(avg)
stream positive = centered |> abs()
stream smoothed = centered |> ema(0.1)     // both use centered

// GOOD USE of intermediate — logical boundary
stream cleaned = raw_data |> normalize() |> clamp(0.0, 1.0)
// --- analysis phase ---
stream trend = cleaned |> ema(0.2)
stream volatility = cleaned |> rolling_std(20)
```

---

## 10. Error Prevention

### 10.1 Division Safety

Division by zero produces infinity or NaN on GPU. Always guard:

```flow
// GOOD — ensure denominator is never zero
let range = mx - mn
// If range is 0, all values are identical — output zeros or ones
stream normalized = data |> subtract(mn) |> divide(range)

// BETTER (when pre-flight system is available):
// The compiler will warn about potential divide-by-zero at Arm 3
```

### 10.2 Numeric Range Awareness

Know your data ranges. GPU float32 has ~7 decimal digits of precision.

```flow
// DANGEROUS — exp can overflow float32
data |> exp()     // exp(89) = 4.5e38 (near float32 max)
                  // exp(90) = infinity

// SAFE — clamp input before exp
data |> clamp(-80.0, 80.0) |> exp()   // output stays in float32 range
```

```flow
// DANGEROUS — pow with large exponents
data |> pow(10.0)    // 100^10 = 1e20 (fine)
                     // 1000^10 = 1e30 (fine)
                     // 10000^10 = 1e40 (overflow!)

// SAFE — know your input range or use log-space
data |> log() |> multiply(10.0) |> exp()   // equivalent but safer
```

### 10.3 Pixel Conventions

For image processing, establish your convention and stick to it:

```
CONVENTION A: uint8 range [0, 255] — integer pixel values
  brightness: add(20.0) |> clamp(0.0, 255.0)
  normalize:  divide(255.0)   -> converts to [0, 1]
  denormalize: multiply(255.0) -> converts back to [0, 255]

CONVENTION B: float range [0.0, 1.0] — normalized pixel values
  brightness: add(0.08) |> clamp(0.0, 1.0)
  This is mathematically cleaner for operations like gamma.

RECOMMENDATION: Use [0, 255] for I/O, [0, 1] for internal processing.
```

```flow
// Standard pattern: normalize -> process -> denormalize
stream input = tap("pixels.csv")               // [0, 255]
stream normalized = input |> divide(255.0)      // [0, 1]
stream processed = normalized
    |> gamma(2.2)                               // works naturally in [0,1]
    |> contrast(1.3)
stream output = processed
    |> multiply(255.0)                          // back to [0, 255]
    |> clamp(0.0, 255.0)
    |> round()                                  // integer pixels
emit(output, "output.csv")
```

---

## 11. LLM Code Generation Rules

These rules are for LLMs generating OctoFlow code. Follow them exactly.

### 11.1 Always Use Vanilla Ops

Never generate raw math when a vanilla op exists.

```flow
// GENERATE THIS:
data |> clamp(0.0, 255.0)
data |> abs()
data |> normalize()

// NEVER GENERATE THIS:
data |> max(0.0) |> min(255.0)              // use clamp
data |> pow(2.0) |> sqrt()                  // use abs
data |> subtract(mn) |> divide(mx - mn)     // use normalize (when available)
```

### 11.2 Always Clamp Pixel Output

Any pipeline producing pixel values MUST end with clamp:

```flow
// ALWAYS do this for pixel output
stream output = pixels |> [operations] |> clamp(0.0, 255.0)
```

### 11.3 Always Include tap and emit

Every generated program must have a data source and sink:

```flow
stream input = tap("input.csv")          // REQUIRED: data source
// ... pipeline ...
emit(output, "output.csv")               // REQUIRED: data sink
```

### 11.4 Prefer Functions Over Inline

When generating complex pipelines, create named functions:

```flow
// GENERATE THIS:
fn warm_grade():
    add(10.0) |> clamp(0.0, 255.0)

fn enhance(bright, cont):
    brightness(bright) |> contrast(cont)

stream output = pixels |> warm_grade() |> enhance(15.0, 1.2)

// NOT THIS:
stream output = pixels |> add(10.0) |> clamp(0.0, 255.0)
    |> add(15.0) |> clamp(0.0, 255.0) |> subtract(128.0)
    |> multiply(1.2) |> add(128.0) |> clamp(0.0, 255.0)
```

### 11.5 Comment the Intent, Not the Mechanism

```flow
// GOOD comments:
stream centered = data |> subtract(avg)      // remove DC offset
stream normalized = data |> divide(range)    // scale to [0, 1]
stream detrended = data |> subtract(trend)   // isolate cyclical component

// BAD comments:
stream centered = data |> subtract(avg)      // subtract average from data
stream normalized = data |> divide(range)    // divide by range
```

### 11.6 Standard Preamble for Generated Code

LLM-generated code should start with:

```flow
// Generated by OctoFlow LLM Frontend
// Intent: [natural language description of what user asked for]
// Arms used: [list which arms this pipeline uses]
// Input: [expected input format]
// Output: [expected output format]
```

Example:

```flow
// Generated by OctoFlow LLM Frontend
// Intent: Brighten image by 15%, increase contrast, apply warm color grade
// Arms used: Arm 1 (arithmetic), Arm 2 (comparison), Arm 7 (math)
// Input: pixel values [0, 255]
// Output: pixel values [0, 255]

use filters

stream pixels = tap("input.csv")
stream output = pixels
    |> filters.brightness(38.0)      // +15% of 255
    |> filters.contrast(1.2)
    |> filters.gamma(0.95)           // slight warm lift
    |> clamp(0.0, 255.0)
emit(output, "output.csv")
```

### 11.7 Arm Selection Guide for LLMs

When translating natural language to OctoFlow:

```
"brighter/darker/lighter"        -> Arm 1: add/subtract
"more/less contrast"             -> Arm 1: subtract(128), multiply, add(128)
"warmer/cooler"                  -> Arm 1: add to red, subtract from blue
"sharper"                        -> Arm 1: subtract blurred, multiply, add
"smoother/softer"                -> Arm 4: ema (temporal smoothing)
"remove noise"                   -> Arm 4: ema (temporal denoise)
"normalize/standardize"          -> Arm 3: reduce, then Arm 1: subtract/divide
"average/total/count"            -> Arm 3: reduce (mean/sum/count)
"limit/bound/cap"                -> Arm 2: clamp
"gamma/power curve"              -> Arm 7: pow
"logarithmic scale"              -> Arm 7: log
"fade in/out"                    -> Arm 1: multiply with ramp
"invert/negative"                -> Arm 1: multiply(-1), add(255)
"round to integers"              -> Arm 7: round/floor/ceil
"absolute value"                 -> Arm 1: abs
"modular/wrap around"            -> Arm 1: mod
"oscillating/wave"               -> Arm 7: sin/cos
"exponential growth/decay"       -> Arm 4: ema/decay, or Arm 7: exp
```

---

## 12. File Organization

### 12.1 Project Structure

```
my-project/
  pipeline.flow           Entry point (main pipeline)
  filters.flow            Image/signal filter functions
  color.flow              Color grading functions
  temporal.flow           Temporal processing functions
  data/
    input.csv             Input data
  output/
    (generated files go here)
```

### 12.2 Entry Point Convention

The main pipeline file is always the entry point. It uses modules and wires everything together.

```flow
// pipeline.flow — the entry point
use filters
use color
use temporal

stream input = tap("data/input.csv")

let avg = input |> mean()
stream centered = input |> subtract(avg)

stream processed = centered
    |> temporal.smooth(0.1)
    |> filters.contrast(1.2)
    |> color.warm_grade()
    |> clamp(0.0, 255.0)

emit(processed, "output/result.csv")
```

### 12.3 Module File Convention

Module files contain ONLY function definitions. No tap, no emit, no stream declarations, no scalar computations.

```flow
// GOOD — filters.flow (pure functions)
fn brightness(amount):
    add(amount) |> clamp(0.0, 255.0)

fn contrast(factor):
    subtract(128.0) |> multiply(factor) |> add(128.0) |> clamp(0.0, 255.0)

fn gamma(g):
    divide(255.0) |> pow(1.0 / g) |> multiply(255.0)

// BAD — module with side effects
stream data = tap("some_file.csv")       // NO: modules don't read files
fn brightness(amount):
    add(amount) |> clamp(0.0, 255.0)
emit(data, "output.csv")                  // NO: modules don't write files
```

---

## 13. The flow graph Command

Use `flow graph` to inspect what the compiler decided BEFORE execution.

```bash
$ flow graph pipeline.flow

  stream input       [I/O:TAP]     data/input.csv
  scalar avg         [GPU:RED]     mean
  stream centered    [GPU:MAP]     subtract(avg)
  stream smooth      [GPU:TMP]     ema(0.1)
  stream contrasted  [FUSED]       subtract(128) -> multiply(1.2) -> add(128) -> clamp(0,255)
  stream output      [I/O:EMIT]    output/result.csv

  GPU dispatches: 3 (reduce, temporal, fused map)
  Fusion: 4 stages -> 1 kernel
```

**Read the graph output. Verify:**

1. Are reduces happening before maps? (Should be.)
2. Are map stages being fused? (Should be.)
3. Is the dispatch count reasonable? (Fewer = better.)
4. Are there unnecessary intermediate streams? (Remove them.)

---

## 14. Common Recipes

### 14.1 Normalize to [0, 1]

```flow
let mn = data |> min()
let mx = data |> max()
stream normalized = data |> subtract(mn) |> divide(mx - mn)
```

### 14.2 Z-Score Standardization

```flow
let avg = data |> mean()
let sd = data |> std()
stream standardized = data |> subtract(avg) |> divide(sd)
```

### 14.3 Image Brightness + Contrast + Gamma

```flow
use filters

stream output = pixels
    |> filters.brightness(15.0)
    |> filters.contrast(1.2)
    |> filters.gamma(2.2)
    |> clamp(0.0, 255.0)
```

### 14.4 Temporal Smoothing (Denoise)

```flow
stream smooth = noisy_data |> ema(0.15)
```

### 14.5 Exponential Moving Average Crossover

```flow
stream fast = data |> ema(0.1)      // fast EMA (short window)
stream slow = data |> ema(0.03)     // slow EMA (long window)
// Future: stream signal = fast |> subtract(slow) — needs stream-to-stream ops
```

### 14.6 Clamp to Valid Range

```flow
stream safe = data |> clamp(lower_bound, upper_bound)
```

### 14.7 Log Transform

```flow
stream log_data = data |> clamp(0.001, 999999.0) |> log()
// Always clamp before log to avoid log(0) = -infinity
```

### 14.8 Power Law / Gamma Correction

```flow
// Image gamma correction
stream corrected = pixels
    |> divide(255.0)              // normalize to [0, 1]
    |> pow(1.0 / gamma_value)     // apply gamma curve
    |> multiply(255.0)            // denormalize
    |> clamp(0.0, 255.0)
```

---

## 15. Anti-Patterns

### 15.1 Don't Fight the Dataflow

```flow
// ANTI-PATTERN: trying to use OctoFlow like Python
// OctoFlow has no for loops, no mutation, no if/else on elements
// If you're thinking "for each element, if X then Y" — use clamp, min, max instead

// BAD THINKING: "for each pixel, if > 200 set to 200"
// GOOD THINKING: data |> min(200.0)

// BAD THINKING: "for each value, if negative make positive"
// GOOD THINKING: data |> abs()

// BAD THINKING: "for each value, clip between 0 and 1"
// GOOD THINKING: data |> clamp(0.0, 1.0)
```

### 15.2 Don't Chain Redundant Clamps

```flow
// BAD — redundant clamps waste a few cycles
data |> add(10.0) |> clamp(0.0, 255.0) |> add(5.0) |> clamp(0.0, 255.0)

// GOOD — if you know the combined range, clamp once
data |> add(15.0) |> clamp(0.0, 255.0)

// EXCEPTION — if intermediate clamping changes the result:
data |> multiply(3.0) |> clamp(0.0, 255.0) |> subtract(100.0) |> clamp(0.0, 255.0)
// Here removing the first clamp would change output. Keep both.
```

### 15.3 Don't Use Math When Vanilla Ops Exist

```flow
// BAD
data |> multiply(data) |> sqrt()        // manual abs via x^2->sqrt(x)
// GOOD
data |> abs()

// BAD
data |> multiply(-1.0)                  // manual negate
// GOOD
data |> negate()

// BAD
data |> subtract(floor_of_data)         // manual mod
// GOOD
data |> mod(divisor)
```

### 15.4 Don't Create Unused Streams

```flow
// BAD — computed but never used
stream debug = data |> mean()            // never emitted or used
stream output = data |> normalize()
emit(output, "out.csv")

// GOOD — everything has a purpose
stream output = data |> normalize()
emit(output, "out.csv")
```

---

## 16. Iteration Without Loops

OctoFlow has no `for`, `while`, or `loop` keywords. But it HAS iteration — through the five GPU patterns already built. Each pattern IS iteration, just parallel.

```
IMPERATIVE (loop):              OCTOFLOW (pattern):
---                             ---
for each element: do X          -> Map (all elements, simultaneously)
for each element: accumulate    -> Reduce (tree, parallel)
for each element: running total -> Scan (prefix sum, parallel)
for each timestep: update       -> Temporal (sequential on time, parallel on data)
for step 1, then step 2...      -> Pipeline (stages, fused)
```

### 16.1 Map IS the For-Each Loop

```python
# Python loop
result = []
for x in data:
    result.append(x * 2.0)
```

```flow
// OctoFlow — same result, 1000 threads simultaneously
stream result = data |> multiply(2.0)
```

The GPU launches 1000 threads. Each thread processes one element. No loop. No iteration variable. No sequential dependency. All elements computed at once.

### 16.2 Reduce IS the Accumulation Loop

```python
# Python loop
total = 0
for x in data:
    total += x
```

```flow
// OctoFlow — parallel tree reduction
let total = data |> sum()
```

The GPU does this in log2(N) steps instead of N steps. 1 million elements -> 20 steps, not 1 million steps. That's the tree reduction pattern from Phase 1.

### 16.3 Scan IS the Running-Total Loop

```python
# Python loop
running = []
total = 0
for x in data:
    total += x
    running.append(total)
```

```flow
// OctoFlow — Hillis-Steele parallel prefix sum
stream running = data |> cumsum()
```

The scan pattern from Phase 1. Parallel prefix computation. Every element gets its running total, computed in log2(N) parallel steps.

### 16.4 Temporal IS the Stateful Loop

```python
# Python loop
smoothed = []
prev = data[0]
for x in data:
    prev = 0.1 * x + 0.9 * prev
    smoothed.append(prev)
```

```flow
// OctoFlow — temporal pattern with OpPhi loop
stream smoothed = data |> ema(0.1)
```

This one IS sequential on the time axis — each value depends on the previous. But it's parallel across instruments. 1000 instruments x 1000 timesteps? Sequential on 1000 timesteps, parallel across 1000 instruments. That's the temporal SPIR-V pattern with OpPhi from Phase 1.

### 16.5 Pipeline IS the Multi-Step Loop

```python
# Python loop
result = []
for x in data:
    step1 = x - mean
    step2 = step1 / std
    step3 = max(min(step2, 3.0), -3.0)
    result.append(step3)
```

```flow
// OctoFlow — fused pipeline, one GPU kernel
stream result = data
    |> subtract(avg)
    |> divide(std)
    |> clamp(-3.0, 3.0)
```

The compiler fuses subtract + divide + clamp into ONE kernel. Each thread does all three operations on its element. One memory read, three operations, one memory write. Faster than three separate loops even in C.

### 16.6 The Full Pattern Map

```
IMPERATIVE PATTERN            OCTOFLOW EQUIVALENT           GPU PATTERN
----------------------------------------------------------------------
for each x: transform(x)     |> vanilla_op()               Parallel Map
for each x: total += x       |> sum()                      Tree Reduction
for each x: if x > t         |> clamp(t, max)              Parallel Map
running = f(running, x)      |> ema() / decay() / cumsum() Temporal
repeat N times: refine        |> repeat(N, fn)              Temporal (future)
repeat until converged        |> converge(tol, fn)          Temporal (future)
for each pair: combine        |> zip(other) |> map_op()     Parallel Map
prefix sum / scan             |> scan(add)                  Hillis-Steele Scan
```

### 16.7 Conditionals Per Element — Branchless Math

"For each pixel, if above threshold set to white, else set to black."

OctoFlow uses math instead of branching. GPUs hate branches — they cause thread divergence.

```python
# Python: branch per element
for pixel in data:
    if pixel > threshold:
        result = 255
    else:
        result = 0
```

```flow
// OctoFlow: branchless using vanilla ops
data |> subtract(threshold)      // positive = above, negative = below
     |> clamp(0.0, 1.0)         // 0 or positive -> clamped to [0,1]
     |> ceil()                   // 0 stays 0, anything positive -> 1
     |> multiply(255.0)          // 0 -> 0 (black), 1 -> 255 (white)
```

No branch. No `if`. Pure math. Every GPU thread executes the same instructions. Maximum parallelism.

Common branchless patterns:

```flow
// Binary threshold (0 or 255):
data |> subtract(t) |> clamp(0.0, 1.0) |> ceil() |> multiply(255.0)

// Soft threshold (gradual transition):
data |> subtract(t) |> multiply(sharpness) |> clamp(0.0, 1.0) |> multiply(255.0)

// Absolute value (equivalent to if negative: negate):
data |> abs()

// Max of two values (equivalent to if a > b: a else b):
data |> max(other_value)

// Min of two values:
data |> min(other_value)

// Clamp (equivalent to if x < lo: lo, elif x > hi: hi, else: x):
data |> clamp(lo, hi)
```

Full per-element conditionals (`if/else` inside a map) need either:
- Branchless GPU math (select instruction in SPIR-V) — planned
- Or `where` / `mask` operations — planned for Arm 6 (Logic)

For OctoMedia, most pixel operations don't need per-element conditionals. Brightness, contrast, gamma, blur, denoise — all pure math on every pixel, no branching.

### 16.8 Convergence — "Repeat Until Done"

```python
# Python — iterative algorithm
while error > tolerance:
    estimate = refine(estimate)
    error = compute_error(estimate)
```

Options in OctoFlow:

```
OPTION 1: Fixed iteration count (most common in practice)
  stream result = data
      |> refine()
      |> refine()
      |> refine()      // just call it N times
      |> refine()      // compiler can optimize repeated identical stages

OPTION 2: Temporal pattern with convergence (future)
  // Extend temporal ops to support custom update functions

OPTION 3: Host-side loop (future)
  // The runtime dispatches GPU stages in a loop
  // Check convergence on CPU between dispatches
```

For OctoMedia and most data processing, fixed iteration count covers 95% of cases.

### 16.9 The Mental Model Shift

```
IMPERATIVE:  "Step through each element, one at a time, doing operations"
DATAFLOW:    "Describe the transformation. All elements transform at once."

IMPERATIVE:  "Loop N times"
DATAFLOW:    "Pipeline with N stages"

IMPERATIVE:  "Accumulate into a variable"
DATAFLOW:    "Reduce to a scalar"

IMPERATIVE:  "Update state each step"
DATAFLOW:    "Temporal pattern carries state"

IMPERATIVE:  "Nested loops (i, j)"
DATAFLOW:    "2D map — GPU threads are indexed in 2D natively"
```

No loops doesn't mean no iteration. It means iteration is expressed as patterns — map, reduce, scan, temporal, pipeline — and the GPU executes them in parallel. The programmer describes WHAT transformation happens. The compiler and GPU decide HOW to iterate.

### 16.10 When OctoFlow Genuinely Can't Express Something

Some algorithms are fundamentally sequential and can't be parallelized:

```
- Algorithms where step N depends on the RESULT of step N-1 in unpredictable ways
- Graph traversal where the path depends on discovered nodes
- Recursive algorithms with data-dependent branching
```

For these cases (rare in media processing, more common in general computing):
- The temporal pattern handles fixed sequential dependencies
- CPU handles truly sequential algorithms (I/O-bound, scalar logic)
- Future `repeat` and `converge` constructs handle iterative refinement

OctoMedia doesn't need any of these. Video processing is overwhelmingly element-wise with statistics and temporal smoothing.

---

## Summary: The Ten Commandments of OctoFlow

```
I.    Data flows forward. Never backward, never in loops.
II.   One stage, one transform. The compiler fuses.
III.  Name streams as nouns (what the data IS).
IV.   Name functions as verbs (what the transform DOES).
V.    Use the highest-level vanilla op. Don't reinvent.
VI.   Reduce first, then map. Statistics before transforms.
VII.  Clamp after arithmetic. Guard your ranges.
VIII. Keep functions small. 1-6 stages. Compose, don't monolith.
IX.   One domain per module file. No utils.flow.
X.    Run flow graph. Read it. Verify fusion. Trust the compiler.
```

---

*This guide will evolve as the OctoFlow language grows. Current version covers Phase 4 capabilities (vanilla ops, functions, modules, scalars, pipe composition). Future additions: records, conditionals, stream-to-stream operations, pre-flight patterns, async/streaming patterns, GPU memory management, testing patterns.*
