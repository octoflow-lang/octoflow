# OctoFlow

A GPU-native general-purpose programming language.
Designed for the GPU from scratch. Compiled by itself. Run on any GPU.

**CPU on demand, not GPU on demand.**

2.2 MB binary. Zero dependencies. Any GPU vendor. One file download.

## Install

Download from [Releases](https://github.com/octoflow-lang/octoflow/releases/latest).

| Platform | File |
|---|---|
| Windows x64 | `octoflow-windows-x64.zip` |
| Linux x64 | `octoflow-linux-x64.tar.gz` |

Unzip. Run. No installer, no SDK, no dependencies. Just a 2.2 MB binary.

Requirements: Any GPU with Vulkan driver (NVIDIA, AMD, Intel).

## Hello World

```
print("Hello, World!")
```

```
$ octoflow run hello.flow
Hello, World!
```

## Hello GPU

```
let a = gpu_fill(1.0, 1000000)
let b = gpu_fill(2.0, 1000000)
let c = gpu_add(a, b)
let total = gpu_sum(c)
print("1M elements on GPU: {total}")
```

```
$ octoflow run hello_gpu.flow
1M elements on GPU: 3000000
```

1 million elements. GPU-parallel. One command.

## REPL

```
$ octoflow repl
OctoFlow 0.83.0 — GPU-native language
GPU: NVIDIA GeForce GTX 1660 SUPER
>>> 2 + 2
4
>>> let a = gpu_fill(1.0, 10000000)
>>> gpu_sum(a)
10000000
>>> :quit
```

## Language

```
// Variables and control flow
let mut sum = 0.0
for i in range(1, 101)
  sum = sum + i
end
print("sum 1-100: {sum}")

// Functions
fn fibonacci(n)
  let mut a = 0.0
  let mut b = 1.0
  for i in range(0, n)
    let tmp = b
    b = a + b
    a = tmp
  end
  return a
end

// Arrays
let prices = [100.0, 102.0, 101.5, 103.0, 104.5]
let avg = mean(prices)
print("average: {avg}")

// GPU compute — data stays in VRAM
let data = gpu_fill(1.0, 10000000)
let doubled = gpu_scale(data, 2.0)
let result = gpu_sum(doubled)

// Lambdas and higher-order functions
let evens = filter(numbers, fn(x) x > 0.0 end)
let squares = map_each(evens, fn(x) x * x end)

// Stream pipelines
stream photo = tap("input.jpg")
stream warm = photo |> brightness(20.0) |> contrast(1.2)
emit(warm, "output.png")
```

## Standard Library — 51 Modules

### collections — Data Structures
`use stack` `use queue` `use heap` `use graph` `use collections`

Stack, queue, min/max heap, weighted directed graph with Dijkstra shortest path, and unified collection utilities.

### data — Data Processing
`use csv` `use io` `use pipeline` `use transform` `use validate`

CSV read/write, file I/O helpers, data pipeline composition, column transforms (scale, normalize, encode), and data validation rules.

### db — In-Memory Database
`use core` `use query` `use schema`

Columnar in-memory database with SQL-like operations. Create tables, insert rows, SELECT/WHERE/JOIN/GROUP BY/ORDER BY. Import from CSV.

### ml — Machine Learning
`use regression` `use classify` `use cluster` `use nn` `use tree` `use ensemble` `use linalg` `use metrics` `use preprocess`

Linear and ridge regression, KNN and naive Bayes classification, k-means clustering, neural network primitives (dense, relu, sigmoid, softmax, SGD, dropout, batch norm), decision trees, bagging ensembles, matrix algebra, accuracy/precision/recall/F1, and train/test split with scaling.

### science — Scientific Computing
`use calculus` `use constants` `use interpolate` `use matrix` `use physics` `use signal`

Numerical differentiation and integration, physical constants, linear/bilinear interpolation, matrix operations, Euler/RK4 integrators with spring-damper and projectile dynamics, and DSP (convolution, FIR filters, windowing, peak detection).

### stats — Statistical Analysis
`use descriptive` `use correlation` `use distribution` `use hypothesis` `use risk` `use timeseries` `use math_ext`

Mean, median, stddev, skewness, kurtosis. Pearson/Spearman correlation. Normal/uniform/exponential distributions. T-test, chi-squared, ANOVA. Sharpe ratio, VaR, drawdown. SMA, EMA, MACD, Bollinger bands. Gamma, beta, erf functions.

### string — Text Processing
`use string` `use regex` `use format`

String manipulation, regular expression matching and extraction, and string formatting utilities.

### sys — System
`use args` `use env` `use memory` `use platform` `use timer`

Command-line argument parsing, environment variables, memory tracking, platform detection, and execution timing.

### time — Date and Time
`use datetime`

Date/time formatting and arithmetic.

### web — Network
`use http` `use json_util` `use url`

HTTP GET/POST client, JSON parsing and serialization, URL encoding and parsing.

### core (top-level)
`use math` `use sort` `use array_utils` `use io`

Math functions, sorting algorithms, array utilities, and file I/O.

## GPU Computing

GPU operations are built into the language. No separate kernel files, no shader compilation step.

```
// Element-wise operations (10M elements, <1ms each)
let a = gpu_fill(1.0, 10000000)
let b = gpu_add(a, a)
let c = gpu_mul(b, a)
let d = gpu_scale(c, 0.5)

// Reductions
let total = gpu_sum(d)
let maximum = gpu_max(d)
let minimum = gpu_min(d)

// Math functions
let s = gpu_sin(a)
let e = gpu_exp(a)
let r = gpu_sqrt(a)

// Matrix multiply
let result = gpu_matmul(mat_a, mat_b, rows_a, cols_a, cols_b)
```

All GPU data stays in VRAM between operations. No CPU round-trips until you need the result.

## Security

Sandboxed by default. Scripts need explicit permission flags:

```
octoflow run script.flow                        # no I/O, no network
octoflow run script.flow --allow-read            # can read files
octoflow run script.flow --allow-read --allow-net # can read files + network
```

| Flag | Grants |
|---|---|
| `--allow-read` | File system read |
| `--allow-write` | File system write |
| `--allow-net` | Network (HTTP, TCP) |
| `--allow-exec` | Subprocess execution |

## Examples

```
octoflow run examples/hello.flow
octoflow run examples/hello_gpu.flow
octoflow run examples/fractal.flow
octoflow run examples/stats.flow
octoflow run examples/csv_demo.flow --allow-read
octoflow run examples/http_demo.flow --allow-net
```

See [examples/](examples/) for all 16 runnable demos.

## Documentation

- [Quickstart](docs/quickstart.md) — install to GPU in 5 minutes
- [Language Guide](docs/language-guide.md) — full syntax reference
- [Builtins](docs/builtins.md) — all built-in functions
- [GPU Guide](docs/gpu-guide.md) — GPU computing
- [REPL](docs/repl.md) — interactive mode
- [Coding Guide](docs/CODING-GUIDE.md) — complete language reference

### LLM / AI-Assisted Development

The [Coding Guide](docs/CODING-GUIDE.md) and [Language Guide](docs/language-guide.md) are designed to work as RAG context for LLMs. Feed them to your AI assistant and it can write OctoFlow code, debug .flow scripts, and use the full stdlib — no training required.

## License

- **Standard library, examples, docs**: Apache 2.0 ([LICENSE-STDLIB](LICENSE-STDLIB))
- **Compiler binary**: Free to download and use for any purpose
- **Your .flow programs**: Yours entirely

See [LICENSE](LICENSE) for full terms.
