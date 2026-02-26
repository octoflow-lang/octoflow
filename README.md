# OctoFlow

![License: Compiler Binary - Free Use](https://img.shields.io/badge/compiler-free%20to%20use-blue)
![License: Stdlib - Apache 2.0](https://img.shields.io/badge/stdlib-Apache%202.0-green)
![Tests: 658 passing](https://img.shields.io/badge/tests-1200%20passing-brightgreen)
![Dependencies: 0](https://img.shields.io/badge/dependencies-0-brightgreen)
![Binary: 2.8 MB](https://img.shields.io/badge/binary-2.8%20MB-blue)

The first LLM-native, GPU-native programming language.

**One binary. Zero dependencies. Your AI writes it. Your GPU runs it.**

## Why OctoFlow

**Your AI already knows it.** 23 language concepts. The entire language fits in one prompt. Any LLM — Claude, ChatGPT, Copilot, Llama — can write OctoFlow fluently after reading one document.

**Zero setup friction.** One 2.8 MB binary. No installer, no SDK, no runtime, no PATH configuration. Download, unzip, run. The LLM's setup instructions are one sentence.

**Batteries included.** 246 stdlib modules across 18 domains — AI, data science, ML, media codecs, statistics, web, crypto, and more. No `pip install`. No `npm install`. No version conflicts. Everything is already there.

**Safe to run AI-generated code.** Sandboxed by default. Scripts can't read files, access the network, or run commands unless you explicitly allow it. You can paste code from any LLM and run it without worry.

**GPU speed without GPU knowledge.** Your code runs on the GPU automatically. You never think about kernels, memory management, or parallel programming. You write simple code. OctoFlow makes it fast.

## Install

**Terminal (recommended):**

```powershell
# Windows (PowerShell)
irm https://octoflow-lang.github.io/octoflow/install.ps1 | iex
```

```bash
# Linux
curl -fsSL https://octoflow-lang.github.io/octoflow/install.sh | sh
```

**Manual download:**

Download from [Releases](https://github.com/octoflow-lang/octoflow/releases/latest).

| Platform | File |
|---|---|
| Windows x64 | `octoflow-windows-x64.zip` |
| Linux x64 | `octoflow-linux-x64.tar.gz` |

Unzip. Run. No installer, no SDK, no dependencies. Just a 2.8 MB binary.

Requirements: Any GPU with Vulkan driver (NVIDIA, AMD, Intel).

## Hello World

```
print("Hello, World!")
```

```
$ octoflow run hello.flow
Hello, World!
```

## Give This to Your AI

The [Language Guide](docs/language-guide.md) is designed as LLM context. Paste it into Claude, ChatGPT, Copilot — any AI assistant — and it writes OctoFlow fluently. No fine-tuning, no training, no plugins. One document, and your AI becomes an OctoFlow developer.

Use it as:
- **System prompt** — drop it into your AI's context window
- **RAG source** — index it for retrieval-augmented generation
- **Project context** — add it to Claude Code, Cursor, or Windsurf

The language has 23 concepts. An LLM holds the entire language in context and generates correct code on the first try. Python takes 500-2000 tokens to express a GPU pipeline. OctoFlow takes 30-100.

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

1 million elements. GPU-parallel. One command. You didn't write a kernel — OctoFlow did.

## REPL

```
$ octoflow repl
OctoFlow 1.0.0 — GPU-native language
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

No semicolons. No braces. No type annotations. Blocks end with `end`. An LLM generates this correctly every time because there's nothing to get wrong.

## Standard Library — 246 Modules

246 recipes across 18 domains. Your AI picks the right ones.

| Domain | Modules | What You Get |
|---|---|---|
| **ai** | transformer, inference, generate, weight_loader | LLM inference, GGUF model loading |
| **collections** | stack, queue, heap, graph | Data structures |
| **compiler** | lexer, eval, parser, preflight, codegen, ir | Self-hosted compiler |
| **crypto** | hash, encoding, random | SHA-256, base64, CSPRNG |
| **data** | csv, io, pipeline, transform, validate | ETL and data processing |
| **db** | core, query, schema | Database abstractions |
| **formats** | gguf, json | Tensor files, JSON |
| **gpu** | Loom Engine, emitters, runtime, kernels | 73+ GPU compute operations |
| **gui** | widgets, layout, themes, events | Native GPU-rendered GUI toolkit |
| **llm** | generate, stream, chat, decompose | Run LLMs on your GPU |
| **media** | image (PNG/JPEG/GIF/BMP), video (AVI/MP4/H.264), audio (WAV) | Native codecs, no ffmpeg |
| **ml** | nn, regression, classify, cluster, tree, linalg | Machine learning primitives |
| **science** | calculus, physics, signal, matrix, optimize | Scientific computing |
| **stats** | descriptive, distribution, correlation, risk | Statistical analysis |
| **string** | string, regex, format | Text processing |
| **sys** | args, env, memory, platform, timer | System interfaces |
| **terminal** | term_image, colors | GPU-rendered images in terminal |
| **web** | http, json_util, url | HTTP client, JSON, URLs |

### Core (top-level)
`use math` `use sort` `use array_utils` `use io`

Math functions, sorting algorithms, array utilities, and file I/O.

## Security — Safe to Let AI Write Your Code

Sandboxed by default. Nothing runs unless you say so:

```
octoflow run script.flow                        # no I/O, no network — pure compute
octoflow run script.flow --allow-read            # can read files
octoflow run script.flow --allow-read --allow-net # can read files + network
```

| Flag | Grants |
|---|---|
| `--allow-read` | File system read |
| `--allow-write` | File system write |
| `--allow-net` | Network (HTTP, TCP) |
| `--allow-exec` | Subprocess execution |

Paste code from any LLM. Run it sandboxed. Review what it does. Grant permissions when you're ready. This is how AI-generated code should work.

## Under the Hood — GPU Computing

GPU operations are built into the language. No separate kernel files, no shader compilation step, no CUDA.

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

All GPU data stays in VRAM between operations. No CPU round-trips until you need the result. You write simple function calls — the compiler generates GPU compute shaders and manages everything.

## Loom Engine — The Deep End

For those who want to go deeper: OctoFlow's Loom Engine weaves thousands of GPU threads into coordinated computation. You record a dispatch chain — a sequence of kernel operations — and the GPU executes the entire chain in a single submission. No CPU round-trips between stages.

```octoflow
// Boot a compute unit, load data
let unit = loom_boot(1.0, 8194, 4096)
loom_write(unit, 0.0, input_data)

// Record dispatch chain — three kernels, one pass
loom_dispatch(unit, "init.spv", params, 32.0)
loom_dispatch(unit, "process.spv", params, 32.0)
loom_dispatch(unit, "reduce.spv", params, 32.0)

// Build, launch, read results
let prog = loom_build(unit)
loom_run(prog)
let result = loom_read(unit, 0.0, 0.0, 8194)
```

The Loom Engine powers LLM inference, database queries, compression, and multi-agent workloads — all on the same GPU runtime. 95,000+ kernel dispatches in a single submission.

See [Loom Engine Guide](docs/loom-engine.md) for details.

## Examples

```
octoflow run examples/hello.flow
octoflow run examples/hello_gpu.flow
octoflow run examples/fractal.flow
octoflow run examples/stats.flow
octoflow run examples/csv_demo.flow --allow-read
octoflow run examples/http_demo.flow --allow-net
```

See [examples/](examples/) for all runnable demos.

## Documentation

- [Quickstart](docs/quickstart.md) — install to GPU in 5 minutes
- [Language Guide](docs/language-guide.md) — full syntax reference (LLM-ready)
- [Builtins](docs/builtins.md) — all 250+ built-in functions
- [GPU Guide](docs/gpu-guide.md) — GPU computing
- [REPL](docs/repl.md) — interactive mode

## License

- **Standard library, examples, docs**: Apache 2.0 ([LICENSE-STDLIB](LICENSE-STDLIB))
- **Compiler binary**: Free to download and use for any purpose
- **Your .flow programs**: Yours entirely

See [LICENSE](LICENSE) for full terms.

## Transparency

### How It's Built

OctoFlow is vibe-coded. The architecture, design decisions, and system thinking are
human-driven. The implementation is AI-assisted — built in collaboration with LLMs
(primarily Claude), which is how a solo developer ships 200K+ lines of code across a
compiler, GPU runtime, and 246 stdlib modules.

Stated openly because honesty matters more than optics. Every line of code is reviewed,
tested (1,200+ tests passing), and understood. The AI accelerates implementation; it
doesn't replace understanding.

### Self-Hosted

69% of OctoFlow is written in OctoFlow. The compiler — lexer, parser, preflight,
evaluator, SPIR-V codegen — is written in .flow files. The remaining 31% is ~37K lines
of first-party Rust that provides the OS boundary: Vulkan GPU access, file I/O, windowing,
and network sockets. Zero external Rust crates. Zero transitive dependencies.

### Binary Size

The compiled binary is **2.8 MB**. For context: Python is ~30 MB, Node.js is ~40 MB,
Go is ~150 MB (with stdlib), Rust's compiler is ~200+ MB. OctoFlow ships a compiler,
GPU runtime, VM, 246 stdlib modules, media codecs, and a GUI toolkit in 2.8 MB.

### Looking for Maintainers

OctoFlow is a solo project looking for active maintainers and contributors, especially in:

- **GPU compute** — kernel optimization, new GPU operations, benchmarking
- **Stdlib modules** — expanding the 246 modules, improving test coverage
- **Platform support** — macOS/Metal backend, ARM/aarch64, Wayland
- **Documentation** — tutorials, guides, API reference improvements
- **Testing** — edge cases, fuzzing, performance regression tests

Open an issue or reach out. All contributions to stdlib, examples, and docs are Apache 2.0.
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
