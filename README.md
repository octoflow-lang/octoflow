# OctoFlow

**Describe it. Build it. Ship it.** The complete vibe coding stack in one binary.

OctoFlow is a GPU-native programming language with built-in AI code generation. Describe what you want in natural language — OctoFlow generates, validates, auto-repairs, and runs the code. Zero external dependencies. 2.8 MB binary. Any Vulkan GPU.

```
$ octoflow chat --model qwen2.5-coder-1.5b.gguf
> Build a program that counts primes below one million using the GPU

Generating...
✓ Syntax valid
✓ Running...

Primes below 1000000: 78498
```

## Quickstart

```
$ octoflow --version
OctoFlow 1.2.0

$ octoflow repl
OctoFlow 1.2.0 — GPU-native language (246 stdlib modules)
GPU: NVIDIA GeForce GTX 1660 SUPER
>>> let a = gpu_fill(1.0, 10000000)
>>> let b = gpu_fill(2.0, 10000000)
>>> let c = gpu_add(a, b)
>>> gpu_sum(c)
30000000
```

Download the binary. Unzip. Run. GPU detected automatically. Works without GPU too — all operations fall back to CPU.

## Features

### AI Code Generation (`octoflow chat`)

Built-in LLM inference with GGUF models. Describe what you want, get working code.

```
$ octoflow chat --model model.gguf
> Create a dashboard that reads sales.csv and shows total revenue per region

[Generating → Validating → Running]
```

- Auto-repair: structured errors fed back to LLM, up to 3 fix attempts
- ReAct tool-use: LLM searches the web and reads pages for context (`--web-tools`)
- Streaming output, 8-message conversation memory, multiline input
- Works with any GGUF model (Qwen2.5-Coder, Llama, etc.)

### Web Builtins

```flow
let results = web_search("OctoFlow GPU language")
for r in results
    print("{r.title}: {r.url}")
end

let page = web_read("https://example.com")
print("{page.text}")
```

Gated on `--allow-net`. DuckDuckGo HTML search + page content extraction.

### Integer Type

```flow
let count = 42         // int (i64)
let pi = 3.14          // float (f32)
let result = count + 1 // int + int = int
let mixed = count + pi // int + float = float (auto-promotion)
let even = x % 2 == 0  // modulo operator
```

### Scoped Permissions

Deno-inspired security with path and host scoping:

```bash
octoflow run app.flow --allow-read=./data --allow-write=./output --allow-net=api.example.com
```

### Single-File Bundler (`octoflow build`)

Bundle multi-file projects into one distributable `.flow` file:

```bash
octoflow build main.flow -o bundle.flow
octoflow build main.flow --list   # show dependency tree
```

Recursive import tracing, topological sort, circular import detection.

### Structured Errors (`--format json`)

Machine-readable error output for tooling and LLM integration:

```bash
octoflow check program.flow --format json
```

```json
{"code":"E016","message":"expected 'end' to close 'if' block","line":15,"suggestion":"Add 'end' on its own line"}
```

69 error codes with per-code fix suggestions.

### CPU Fallback

All GPU operations work on machines without a GPU. Matrix multiply, reductions, map ops, sort — everything runs on CPU when no Vulkan device is detected. A startup note tells you which mode you're in.

### VS Code Extension

Syntax highlighting for `.flow` files with ~90 builtins, string interpolation, keywords, operators, and code folding:

```bash
code --install-extension vscode/octoflow-0.1.0.vsix
```

### Project Scaffolding (`octoflow new`)

```bash
octoflow new dashboard my-project    # 7 built-in templates
octoflow new api my-service
octoflow new gpu-compute my-pipeline
```

## Performance

| Operation (10M elements) | CUDA 12.4 | OctoFlow (Vulkan) | Notes |
|---|---|---|---|
| gpu_add | 0.40 ms | 0.46 ms (deferred) | Batched command buffer |
| gpu_mul | 0.53 ms | 3.27 ms (deferred) | Single fence per chain |
| 5-step pipeline | 2.57 ms | 75 ms | Upload + compute + reduce |
| Install size | ~4 GB SDK | **2.8 MB** binary | Zero dependencies |

Deferred dispatch batches chained GPU operations into a single Vulkan command buffer submission.

## What Makes OctoFlow Different

| | CUDA/OpenCL | GLSL/HLSL/WGSL | OctoFlow |
|---|---|---|---|
| **Primary target** | CPU (with GPU kernels) | GPU (shaders only) | **GPU (general purpose)** |
| **AI code gen** | No | No | **Built-in** |
| **Self-hosting** | No | No | **Yes (69% .flow)** |
| **External deps** | NVIDIA SDK / vendor SDK | Graphics API | **None** |
| **Install** | Multi-GB SDK | Driver-only | **2.8 MB binary** |

## Loom Engine

OctoFlow's GPU compute runtime — weaves thousands of parallel threads into coordinated computation.

```flow
// Boot a compute unit, upload data, dispatch kernels
let unit = loom_boot(1.0, 8194, 4096)
loom_write(unit, 0.0, data)

loom_dispatch(unit, "stdlib/loom/kernels/sieve_init.spv", params, 32.0)
loom_dispatch(unit, "stdlib/loom/kernels/sieve_mark.spv", params, 32.0)
loom_dispatch(unit, "stdlib/loom/kernels/sieve_count.spv", params, 32.0)

let prog = loom_build(unit)
loom_launch(prog)

while loom_poll(prog) < 0.5
end
let result = loom_read(unit, 0.0, 0.0, 8194)
```

- **40+ compute kernels**: sieve, reduce, affine, matvec, WHERE, delta encode/decode
- **Async execution**: `loom_launch` + `loom_poll` for non-blocking GPU work
- **JIT kernels**: `loom_dispatch_jit` compiles SPIR-V from IR at runtime
- **HOST_VISIBLE polling**: CPU reads GPU status in ~1us (zero-copy)

## Language

```flow
// GPU pipeline — data stays in VRAM between operations
let prices = gpu_fill(100.0, 1000000)
let returns = gpu_scale(prices, 0.02)
let signal = gpu_add(returns, prices)
let total = gpu_sum(signal)
print("total: {total}")

// Imperative — full control flow
fn fibonacci(n)
  let mut a = 0
  let mut b = 1
  for i in range(0, n)
    let tmp = b
    b = a + b
    a = tmp
  end
  return a
end

// String interpolation, maps, closures
let user = { name: "Alice", score: 95 }
print("Hello {user.name}, score: {user.score}")
```

## Standard Library — 246 Modules

| Domain | Highlights |
|---|---|
| **ai** | Transformer inference, GGUF model loading, tokenization |
| **collections** | Stack, queue, heap, graph |
| **compiler** | Self-hosted lexer, parser, eval, codegen, IR |
| **crypto** | SHA-256, base64, CSPRNG |
| **data** | CSV, pipeline, transform, validate |
| **db** | Columnar storage, query, schema |
| **devops** | Config, filesystem, logging, process |
| **formats** | GGUF tensor files, JSON |
| **gui** | Widgets, layout, themes, events |
| **llm** | LLM inference, streaming, chat, decompose |
| **loom** | 40+ GPU compute kernels, IR builder, patterns |
| **media** | PNG, JPEG, GIF, BMP, AVI, MP4, H.264, WAV |
| **ml** | Neural nets, regression, classification, clustering |
| **science** | Calculus, physics, signal processing, optimization |
| **stats** | Descriptive, distributions, correlation, risk |
| **string** | Regex, formatting, text processing |
| **sys** | Args, env, memory, platform, timer |
| **web** | HTTP client, web search, web read, JSON, URLs |

## Architecture

```
.flow source → Parser → Preflight → Runtime → Loom Engine / Vulkan → GPU
                                        |
                           SPIR-V emitters (written in .flow)
                           40+ pre-built compute kernels
```

Three crates. Zero external Rust dependencies. Only system libraries (vulkan-1, ws2_32).

## Tests

**1,017+ tests**, all passing. Zero failures.

```
Rust unit tests:       854 (compiler, parser, Vulkan, OctoView)
.flow stdlib tests:    163 (GPU, media, data, formats, AI/LLM)
```

## License

- **Compiler**: Proprietary — free binary download
- **Standard library, docs, examples**: Apache 2.0
- **Your .flow programs**: Yours entirely
