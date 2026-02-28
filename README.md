# OctoFlow

**Describe it. Build it. Ship it.** The complete vibe coding stack in one binary.

OctoFlow is a GPU-native programming language with built-in AI code generation. Describe what you want in natural language — OctoFlow generates, validates, auto-repairs, and runs the code. Zero external dependencies. 3.3 MB binary. Any Vulkan GPU.

## `octoflow chat` — Describe What You Want

```
$ octoflow chat --allow-net
OctoFlow v1.2 — Chat Mode (type :help for commands)

> Build a program that fetches today's weather for Tokyo and prints the temperature

[Generating...]

let result = web_search("Tokyo weather today temperature")
let page = web_read(result[0].url)
let lines = split(page.text, "\n")
for line in lines
    if contains(line, "°")
        print("{line}")
    end
end

[Running...]
Current: 12°C (54°F), Partly cloudy

> Now make it also show the humidity

[Generating...]
// ... updated code with humidity extraction ...
[Running...]
Current: 12°C (54°F), Partly cloudy
Humidity: 65%
```

Describe what you want in plain English. OctoFlow generates code, validates it,
runs it, and auto-repairs errors (up to 3 attempts). Multi-turn conversation
with 8-message memory.

Works with any OpenAI-compatible API:
```bash
octoflow chat                              # local model (GGUF)
octoflow chat --api http://localhost:8080   # local API server
octoflow chat --allow-net --web-tools      # enable web search during generation
```

## Quickstart

```
$ octoflow run hello.flow
Hello, OctoFlow!

$ octoflow check program.flow
No errors found.

$ octoflow check program.flow --format json
{"code":"E016","message":"expected 'end' to close 'if' block","line":15,"suggestion":"Add 'end' on its own line"}

$ octoflow build main.flow -o bundle.flow
Bundled 5 modules into bundle.flow

$ octoflow repl
OctoFlow v1.2 — GPU-native language (246 stdlib modules)
GPU: NVIDIA GeForce GTX 1660 SUPER
>>> let a = gpu_fill(1.0, 10000000)
>>> let b = gpu_fill(2.0, 10000000)
>>> let c = gpu_add(a, b)
>>> gpu_sum(c)
30000000
```

Download the binary. Unzip. Run. GPU detected automatically.
Works without a GPU too — all GPU operations fall back to CPU.

## What's New in v1.2

### Integer Type

`42` is `int`, `42.0` is `float`. Arithmetic auto-promotes: `int + float = float`.
Integer division is exact: `7 / 2 = 3`.

```flow
let count = 42          // int (i64)
let pi = 3.14           // float (f32)
let result = count + 1  // int
let mixed = count + pi  // float (auto-promoted)
let even = count % 2    // modulo works with both types
```

### `none` Value

First-class null representation. Map lookups return `none` for missing keys.
JSON `null` converts to `none` and back.

```flow
let x = none
if is_none(x)
    print("x is none")
end

let data = json_decode("{\"name\": \"OctoFlow\", \"version\": none}")
let version = map_get(data, "version")  // none
```

### Web Builtins

`web_search(query)` and `web_read(url)` for web-connected programs.
Requires `--allow-net`.

```flow
let results = web_search("OctoFlow GPU language")
for r in results
    print("{r.title}: {r.url}")
end

let page = web_read("https://example.com")
print("{page.title}")
print("{page.text}")
```

### Scoped Permissions

Deno-style fine-grained permission control with path scoping.

```bash
octoflow run app.flow --allow-read=./data --allow-write=./output
octoflow run api.flow --allow-net=api.example.com
octoflow run tool.flow --allow-exec=/usr/bin/git
```

### `octoflow build` — Single-File Bundler

Bundle a multi-file project into one self-contained `.flow` file.

```bash
octoflow build main.flow -o bundle.flow    # bundle everything
octoflow build main.flow --list            # show dependency tree
octoflow run bundle.flow                   # runs anywhere
```

### Structured Errors

69 error codes with JSON output for tooling integration.

```json
{
  "code": "E016",
  "message": "expected 'end' to close 'if' block",
  "line": 15,
  "suggestion": "Add 'end' on its own line to close the block",
  "context": "if x > 10\n    print(\"big\")"
}
```

### CPU Fallback

All GPU operations work on CPU-only machines. Programs that use `gpu_matmul`,
`gpu_add`, etc. run everywhere — with a performance note on startup.

```
$ octoflow run gpu_demo.flow
[note] No GPU detected — using CPU fallback
total: 30000000
```

### VS Code Extension

Syntax highlighting for `.flow` files with 90+ builtin keywords,
string interpolation, code folding, and all operators.

Install from releases: `code --install-extension octoflow-0.1.0.vsix`

## Performance

| Operation (10M elements) | CUDA 12.4 | OctoFlow (Vulkan) | Notes |
|---|---|---|---|
| gpu_add | 0.40 ms | 0.46 ms (deferred) | Batched command buffer |
| gpu_mul | 0.53 ms | 3.27 ms (deferred) | Single fence per chain |
| 5-step pipeline | 2.57 ms | 75 ms | Upload + compute + reduce |
| Install size | ~4 GB SDK | **3.3 MB** binary | Zero dependencies |

Deferred dispatch batches chained GPU operations into a single Vulkan command buffer submission.

## What Makes OctoFlow Different

| | CUDA/OpenCL | GLSL/HLSL/WGSL | OctoFlow |
|---|---|---|---|
| **Primary target** | CPU (with GPU kernels) | GPU (shaders only) | **GPU (general purpose)** |
| **LLM code gen** | No training data | No training data | **51K training pairs, GBNF grammar** |
| **Self-hosting** | No | No | **Yes (69% .flow)** |
| **External deps** | NVIDIA SDK / vendor SDK | Graphics API | **None** |
| **CPU fallback** | No | No | **Yes** |
| **Install** | Multi-GB SDK | Driver-only | **3.3 MB binary** |

## Loom Engine

OctoFlow's GPU compute runtime — weaves thousands of parallel threads into coordinated computation.

```flow
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
// Types: int, float, string, array, map, none
let count = 42
let pi = 3.14
let name = "OctoFlow"
let items = [1, 2, 3]
let config = {"key": "value", "count": 42}
let empty = none

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

// Stream pipelines — GPU-dispatched map/reduce
use filters
stream photo = tap("input.jpg")
stream warm = photo |> filters.brightness(20.0) |> filters.contrast(1.2)
emit(warm, "output.png")
```

Security: `octoflow run server.flow --allow-read=./data --allow-net=api.example.com`

## Standard Library — 246 Modules

| Domain | Modules | Coverage |
|---|---|---|
| **ai** | transformer, inference, generate, sampling, tokenizer | GGUF model loading, LLM inference |
| **collections** | stack, queue, heap, graph | Data structures |
| **compiler** | lexer, eval, parser, preflight, codegen, ir | Self-hosted compiler |
| **crypto** | hash, encoding, random | DJB2, FNV-1a, UUID, CSPRNG |
| **data** | csv, io, json, pipeline, transform, validate | ETL and data processing |
| **db** | core, query, schema | In-memory columnar database |
| **devops** | config, fs, log, process, template | System automation |
| **formats** | gguf | GGUF tensor files (11 quantization types) |
| **gui** | 16 widgets, layout, plot, themes, canvas | Native GUI toolkit |
| **llm** | generate, stream, chat, decompose, sampling | LLM inference (Qwen/LLaMA/Gemma/Phi) |
| **loom** | VM, emitters, runtime, kernels | 40+ GPU compute kernels |
| **media** | BMP, GIF, H.264, AVI, MP4, TTF, WAV | Native codecs (encode + decode) |
| **ml** | nn, regression, classify, cluster, tree, metrics | Machine learning primitives |
| **science** | calculus, physics, signal, matrix, optimize | Scientific computing |
| **stats** | descriptive, distribution, correlation, risk, timeseries | Statistical analysis |
| **string** | string, regex, format | Text processing |
| **sys** | args, env, memory, platform, timer | System interfaces |
| **terminal** | halfblock, kitty, sixel, digits, render | Terminal graphics (4 protocols) |
| **web** | http, server, json_util, url | HTTP client + server |

## Architecture

```
.flow source → Parser → Preflight → Runtime → Loom Engine / Vulkan → GPU
                                        |
                           SPIR-V emitters (written in .flow)
                           40+ pre-built compute kernels
```

```
octoflow chat → LLM → Code Generation → octoflow check → octoflow run → Auto-repair loop
                  |                                                          |
             CONTRACT.md (system prompt)                          Structured error feedback
             GBNF grammar (constrained decoding)                  (up to 3 fix attempts)
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
