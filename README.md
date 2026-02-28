# OctoFlow

OctoFlow is a general-purpose programming language where the GPU is the primary execution target. Data born on the GPU stays on the GPU. The CPU handles I/O — file reads, network, console — and nothing else. Zero external dependencies.

**CPU on demand, not GPU on demand.**

3.7 MB binary. 246 stdlib modules. 1,209 tests.
Any GPU vendor. One file download. No CUDA. No Python. No pip.

## Quickstart

```
$ octoflow --version
OctoFlow 1.3.0

$ octoflow repl
OctoFlow 1.3.0 — GPU-native language (246 stdlib modules)
GPU: NVIDIA GeForce GTX 1660 SUPER
>>> let a = gpu_fill(1.0, 10000000)
>>> let b = gpu_fill(2.0, 10000000)
>>> let c = gpu_add(a, b)
>>> gpu_sum(c)
30000000
```

Download the binary. Unzip. Run. GPU detected automatically.

## Performance

| Operation (10M elements) | CUDA 12.4 | OctoFlow (Vulkan) | Notes |
|---|---|---|---|
| gpu_add | 0.40 ms | 0.46 ms (deferred) | Batched command buffer |
| gpu_mul | 0.53 ms | 3.27 ms (deferred) | Single fence per chain |
| 5-step pipeline | 2.57 ms | 75 ms | Upload + compute + reduce |
| Install size | ~4 GB SDK | **3.7 MB** binary | Zero dependencies |

Deferred dispatch batches chained GPU operations into a single Vulkan command buffer submission. Per-operation overhead drops from ~12ms (synchronous) to ~4ms (batched) as chains grow.

## What Makes OctoFlow Different

| | CUDA/OpenCL | GLSL/HLSL/WGSL | OctoFlow |
|---|---|---|---|
| **Primary target** | CPU (with GPU kernels) | GPU (shaders only) | **GPU (general purpose)** |
| **Built-in LLM** | No | No | **Yes** (local GGUF inference) |
| **External deps** | NVIDIA SDK / vendor SDK | Graphics API | **None** |
| **GPU VM** | N/A | N/A | **Built-in** (5 SSBOs, indirect dispatch) |
| **Install** | Multi-GB SDK | Driver-only | **3.7 MB binary** |

## GPU Virtual Machine

OctoFlow includes a GPU-resident virtual machine with 5 memory regions:

```flow
let vm = loom_boot(1.0, 8.0, 16.0)

// Load data, dispatch compute kernels, read results
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
let _w = loom_write_register(vm, 0.0, 0.0, data)
let pc = [0.0, 3.0, 8.0]
let _d = loom_dispatch(vm, "stdlib/gpu/kernels/loom_scale.spv", pc, 1.0)

let prog = loom_build(vm)
let _e = loom_execute(prog)

let result = loom_read_register(vm, 0.0, 0.0, 8.0)
// result = [3, 6, 9, 12, 15, 18, 21, 24]
```

- **HOST_VISIBLE polling**: CPU reads GPU status in ~1us (zero-copy)
- **Dormant VMs**: Over-provisioned command buffers with indirect dispatch
- **I/O streaming**: CPU feeds data batches, GPU processes with reusable command buffers
- **Homeostasis**: GPU self-regulates via maxnorm + regulator kernels
- **102 compute kernels**: Scale, affine, matvec, reduce, WHERE, delta encode/decode, dictionary lookup

**LoomDB** — GPU-resident computed-data log. Captures pipeline results without I/O. Similarity search (cosine, dot, euclidean) runs entirely in VRAM. Async persistence to OctoDB — your main compute pipeline never touches disk.

**OctoDB** — Embedded database with CRUD, multi-condition queries, indexing, `.odb` persistence. Serves as LoomDB's cold storage tier.

> [Loom Engine Guide](docs/loom-engine.md) — chain dispatch, LoomDB, OctoDB, API reference

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
  let mut a = 0.0
  let mut b = 1.0
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

Deno-inspired security: `octoflow run server.flow --allow-read --allow-net`

## Standard Library — 246 Modules

| Domain | Modules | Coverage |
|---|---|---|
| **ai** | transformer, inference, generate, weight_loader | GGUF model loading, tokenization |
| **collections** | stack, queue, heap, graph | Data structures |
| **compiler** | lexer, eval, parser, preflight, codegen, ir | Compiler modules (written in .flow) |
| **crypto** | hash, encoding, random | SHA-256, base64, CSPRNG |
| **data** | csv, io, pipeline, transform, validate | ETL and data processing |
| **db** | core, engine, vector, persist | OctoDB (CRUD, indexing, .odb) + LoomDB (GPU-resident, vector search) |
| **devops** | config, fs, log, process, template | System automation |
| **formats** | gguf, json | GGUF tensor files, JSON |
| **gpu** | VM, emitters, runtime, kernels | 102 GPU compute kernels |
| **gui** | widgets, layout, canvas, plot, themes, buffer_view | 16 widget types, 3 layouts, 5 chart types, canvas drawing (Windows) |
| **llm** | generate, stream, chat, decompose | LLM inference (Qwen3-1.7B) |
| **media** | image (PNG/JPEG/GIF/BMP), audio (WAV), video (MP4 stills) | Native codecs |
| **ml** | nn, regression, classify, cluster, tree, linalg | Machine learning primitives |
| **science** | calculus, physics, signal, matrix, optimize | Scientific computing |
| **stats** | descriptive, distribution, correlation, risk | Statistical analysis |
| **string** | string, regex, format | Text processing |
| **sys** | args, env, memory, platform, timer | System interfaces |
| **terminal** | term_image, colors | Kitty/Sixel/halfblock graphics |
| **web** | http, json_util, url | HTTP client, JSON, URLs |

## Documentation

- [Quickstart](docs/quickstart.md) — five minutes from download to GPU compute
- [Chat Mode](docs/chat.md) — AI code generation from natural language
- [MCP Server](docs/mcp.md) — connect to Claude Desktop, Cursor, VS Code
- [Permissions](docs/permissions.md) — Deno-style security model
- [Installation](docs/installation.md) — Windows, Linux, macOS setup
- [Feature Status](docs/features.md) — what's stable, beta, planned
- [All docs](docs/README.md)

## Architecture

```
.flow source -> Parser -> Preflight -> Compiler -> GPU VM / Vulkan Dispatch -> GPU
                                          |
                              SPIR-V emitters (written in .flow)
                              102 pre-built compute kernels
```

Four crates. Zero external Rust dependencies. Only system libraries (vulkan-1, ws2_32).

~252K lines total: ~72K Rust + ~180K .flow (stdlib + examples). Zero external dependencies.

## License

MIT — see [LICENSE](LICENSE) for details. Your .flow programs are yours entirely.
