# OctoFlow

The languages at the frontier of computing — C (1972), C++ (1979), Python (1991) — were designed when the CPU was the only processor. They still work. But accessing a GPU, which is 10–100x faster for parallel workloads, requires CUDA (NVIDIA-only, 4 GB SDK), OpenCL (effectively abandoned), or shader languages designed for graphics, not general compute.

Every computer sold today has a GPU. Most of it sits idle.

OctoFlow is built from scratch for this reality. The GPU is the primary execution target. Data born on VRAM stays on VRAM. The CPU handles I/O — files, network, console — and nothing else. Single binary, any GPU vendor, zero dependencies.

**CPU on demand, not GPU on demand.**

> **On building this.** OctoFlow is AI-assisted — every architectural decision is human, LLMs generate the bulk of the code. Rust at the OS boundary, Vulkan for cross-vendor GPU, the Loom Engine's main/support split, JIT kernel emission via IR builder — all human decisions. AI writes the code; a human decides what code to write and why.

**Works on any GPU.** NVIDIA, AMD, Intel — no CUDA required, no vendor SDK, no driver headaches. Just Vulkan.

---

4.1 MB binary. 423 stdlib modules. 1,394 tests. 113 GPU kernels.

---

## Quickstart

```
$ octoflow --version
OctoFlow 1.4.0

$ octoflow repl
OctoFlow 1.4.0 — GPU-native language (423 stdlib modules)
GPU: NVIDIA GeForce GTX 1660 SUPER
>>> let a = gpu_fill(1.0, 10000000)
>>> let b = gpu_fill(2.0, 10000000)
>>> let c = gpu_add(a, b)
>>> gpu_sum(c)
30000000
```

Download the binary. Unzip. Run. GPU detected automatically.

## What Makes OctoFlow Different

| | CUDA/OpenCL | GLSL/HLSL/WGSL | OctoFlow |
|---|---|---|---|
| **Primary target** | CPU (with GPU kernels) | GPU (shaders only) | **GPU (general purpose)** |
| **Built-in LLM** | No | No | **Yes** (local GGUF inference) |
| **External deps** | NVIDIA SDK / vendor SDK | Graphics API | **None** |
| **GPU VM** | N/A | N/A | **Built-in** (Loom Engine) |
| **Install** | Multi-GB SDK | Driver-only | **4.1 MB binary** |

## Performance

| Operation (10M elements) | CUDA 12.4 | OctoFlow (Vulkan) | Notes |
|---|---|---|---|
| gpu_add | 0.40 ms | 0.46 ms (deferred) | Batched command buffer |
| gpu_mul | 0.53 ms | 3.27 ms (deferred) | Single fence per chain |
| 5-step pipeline | 2.57 ms | 75 ms | Upload + compute + reduce |
| Install size | ~4 GB SDK | **4.1 MB** binary | Zero dependencies |

Deferred dispatch batches chained GPU operations into a single Vulkan command buffer submission. Per-operation overhead drops from ~12ms (synchronous) to ~4ms (batched) as chains grow.

## GPU Virtual Machine — Loom Engine

OctoFlow includes a GPU-resident virtual machine called the Loom Engine:

```flow
let vm = loom_boot(1.0, 0.0, 16.0)

// Load data, dispatch compute kernel, read results
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
loom_write(vm, 0.0, data)
loom_dispatch(vm, "stdlib/loom/kernels/ops/loom_scale.spv", [0.0, 3.0, 8.0], 1.0)

let prog = loom_build(vm)
loom_run(prog)

let result = loom_read_globals(vm, 0.0, 8.0)
// result = [3, 6, 9, 12, 15, 18, 21, 24]

loom_free(prog)
loom_shutdown(vm)
```

- **Main Loom** = GPU-only compute. Receives dispatches. Never initiates I/O.
- **Support Loom** = CPU–GPU I/O bridge. Owns boot, state, double buffer, presentation.
- **Park/unpark**: Suspend and resume GPU VMs with zero reallocation.
- **JIT kernels**: Emit SPIR-V at runtime via the IR builder (80+ ops).
- **Homeostasis**: GPU self-regulates dispatch pacing via timing feedback.
- **113 compute kernels**: Scale, affine, matvec, reduce, sort, compress, ML ops.

> [Loom Engine Guide](docs/loom-engine.md) — architecture, dispatch chains, API reference

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

## Standard Library — 423 Modules

| Domain | Modules | Highlights |
|---|---|---|
| **loom** | GPU VM, JIT kernels, OctoPress compression, ASE | Loom Engine runtime + IR builder |
| **llm** | GGUF loader, transformer, tokenizer, sampling | Local LLM inference on GPU |
| **game** | ECS, sprite, physics, collision, AI, scene | Parallel-array game engine |
| **gui** | Widgets, layout, canvas, chart, themes | 16 widget types, canvas drawing |
| **media** | Audio DSP, image, video, codecs, timeline | WAV/BMP/GIF/H.264/MP4/AVI/TTF |
| **compiler** | Lexer, parser, eval, preflight, codegen, IR | Self-hosted compiler (written in .flow) |
| **collections** | Stack, queue, heap, graph, trie, skip list | Data structures |
| **math** | Matrix, vector, complex, probability, noise | Scientific computing |
| **algo** | Sort, search, pathfinding, geometry | Algorithms + A* pathfinding |
| **ml** | Neural nets, regression, clustering, GPU ML | Machine learning primitives |
| **viz** | LoomView renderer, data fingerprinting | GPU visualization toolkit |
| **ai** | Transformer blocks, weight loading | AI model components |
| **data** | CSV, I/O, pipeline, transform, validate | ETL and data processing |
| **db** | CRUD, indexing, vector search, persistence | Embedded database + Loom State |
| **search** | OctoSearch — GPU-first full-text search | BM25 scoring on GPU |
| **stats** | Descriptive, distribution, correlation, risk | Statistical analysis |
| **science** | Calculus, physics, signal, optimize | Scientific computing |
| **agent** | Tool use, planning, memory | AI agent framework |
| **string** | String ops, regex, formatting | Text processing |
| **devops** | Config, filesystem, logging, templates | System automation |
| **crypto** | Hashing, encoding, random, UUID | Cryptographic primitives |
| **web** | HTTP client, JSON utilities, URL parsing | Web stack |
| **sys** | Args, env, memory, platform, timer | System interfaces |
| **terminal** | Kitty/Sixel/halfblock image rendering | Terminal graphics |
| **formats** | GGUF, JSON parsers | File format support |

## Documentation

- [Quickstart](docs/quickstart.md) — download to GPU compute in five minutes
- [Installation](docs/installation.md) — Windows, Linux, macOS setup
- [Language Guide](docs/language-guide.md) — complete language reference
- [Builtins](docs/builtins.md) — all built-in functions
- [GPU Guide](docs/gpu-guide.md) — GPU compute operations
- [GPU Recipes](docs/gpu-recipes.md) — common GPU patterns
- [Loom Engine](docs/loom-engine.md) — GPU VM architecture and API
- [Loom Use Cases](docs/loom-engine-use-cases.md) — real-world Loom patterns
- [GPU Sieve](docs/gpu-sieve.md) — prime sieve benchmark walkthrough
- [GPU Benchmarks](docs/benchmark-gpu.md) — performance measurements
- [GUI Toolkit](docs/gui.md) — widget system and canvas drawing
- [Chat Mode](docs/chat.md) — AI code generation from natural language
- [MCP Server](docs/mcp.md) — connect to Claude Desktop, Cursor, VS Code
- [Streams](docs/streams.md) — stream processing pipelines
- [Permissions](docs/permissions.md) — Deno-style security model
- [REPL](docs/repl.md) — interactive GPU computing
- [Features](docs/features.md) — what's stable, beta, planned
- [Web Builtins](docs/web-builtins.md) — HTTP and web functions
- [Stdlib Reference](docs/stdlib.md) — full standard library documentation
- [Vibe Coding](docs/vibe-coding.md) — AI-assisted development with OctoFlow
- [Roadmap](docs/roadmap.md) — what's next
- [All docs](docs/README.md)

## Architecture

```
.flow source -> Parser -> Preflight -> Compiler -> GPU VM / Vulkan Dispatch -> GPU
                                          |
                              SPIR-V emitters (written in .flow)
                              113 pre-built compute kernels
```

Zero external dependencies. Only system libraries (vulkan-1, ws2_32).

## Supported GPUs

Any GPU with Vulkan 1.0+ support:

- **NVIDIA** — GeForce, Quadro, Tesla (all modern cards)
- **AMD** — Radeon, Radeon Pro, Instinct
- **Intel** — Arc, integrated (Gen 9+)
- **Apple** — via MoltenVK

No CUDA required. No vendor lock-in. No SDK to install.

## Contributing

Contributions are welcome:

- **stdlib modules** — `.flow` files in `stdlib/`
- **examples** — `.flow` files in `examples/`
- **documentation** — `docs/`
- **bug reports and feature requests** — [open an issue](https://github.com/octoflow-lang/octoflow/issues)

If you're interested in GPU computing, language design, or runtime engineering — [get in touch](https://github.com/octoflow-lang/octoflow/issues).

## License

MIT — see [LICENSE](LICENSE) for details. Your `.flow` programs are yours entirely.
