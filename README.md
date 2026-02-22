# OctoFlow

![License: Compiler Binary - Free Use](https://img.shields.io/badge/compiler-free%20to%20use-blue)
![License: Stdlib - Apache 2.0](https://img.shields.io/badge/stdlib-Apache%202.0-green)
![Tests: 658 passing](https://img.shields.io/badge/tests-1200%20passing-brightgreen)
![Dependencies: 0](https://img.shields.io/badge/dependencies-0-brightgreen)
![Binary: 2.8 MB](https://img.shields.io/badge/binary-2.3%20MB-blue)

A GPU-native general-purpose programming language.
Designed for the GPU from scratch. Compiled by itself. Run on any GPU.

**CPU on demand, not GPU on demand.**

2.8 MB binary. Zero dependencies. Any GPU vendor. One file download.

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

## Zero Dependencies

No external libraries. No package manager. No runtime.

The compiler is a single static binary — zero Rust crates, zero C libraries, zero transitive dependencies.
System links: Vulkan driver only. Binary size: 2.8 MB.

No supply chain risk. Nothing to audit except the binary itself.

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

## Standard Library — 246 Modules

| Domain | Modules | Coverage |
|---|---|---|
| **ai** | transformer, inference, generate, weight_loader | GGUF model loading, tokenization |
| **collections** | stack, queue, heap, graph | Data structures |
| **compiler** | lexer, eval, parser, preflight, codegen, ir | Self-hosted compiler |
| **crypto** | hash, encoding, random | SHA-256, base64, CSPRNG |
| **data** | csv, io, pipeline, transform, validate | ETL and data processing |
| **db** | core, query, schema | Database abstractions |
| **formats** | gguf, json | GGUF tensor files, JSON |
| **gpu** | VM, emitters, runtime, kernels | 73+ GPU compute kernels |
| **gui** | widgets, layout, themes, events | Native GUI toolkit |
| **llm** | generate, stream, chat, decompose | LLM inference (Qwen2.5) |
| **media** | image (PNG/JPEG/GIF/BMP), video (AVI/MP4/H.264), audio (WAV) | Native codecs |
| **ml** | nn, regression, classify, cluster, tree, linalg | Machine learning primitives |
| **science** | calculus, physics, signal, matrix, optimize | Scientific computing |
| **stats** | descriptive, distribution, correlation, risk | Statistical analysis |
| **string** | string, regex, format | Text processing |
| **sys** | args, env, memory, platform, timer | System interfaces |
| **terminal** | term_image, colors | Kitty/Sixel/halfblock graphics |
| **web** | http, json_util, url | HTTP client, JSON, URLs |

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

## GPU VM

OctoFlow includes a GPU-native virtual machine where the GPU runs autonomously and the CPU acts only as a BIOS.

```octoflow
// Boot a GPU VM with 4 instances
let vm = vm_boot()
let prog = vm_program(vm, kernels, 4)

// Write input, execute entire dispatch chain in one GPU submit
vm_write_register(vm, 0, 0, input_data)
vm_execute(prog)

// Read output — everything between write and read happened on GPU
let result = vm_read_register(vm, 3, 30)
vm_shutdown(vm)
```

VM instances communicate via register-based message passing. The dispatch chain runs as a single `vkQueueSubmit` — no CPU round-trips between stages. Supports LLM inference, database queries, compression, and multi-agent workloads on the same substrate.

See [GPU VM Guide](docs/gpu-vm-guide.md) for details.

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

See [examples/](examples/) for all 19 runnable demos.

## Documentation

- [Quickstart](docs/quickstart.md) — install to GPU in 5 minutes
- [Language Guide](docs/language-guide.md) — full syntax reference
- [Builtins](docs/builtins.md) — all built-in functions
- [GPU Guide](docs/gpu-guide.md) — GPU computing
- [REPL](docs/repl.md) — interactive mode
### LLM / AI-Assisted Development

The [Language Guide](docs/language-guide.md) is designed as LLM context. Feed it to your AI assistant (Claude, ChatGPT, Copilot) and it can write, debug, and explain OctoFlow code — no training required. Use it as RAG, system prompt, or project context.

## License

- **Standard library, examples, docs**: Apache 2.0 ([LICENSE-STDLIB](LICENSE-STDLIB))
- **Compiler binary**: Free to download and use for any purpose
- **Your .flow programs**: Yours entirely

See [LICENSE](LICENSE) for full terms.

## Transparency

### How It's Built

OctoFlow is vibe-coded. The architecture, design decisions, and system thinking are
human-driven. The implementation is AI-assisted — built in collaboration with LLMs
(primarily Claude), which is how a solo developer ships 121K lines of code across a
compiler, GPU runtime, and 246 stdlib modules.

Stated openly because honesty matters more than optics. Every line of code is reviewed,
tested (1,200+ tests passing), and understood. The AI accelerates implementation; it
doesn't replace understanding.

### Rust as Bootstrapper

OctoFlow is 69% self-hosted. The compiler — lexer, parser, preflight, evaluator, SPIR-V
codegen — is written in .flow files. The remaining 31% is a first-party Rust runtime that
provides:

- **Execution engine** that loads and runs .flow files
- **Vulkan FFI** — direct calls to vulkan-1 (no wrapper crates)
- **OS interfaces** — file I/O, windowing (Win32/X11), network sockets, process execution

The Rust layer is the bootstrapper and OS boundary. It is not a framework or library — it
is ~37K lines of first-party Rust with **zero external crates**. `Cargo.lock` contains
only workspace-internal packages. System links at runtime: `vulkan-1` (GPU driver),
`ws2_32`/`kernel32`/`user32`/`gdi32` (Windows OS). On Linux: `libvulkan.so` + libc.

"Zero dependencies" means zero third-party project dependencies. The binary does require
a Vulkan GPU driver installed on the system.

### Binary Size

The compiled binary is **2.8 MB**. Previous documentation stated 2.2-2.3 MB, which
reflected earlier builds before the GPU VM, media codecs, and expanded stdlib were added.
The current size is 2.8 MB — still a single file, still no installer, still no runtime.

For context: Python is ~30 MB, Node.js is ~40 MB, Go is ~150 MB (with stdlib),
Rust's compiler is ~200+ MB. OctoFlow ships a compiler, GPU runtime, VM, 246 stdlib
modules, media codecs, and a GUI toolkit in 2.8 MB.

### GPU Kernel Count

The "73+ GPU kernels" refers to built-in GPU operations available in the language —
functions like `gpu_add`, `gpu_mul`, `gpu_sin`, `gpu_matmul`, `gpu_scatter`, `gpu_gather`,
the GPU VM dispatch kernels (`vm_dequant_q4k`, `vm_where_gt`, `vm_delta_encode`,
`vm_delta_decode`, `vm_dict_lookup`), LLM inference kernels (11 SPIR-V), and internal
kernels (copy_register, regulator). Each compiles to SPIR-V at runtime via the `ir.flow`
emitter pipeline. They are not pre-compiled .spv files — they are generated on demand.

### Looking for Maintainers

OctoFlow is a solo project looking for active maintainers and contributors, especially in:

- **GPU compute** — kernel optimization, new GPU operations, benchmarking
- **Stdlib modules** — expanding the 246 modules, improving test coverage
- **Platform support** — macOS/Metal backend, ARM/aarch64, Wayland
- **Documentation** — tutorials, guides, API reference improvements
- **Testing** — edge cases, fuzzing, performance regression tests

Open an issue or reach out. All contributions to stdlib, examples, and docs are Apache 2.0.
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
