<p align="center">
  <img src="assets/logo.png" width="140" alt="OctoFlow">
</p>

<h1 align="center">OctoFlow</h1>

<p align="center">
  <strong>A GPU-native programming language.</strong><br>
  4.5 MB binary. Zero dependencies. Any GPU vendor. One file download.
</p>

<p align="center">
  <a href="https://github.com/octoflow-lang/octoflow/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/octoflow-lang/octoflow?color=1a9a9a&style=flat-square"></a>
  <a href="https://github.com/octoflow-lang/octoflow/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/octoflow-lang/octoflow?color=1a9a9a&style=flat-square"></a>
  <a href="https://octoflow-lang.github.io/octoflow/"><img alt="Website" src="https://img.shields.io/badge/website-octoflow--lang.github.io-1a9a9a?style=flat-square"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#what-it-looks-like">Code Examples</a> &bull;
  <a href="#the-loom-engine">Loom Engine</a> &bull;
  <a href="#how-this-was-built">How This Was Built</a> &bull;
  <a href="#looking-for-maintainers">Looking for Maintainers</a>
</p>

---

## What is OctoFlow?

OctoFlow is a general-purpose programming language where the GPU is the primary execution target. Not a wrapper around CUDA. Not a shader language. A complete language — with functions, structs, modules, streams, error handling — that happens to run compute on the GPU by default.

```
let a = gpu_fill(1.0, 10000000)
let b = gpu_fill(2.0, 10000000)
let c = gpu_add(a, b)
print("Sum: {gpu_sum(c)}")           // 30000000 — computed on GPU
```

No SDK. No driver toolkit. No package manager. Download one binary, run it.

### At a glance

| | |
|---|---|
| **Binary size** | 4.5 MB (single file, all platforms) |
| **Dependencies** | Zero. Hand-rolled Vulkan bindings, nothing external |
| **GPU support** | NVIDIA, AMD, Intel — anything with Vulkan |
| **Stdlib** | 445 modules across 28 domains |
| **GPU kernels** | 150 pre-compiled SPIR-V shaders, embedded in binary |
| **Tests** | 966 passing |
| **License** | MIT (stdlib + everything in this repo) |

---

## Quickstart

### Install

**Windows** (PowerShell):
```powershell
irm https://octoflow-lang.github.io/octoflow/install.ps1 | iex
```

**Linux / macOS** (bash):
```bash
curl -fsSL https://octoflow-lang.github.io/octoflow/install.sh | sh
```

Or download directly from [Releases](https://github.com/octoflow-lang/octoflow/releases/latest).

### Run

```bash
octoflow run hello.flow          # run a program
octoflow repl                    # interactive REPL
octoflow chat                    # AI-assisted code generation
octoflow check file.flow         # static analysis
```

---

## What It Looks Like

### GPU compute in 5 lines

```
let a = gpu_fill(1.0, 1000000)
let b = gpu_fill(2.0, 1000000)
let c = gpu_add(a, b)
let d = gpu_scale(c, 0.5)
print("Total: {gpu_sum(d)}")       // 1500000
```

Data born on the GPU stays on the GPU. No round-trips until you need the result.

### Functional programming

```
let nums = [1, 2, 3, 4, 5, 6, 7, 8]
let evens = filter(nums, fn(x) x % 2 == 0 end)
let squared = map_each(evens, fn(x) x * x end)
let total = reduce(squared, 0, fn(acc, x) acc + x end)
print("Sum of even squares: {total}")   // 120
```

### Stream pipelines

```
stream photo = tap("input.jpg")
stream enhanced = photo
    |> brightness(20)
    |> contrast(1.2)
    |> saturate(1.1)
emit(enhanced, "output.png")
```

### Data analysis

```
use csv
use descriptive

let data = read_csv("sales.csv")
let revenue = csv_column(data, "revenue")

print("Mean:   {mean(revenue)}")
print("Median: {median(revenue)}")
print("P95:    {quantile(revenue, 0.95)}")
```

### Error handling

```
let result = try(read_file("data.txt"))
if result.ok
  print("Read {len(result.value)} chars")
else
  print("Error: " + result.error)
end
```

Every error returns a structured code (E001-E099) with a human-readable fix action.

---

## The Loom Engine

The Loom Engine is what makes OctoFlow different from "GPU library with a scripting layer."

**The idea:** Queue an entire dispatch chain — hundreds or thousands of GPU kernels — into a single `vkQueueSubmit`. The GPU executes the full pipeline autonomously. Zero CPU interruption.

```
let vm = loom_boot(1, 0, 16)
loom_write(vm, 0, data)
loom_dispatch(vm, "kernel.spv", [0, 3, 8], 1)
let prog = loom_build(vm)
loom_run(prog)
let result = loom_read_globals(vm, 0, 8)
loom_free(prog)
loom_shutdown(vm)
```

Or use the express API:

```
let result = loom_compute("kernel.spv", data, 1024)
```

**Three tiers of GPU access:**
- **Tier 1** — One-call ops: `gpu_fill`, `gpu_add`, `gpu_sum`, `gpu_matmul` (simple, immediate)
- **Tier 2** — Dispatch chains: `loom_boot` → `loom_dispatch` → `loom_build` → `loom_run` (custom pipelines)
- **Tier 3** — JIT SPIR-V: `ir_begin` → `ir_entry` → ... → `ir_finalize` (generate kernels at runtime)

---

## Standard Library

445 modules. All written in OctoFlow itself. All MIT-licensed and in this repo.

| Domain | What's in it |
|--------|-------------|
| **AI & LLM** | Transformer inference, GGUF loader, BPE tokenizer, streaming generation |
| **GPU** | 150 kernels, Loom Engine, SPIR-V codegen, dispatch chains, resident buffers |
| **Media** | Audio DSP, image transforms, video timeline, WAV/BMP/GIF/H.264/MP4/AVI/TTF codecs |
| **ML** | Regression, classification, clustering, neural networks, decision trees, ensembles |
| **Statistics** | Descriptive stats, distributions, hypothesis testing, time series, risk metrics |
| **Science** | Linear algebra, calculus, physics, signal processing, interpolation, optimization |
| **Data** | CSV, JSON, pipelines, validation, transforms |
| **Web** | HTTP client/server, URL parsing |
| **GUI** | Canvas, widgets, layout, ECS, theming, physics2d |
| **DB** | In-memory columnar database with query engine |
| **Crypto** | Hashing, encoding, base64, hex |
| **System** | File I/O, environment, datetime, platform detection, process control |

---

## How This Was Built

OctoFlow is **AI-assisted** from the beginning. LLMs generated the bulk of the code. This is not a secret and not a caveat. It's the point.

But "AI-assisted" does not mean "unreviewed." Every architectural decision has a human at the gate:

- **Rust at the OS boundary, .flow for everything else** — human decision
- **Pure Vulkan, no vendor SDK** — human decision
- **Zero external dependencies** — human decision
- **Loom Engine's dispatch chain model** — human decision
- **23-concept language spec that fits in an LLM prompt** — human decision
- **JIT SPIR-V emission via IR builder** — human decision
- **Self-hosting compiler direction** — human decision

The AI writes code. The human decides what to build, why, and whether it ships.

### The philosophy behind this

Two principles guide every decision:

**Sustainability** — Can this trajectory continue? Is this adding complexity faster than it can be maintained? Is the test count rising? Is the gotcha list shrinking? If the answer to any of these is "no," the developer stops and fixes before shipping more.

**Empowerment** — Does this increase the user's capacity? Can a non-GPU-programmer go from intent to working GPU code? Does the LLM need *less* help generating correct OctoFlow over time? If a feature makes the language harder to learn or harder for AI to generate, it doesn't ship.

These aren't marketing. They're the actual decision framework. Every feature, every refactor, every new builtin gets scored against them. Better to ship less and ship right.

---

## Project Status

OctoFlow is **real, working software** — not a concept or prototype. The compiler runs, the GPU dispatches, the tests pass, the demos are live. You can download it right now and run GPU compute.

That said, it's honest to say:

- **v1.5.8** — actively developed, not yet battle-tested by a community
- **Solo developer** — one person plus AI tools, which is both the strength (fast iteration, coherent vision) and the limitation (bus factor of 1)
- **Compiler is private** — the stdlib, examples, docs, and everything in this repo are MIT. The compiler Rust source is in a private repo for now. See below.

### What works well today

- GPU compute via Tier 1 (one-call ops) and Tier 2 (Loom dispatch chains)
- 445-module stdlib covering AI, ML, media, science, data, GUI, and more
- Interactive REPL with GPU support
- AI-assisted code generation via `octoflow chat`
- Sandboxed execution with granular permission flags
- Cross-vendor GPU support (NVIDIA tested, AMD/Intel via Vulkan)

### What's in progress

- **LLM inference on consumer GPUs** — Running 24GB models on 6GB GPUs via layer streaming. This is the current focus.
- OctoPress weight compression (3-tier hot/warm/cold cache)
- AMD GPU validation
- Tier 3 JIT SPIR-V stabilization

---

## Looking for Maintainers

This project needs more humans.

One developer built it to prove the idea works. The idea works. Now it needs people who want to take it further — not just contributors, but **co-maintainers** who want ownership of parts of the system.

### What's on the table

- **Full open source.** The developer is willing to open-source the entire compiler (Rust source, all 3 modules, the full 966-test suite) once there's a team to develop and sustain it. MIT license, same as everything else.
- **Compiler access now.** Serious maintainer candidates get private repo access immediately. No hoops.
- **Architectural input.** The language is small enough (23 concepts) that a new maintainer can genuinely understand the whole system. You won't be lost in a million-line codebase.

### What would help most

| Area | What's needed |
|------|--------------|
| **GPU runtime** | Vulkan experience, help with AMD/Intel validation, dispatch optimization |
| **Language design** | Someone who cares about keeping the language small and learnable |
| **Stdlib** | Domain experts — ML, audio, scientific computing, data engineering |
| **Testing** | More hardware, more edge cases, fuzzing, property-based testing |
| **Documentation** | Tutorials, guides, examples — written for humans, not just LLMs |
| **Community** | Someone who wants to help people use this thing |

### Why you might want to

- You think GPU compute should be accessible without CUDA
- You want to work on a language that's small enough to hold in your head
- You're curious about what happens when AI writes 90% of the code and a human architects 100% of the decisions
- You believe in building tools that empower users rather than creating dependency

If any of that resonates: [open an issue](https://github.com/octoflow-lang/octoflow/issues), email, or just start reading the code. The stdlib is right here. The docs explain the architecture. Jump in.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Website](https://octoflow-lang.github.io/octoflow/) | Landing page with live demos |
| [Language Guide](docs/language-guide.md) | Full language reference |
| [Loom Engine](docs/loom-engine.md) | GPU VM architecture deep-dive |
| [Stdlib Reference](docs/stdlib.md) | All 445 modules |
| [GPU Guide](docs/gpu-guide.md) | GPU compute patterns and best practices |
| [Builtins](docs/builtins.md) | 210+ built-in functions |

---

## Building from Source

The compiler source is currently in a private repository. If you're interested in building from source or contributing to the compiler, [open an issue](https://github.com/octoflow-lang/octoflow/issues) — the developer will get you access.

The stdlib and everything in this repo can be explored and modified immediately.

---

## License

MIT. The stdlib, examples, documentation, and everything in this repository.

The compiler will be MIT too, once it's open-sourced. No license change, no dual licensing, no surprises.

---

<p align="center">
  <sub>Built with AI. Decided by humans. GPU-native from day one.</sub>
</p>
