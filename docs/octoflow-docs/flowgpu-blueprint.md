# OctoFlow: A General-Purpose Programming Language with Automatic GPU Acceleration

## Blueprint & Architecture Document

**Status:** Concept / Pre-Design  
**Version:** 0.2  
**Date:** February 15, 2026  

---

## 1. Problem Statement

Modern programming languages were designed around CPU-centric paradigms established in the 1960s-1970s. Even languages that target GPUs (CUDA, OpenCL, Triton) treat the GPU as a peripheral accelerator subordinate to a CPU host program. The programmer must:

- Explicitly manage host↔device boundaries
- Manually decide what runs on GPU vs CPU
- Write "kernel launch" code from a CPU-centric worldview
- Handle memory transfers, synchronization, and device management

This forces programmers to think about execution hardware rather than computation. The result: GPU computing remains inaccessible to most programmers, and even experts spend more time on plumbing than on algorithms.

Meanwhile, general-purpose languages like Python and Rust can do everything — web servers, GUIs, system scripts, data analysis — but they treat GPU as an external tool requiring specialized libraries and manual orchestration.

**GPUs are already dataflow machines.** Streaming Multiprocessors are processing stages. Warp schedulers are dataflow schedulers. Shared memory is inter-stage buffering. The register file is local state within a stage. The hardware is dataflow-native — but every existing language forces a sequential, thread-based abstraction on top of it.

**The opportunity**: a general-purpose language where GPU acceleration is invisible — the compiler handles it automatically while the programmer writes normal code for any task: math, web servers, GUIs, scripts, databases, mobile apps.

---

## 2. Vision

A **general-purpose programming language** where:

1. **The computation model is dataflow-native.** Programs are graphs of transformation stages connected by typed data pipes. Parallelism is the default, not an annotation.

2. **The compiler automatically decides GPU vs CPU execution.** Through static analysis, cost modeling, and runtime profiling, the compiler partitions the dataflow graph across available hardware for optimal performance. The programmer never specifies where code runs. String operations silently go to CPU. Matrix operations silently go to GPU.

3. **It can do everything Python and Rust can do.** This is not a GPU DSL — it's a full general-purpose language with standard library modules for file I/O, networking, string manipulation, databases, HTTP, datetime, regex, and more. Extended modules provide desktop GUI, mobile apps, web servers, machine learning, and any domain.

4. **Output is hardware-agnostic.** The compiled program runs on whatever hardware is available — NVIDIA GPU, AMD GPU, CPU-only, heterogeneous — with performance scaling automatically to the available resources. Same code, different hardware, automatic adaptation.

5. **The frontend is LLM-agnostic.** Any LLM (local or cloud, any vendor, any size) can serve as a natural-language-to-code frontend through a constrained intent protocol with validation gates.

6. **The ecosystem grows through LLM-driven composition.** Users describe modules in natural language, LLMs compose them from safe vanilla primitives, the failsafe system validates automatically, and modules publish without manual review. The community scales exponentially.

---

## 3. Why OctoFlow — The Case for a New Language in the LLM Era

### 3.1 The Honest Question

If LLMs can already write CUDA, Python, Rust, and JavaScript — why build a new language? This is the most important question to answer, because if the answer is just "easier to code," OctoFlow loses. LLMs already made every language easy to code.

### 3.2 A Language Built for How Software Is Made Now

Software creation has fundamentally changed. In 2020, humans wrote code and occasionally got autocomplete suggestions. In 2026, LLMs write most of the code and humans review, steer, and compose. OctoFlow is the first language designed from scratch for this reality.

**The token economics argument.** When an LLM writes a GPU-accelerated data pipeline in Python, ~70% of the generated tokens are plumbing: memory management, device transfers, kernel configurations, import boilerplate, type conversions, error handling wrappers. These tokens cost money (API calls), time (latency), and reliability (each token is a potential bug).

OctoFlow eliminates plumbing. The same pipeline that takes 50-100 lines in Python+CUDA takes 3-5 lines in OctoFlow. This means:

| Metric | Python+CUDA (LLM-generated) | OctoFlow (LLM-generated) |
|--------|---------------------------|------------------------|
| Tokens generated | ~500-2000 | ~30-100 |
| Bug surface | Every line | Only the logic |
| GPU correctness | LLM's best guess | Compiler-guaranteed |
| Retry probability | High (CUDA errors are cryptic) | Low (pre-flight catches errors) |
| API cost per generation | Higher | 10-20x lower |

**A fine-tuned small LLM can master OctoFlow completely.** The language has 23 concepts, ~40 vanilla operations, and one composition pattern (pipes). A 1-3B parameter model fine-tuned on OctoFlow can achieve near-perfect code generation — something impossible for Python (massive surface area) or Rust (complex type system). This means:

- Local LLMs on laptops can generate OctoFlow reliably
- Edge devices can generate and execute OctoFlow without cloud LLM calls
- The cost of AI-generated GPU computing drops to near zero
- No vendor lock-in to any LLM provider

### 3.3 What LLMs Cannot Do (That the Compiler Does)

LLMs generate text by pattern matching on training data. They do not execute computation. This creates fundamental gaps that no amount of scaling will close:

**Global pipeline optimization.** An LLM writes one function at a time. It cannot see that data from stage A should stay on GPU for stages B, C, D because the transfer cost exceeds the compute savings of moving stage C to CPU. OctoFlow's compiler sees the entire dataflow graph and makes globally optimal partitioning decisions through actual cost computation — not pattern matching.

**Safety guarantees at scale.** Can an LLM guarantee that a 200-stage pipeline won't run out of GPU memory? That a float32 accumulator won't overflow after 10 million iterations? That backpressure won't stall the system? The pre-flight system checks all of this mathematically before execution. An LLM writing Python is guessing. OctoFlow's compiler is proving.

**Hardware adaptation.** The same OctoFlow program runs optimally on an NVIDIA RTX 4090, an AMD RX 7900, an Intel Arc, or a CPU-only server. The compiler adapts at build time or runtime. An LLM writing CUDA produces NVIDIA-only code. An LLM asked to "make it work on AMD too" produces a second codebase — doubling bugs, doubling maintenance.

### 3.4 The Compelling Use Cases

#### Invisible GPU for Everyone

A forex trader, a data scientist, a small business owner says to an LLM: "analyze my sales data and show me trends." The LLM generates OctoFlow. The compiler puts the heavy math on GPU. The result comes back 100x faster than the Python equivalent.

The user never knew GPU was involved. They didn't ask for it, didn't install CUDA, didn't configure anything. The language made GPU computing as invisible as electricity.

This is the iPhone moment of GPU computing — not making GPU programming better, but making GPU computing accessible to people who don't think of themselves as GPU programmers.

#### 10-20x Cost Reduction in AI-Generated Code

Every LLM API call costs tokens. Every retry costs more tokens. OctoFlow's minimal syntax means fewer tokens per generation, lower bug rates mean fewer retries, and a fine-tuned small model can replace expensive frontier model calls for code generation.

For a company generating thousands of data pipelines per day via LLM, OctoFlow cuts the AI compute budget by an order of magnitude while improving reliability.

#### The Module Flywheel as Network Effect

When someone writes a good function in Python, it helps nobody automatically. When someone creates a OctoFlow module (via LLM, 30 seconds), it is automatically tested, benchmarked, published, and available to every other LLM generating OctoFlow code.

The ecosystem compounds. More modules → richer LLM vocabulary → more sophisticated compositions → more modules. This is the network effect that makes OctoFlow defensible — not the syntax, not the compiler, but the exponentially growing library of safe, composable, GPU-accelerated modules.

#### Write Once, Run on Any Hardware

One OctoFlow codebase compiles to native binary (GPU servers), WebAssembly (browsers), Python module (ML pipelines), shared library (any language), or REST service (any client). The compiler adapts GPU/CPU decisions per target.

No other language offers: "describe your computation once, deploy to server GPU, browser WebGPU, mobile GPU, and CPU-only fallback — with one codebase and zero hardware-specific code."

#### Enterprise Pipeline Consolidation

A typical data platform has Python scripts, Spark jobs, CUDA kernels, SQL queries, and REST microservices — each maintained separately, each with its own failure modes. OctoFlow replaces all of these with one dataflow pipeline that the compiler optimizes end-to-end.

The dataflow graph IS the architecture. No distributed systems team needed to coordinate between services. The compiler handles data placement, parallelism, and fault tolerance.

### 3.5 Positioning Summary

OctoFlow is not competing with Python or Rust as a language humans type. It is competing for a different role entirely:

```
                    Traditional world         LLM era
                    ─────────────────         ────────
Code author:        Human programmer          LLM (any model, any size)
GPU knowledge:      Required                  Invisible (compiler handles it)
Module creation:    Weeks of expert work      30 seconds via natural language
Hardware targeting: Manual per-platform       Automatic per-compilation target
Safety:             Developer discipline      Compiler-enforced guarantees
Ecosystem growth:   Linear (human-authored)   Exponential (LLM-composed)
```

> **OctoFlow is the compilation target for the LLM era — where AI writes the code, the compiler manages the hardware, and humans describe what they want.**

---

## 4. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  LAYER 3: FRONTEND (LLM-Agnostic)                        │
│                                                            │
│  Human Intent (natural language, any human language)       │
│       ↓                                                    │
│  Pre-prompt (grammar, operations catalog, rules, examples) │
│       ↓                                                    │
│  Any LLM → Structured Intent (JSON/YAML)                   │
│       ↓                                                    │
│  Validation Gates:                                         │
│    Gate 1: Schema validation (structural correctness)      │
│    Gate 2: Type checking (pipe compatibility)              │
│    Gate 3: Dependency validation (graph well-formedness)   │
│    Gate 4: Operation validation (op existence, arity)      │
│       ↓ (retry loop on failure → error fed back to LLM)   │
│  Deterministic Transpiler → valid source code              │
├──────────────────────────────────────────────────────────┤
│  LAYER 2: LANGUAGE (Dataflow-Native)                      │
│                                                            │
│  Primitives: streams, pipes, stages, taps, accumulators   │
│  Semantics: parallel-by-default, temporal dependencies    │
│  Type system: dimensional, purity-tracked                 │
│  Control flow: iteration, recursion, conditionals         │
│  Escape hatches: @sequential, @parallel(dim)              │
│  Module system: composable, reusable pipeline fragments   │
├──────────────────────────────────────────────────────────┤
│  LAYER 1: COMPILER (Auto GPU/CPU)                         │
│                                                            │
│  Phase 1: Static Analysis                                 │
│    - Dataflow graph construction from source               │
│    - Dependency analysis (data, temporal, control)         │
│    - Purity checking (side-effect tracking)                │
│    - Arithmetic intensity estimation                       │
│    - Parallel profile annotation per stage                 │
│                                                            │
│  Phase 2: Cost Model                                      │
│    - Hardware-parameterized performance estimation         │
│    - GPU vs CPU execution time modeling                    │
│    - Data transfer overhead calculation                    │
│    - Decision function generation for conditional stages   │
│                                                            │
│  Phase 3: Graph Partitioning                              │
│    - Classify stages: hard-GPU / hard-CPU / conditional    │
│    - Minimum-transfer partitioning (min-cut optimization)  │
│    - GPU promotion analysis (is it cheaper to run a        │
│      mediocre-GPU stage on GPU than to pay transfer cost?) │
│    - Subgraph fusion (merge adjacent same-device stages)   │
│                                                            │
│  Phase 4: Code Generation                                 │
│    - GPU subgraphs → custom MLIR dialect → gpu dialect     │
│      → nvvm/rocdl → LLVM → PTX/AMDGCN                    │
│    - CPU subgraphs → MLIR → LLVM → x86/ARM native         │
│    - Transfer nodes → async memory copy operations         │
│    - Runtime scheduler code generation                     │
│                                                            │
│  Output: Single binary (GPU + CPU code + scheduler)        │
├──────────────────────────────────────────────────────────┤
│  RUNTIME                                                   │
│                                                            │
│  Dataflow scheduler (stage execution ordering)             │
│  Memory manager (GPU/CPU allocation, pooling)              │
│  Transfer coordinator (async GPU↔CPU data movement)        │
│  Hardware discovery (available GPUs, capabilities, memory)  │
│  Profiler (optional: feeds data back to cost model)        │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Layer 1: Compiler — Auto GPU/CPU Classification

### 4.1 The Core Problem

Given an arbitrary computation expressed as a dataflow graph, automatically determine the optimal assignment of each stage to GPU or CPU, minimizing total execution time including data transfer overhead.

### 4.2 Workload Classification Factors

| Factor | GPU Favorable | CPU Favorable |
|--------|--------------|---------------|
| Data size | Large (10,000+ elements) | Small (< 1,000 elements) |
| Operation type | Element-wise, matrix, convolution | String parsing, pointer chasing, tree traversal |
| Memory access | Coalesced, strided, predictable | Random, scattered, unpredictable |
| Arithmetic intensity | High (many ops per byte) | Low (load, one op, store) |
| Control flow | Uniform across data elements | Highly divergent branching |
| Dependencies | Independent across data elements | Chain dependency across all elements |

### 4.3 Static Analysis Phase

The compiler constructs a dataflow graph from source code. Each stage (node) is annotated with:

- **Purity**: Does this function have side effects? Pure functions are parallelizable.
- **Dimensionality**: Which data dimensions does this function operate over? Operations on dimension 0 can be parallelized across dimensions 1..N.
- **Arithmetic intensity**: Ratio of compute operations to memory accesses in the function body.
- **Dependency class**: Independent (embarrassingly parallel), temporal (iteration N depends on N-1), reduction (many→one), or sequential (strict ordering).

Classification output per stage:
- **Hard-GPU**: Statically guaranteed to be GPU-profitable (e.g., large matrix multiply, per-pixel image operation)
- **Hard-CPU**: Statically guaranteed to be CPU-only (e.g., file I/O, system calls, small irregular computation)
- **Conditional**: GPU-profitability depends on runtime data size; compiler generates a decision function

### 4.4 Cost Model Phase

For conditional stages, the compiler generates lightweight runtime decision logic:

```
estimated_gpu_time = data_size / (gpu_throughput × parallelism_factor)
estimated_cpu_time = data_size / cpu_throughput  
transfer_overhead = data_size × 2 / pcie_bandwidth  (round trip if needed)

if estimated_gpu_time + transfer_overhead < estimated_cpu_time:
    assign(GPU)
else:
    assign(CPU)
```

Hardware parameters (GPU throughput, PCIe bandwidth, SM count, etc.) are either:
- Auto-detected at first run and cached
- Provided by the user as a hardware profile
- Looked up from a known-hardware database

### 4.5 Graph Partitioning Phase

Once stages are classified, the compiler partitions the graph:

1. **Identify contiguous GPU subgraphs** — adjacent GPU stages that can share GPU memory without transfers
2. **Identify contiguous CPU subgraphs** — adjacent CPU stages
3. **Insert transfer nodes** at GPU↔CPU boundaries
4. **GPU promotion**: For a CPU stage sandwiched between GPU stages, evaluate whether running it (suboptimally) on GPU is cheaper than two transfers. If yes, promote it to GPU.
5. **Subgraph fusion**: Merge adjacent same-device stages into single execution units where possible

The optimization goal: **minimize total execution time = Σ(stage execution times) + Σ(transfer times)**, subject to the constraint that each stage runs on a device where it produces correct results.

### 4.6 Code Generation Phase

> **Detailed specification in Annex C.**

- **GPU subgraphs**: Compiled to SPIR-V compute shader bytecode via custom emitter → dispatched through Vulkan Compute API → GPU driver handles final compilation to hardware-native code. Vendor-neutral: works on NVIDIA, AMD, Intel, ARM without vendor-specific toolchains.
- **CPU subgraphs**: Compiled via Cranelift (lightweight Rust-native code generator) → native x86/ARM. Alternatively, interpreted for development/debug.
- **Transfer nodes**: Generated as async Vulkan buffer copies with staging buffers and double-buffering for streaming pipelines
- **Scheduler**: Generated as a lightweight coordinator that walks the execution order of the partitioned graph

**Key architectural decision**: No MLIR, no LLVM, no CUDA toolkit. SPIR-V + Vulkan Compute provides a vendor-neutral, specification-stable GPU target with minimal external dependencies. See Annex C Section 1 for full rationale.

### 4.7 Runtime

The compiled binary includes a thin runtime (~5-10MB GPU footprint) that:

- Discovers available hardware at startup via Vulkan device enumeration
- Evaluates conditional stage decision functions with actual data sizes
- Manages GPU memory allocation (pool-based to minimize allocation overhead)
- Coordinates async data transfers overlapped with computation
- Monitors GPU memory usage — OOM prevention is never disabled regardless of execution mode
- Falls back to CPU when GPU memory is insufficient (graceful degradation, never crash)
- Optionally profiles execution to refine cost model for future compilations

---

## 6. Layer 2: Language — Dataflow-Native Semantics

> **Detailed specification in Annex A.**

### 5.1 Design Principles

1. **Parallel by default.** If two computations have no data dependency, they execute in parallel. No annotation required.
2. **Sequential only when necessary.** The compiler identifies temporal and chain dependencies and sequences only those.
3. **Hardware-invisible.** The programmer never mentions GPU, CPU, threads, blocks, warps, or memory hierarchy.
4. **Familiar syntax.** Looks approachable to Python/JavaScript developers. Low barrier to entry.
5. **Escape hatches exist.** For expert users who need to override compiler decisions: `@sequential`, `@parallel(dim)`, `@device(gpu)`, `@device(cpu)`.

### 5.2 Core Primitives

| Primitive | Meaning | GPU Mapping |
|-----------|---------|-------------|
| `stream` | A typed, potentially infinite sequence of data | GPU global memory buffer |
| `stage` | A stateless transformation function | SM workgroup execution |
| `pipe` (`\|>`) | Typed connection between stages | Shared memory / L2 / HBM (compiler decides) |
| `tap` | External I/O boundary (data enters/exits the system) | CPU↔GPU transfer point |
| `accumulator` | Reduction point (many→one or many→few) | GPU parallel reduction |
| `temporal` | Marks a dependency across time steps | Pipelined execution |

### 5.3 Example: Video Processing

```
stream frames = tap("camera_feed")
stream decoded = frames |> h264_decode()
stream resized = decoded |> resize(1080, 1920)
stream filtered = resized |> color_grade(lut)
stream overlay = filtered |> composite(ui_layer)
stream encoded = overlay |> h264_encode()
emit(encoded, "output_file")
```

Compiler classification:
- `h264_decode()` → hard-CPU (external codec, irregular control flow)
- `resize()` → hard-GPU (per-pixel, embarrassingly parallel)
- `color_grade()` → hard-GPU (per-pixel, stays on GPU)
- `composite()` → hard-GPU (per-pixel, stays on GPU)
- `h264_encode()` → hard-CPU (external codec)

Result: One CPU→GPU transfer after decode, three fused GPU stages, one GPU→CPU transfer before encode.

### 5.4 Example: Statistical Analysis

```
stream trades = tap("trade_log")
stream cleaned = trades |> drop_null() |> cast_types()
stream grouped = cleaned |> group_by("symbol")
stream stats = grouped |> agg(mean, std, skew, kurtosis)
stream ranked = stats |> rank("sharpe_ratio")
stream top = ranked |> filter(rank <= 50)
emit(top, "report")
```

Compiler classification:
- `drop_null()`, `cast_types()` → conditional (element-wise with branching; GPU if dataset large)
- `group_by()` → hard-CPU (hash table, irregular memory access)
- `agg(mean, std, skew, kurtosis)` → hard-GPU (independent per group, compute-heavy)
- `rank()` → conditional (sort; GPU if many groups)
- `filter()` → hard-CPU (small output, not worth GPU)

### 5.5 Example: Financial Signal Generation

```
stream prices = tap("market_feed")
stream ema = prices |> temporal decay(0.94)
stream signals = ema |> crossover(fast=12, slow=26)
stream sized = signals |> position_size(risk=0.02)
emit(sized, "execution_engine")
```

Compiler classification:
- `decay()` with `temporal` → pipelined: sequential across time, parallel across instruments
- `crossover()` → hard-GPU (independent per instrument)
- `position_size()` → hard-GPU (independent per signal)

---

## 7. Layer 3: Frontend — LLM-Agnostic Interface

> **Detailed specification in Annex B.**

### 6.1 Design Principles

1. **LLM-agnostic.** The frontend is an interface specification, not an integration with any specific model.
2. **Constrained generation.** LLMs operate within rails defined by a pre-prompt containing grammar, valid operations, type rules, and examples.
3. **Gated validation.** LLM output passes through deterministic validation gates before reaching the compiler. Invalid output triggers a retry loop with specific error feedback.
4. **Extensible operations catalog.** Users can register custom stages with type signatures; these automatically become available in the pre-prompt for LLM consumption.
5. **Multi-frontend.** The same language accepts input from LLMs, human-written code, visual graph editors, or transpilers from other languages. All frontends produce the same Layer 2 code.

### 6.2 Frontend Stack

```
Human intent (natural language, any human language)
    ↓
Pre-prompt injection:
  - Intent schema (JSON/YAML format specification)
  - Operations catalog (available stages + type signatures)
  - Connection rules (type compatibility for pipes)
  - Constraint list (graph well-formedness rules)
  - Few-shot examples (common patterns)
    ↓
Any LLM (local 1.7B quantized to cloud frontier model)
    ↓
Structured Intent output (JSON/YAML)
    ↓
Validation Gates (sequential, fail-fast):
  Gate 1: Schema validation — is the JSON/YAML structurally correct?
  Gate 2: Type checking — do connected pipes have compatible types?
  Gate 3: Dependency validation — is the graph well-formed?
  Gate 4: Operation validation — do all referenced ops exist with correct arity?
    ↓ (on failure: specific error message fed back to LLM for retry)
Deterministic Transpiler
    ↓
Valid Layer 2 source code
```

### 6.3 Pre-prompt Architecture

The pre-prompt is a generated document (not hand-written) composed from:

1. **Static grammar section**: The intent format spec, always the same
2. **Dynamic operations catalog**: Auto-generated from registered stages and their type signatures
3. **Dynamic examples section**: Curated patterns relevant to the user's domain
4. **Constraint rules**: Formal rules the LLM must follow

This means:
- When a user defines a new custom stage, the pre-prompt automatically updates
- Domain-specific pre-prompts can be composed (financial computing, image processing, scientific computing)
- The pre-prompt scales from minimal (for powerful models) to verbose with many examples (for smaller models)

### 6.4 Structured Intent Format

The intermediate representation between LLM output and the language. Designed to be:
- Simple enough for small LLMs (1.7B parameter) to generate reliably
- Expressive enough to represent any valid dataflow program
- Deterministically transpilable to Layer 2 source code

> **Detailed intent schema in Annex B.**

---

## 8. Design Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Computation model | Dataflow graphs | Maps 1:1 to GPU hardware (SMs as stages, shared memory as pipes); parallelism is structural, not annotated |
| GPU/CPU decision | Compiler-automatic | Removes the #1 barrier to GPU adoption; programmers think about computation, not hardware |
| Language generality | General-purpose with dataflow core | Not a restricted DSL; supports iteration, recursion, conditionals. Compiler handles GPU/CPU assignment for all constructs |
| Frontend approach | LLM-agnostic with gated validation | Future-proofs against model churn; works with any model quality; deterministic correctness guarantees |
| Compiler infrastructure | MLIR-based | Avoids building GPU backends from scratch; leverages LLVM's mature NVIDIA/AMD code generation; multi-level optimization |
| Memory model | Compiler-managed | Programmer doesn't manage GPU/CPU memory or transfers; compiler determines placement from access patterns |
| Hardware portability | Single source, multi-target | Same code compiles to NVIDIA, AMD, CPU-only, heterogeneous; hardware differences absorbed by cost model and code gen |

---

## 9. Competitive Landscape

| System | What It Does | How OctoFlow Differs |
|--------|-------------|-------------------|
| CUDA | Low-level GPU programming | OctoFlow: no manual kernel management, automatic GPU/CPU split |
| OpenAI Triton | Block-based GPU kernel DSL | OctoFlow: not a kernel language — full programs; automatic device assignment |
| Mojo | MLIR-based Python superset | OctoFlow: dataflow-native (not imperative); automatic GPU/CPU (not programmer-directed) |
| Bend/HVM | Automatic parallelism via interaction combinators | OctoFlow: dataflow model maps to GPU hardware directly (not interpreted VM on GPU); practical performance |
| Futhark | Functional GPU array language | OctoFlow: general-purpose (not array-only); GPU-native with CPU for I/O only; LLM frontend |
| PyTorch/JAX | ML framework with GPU acceleration | OctoFlow: language-level (not library-level); works for any domain, not just ML |

---

## 10. Resource Constraints

### Target: < 1GB GPU memory for language/compiler overhead

| Component | Estimated GPU Memory | Notes |
|-----------|---------------------|-------|
| Compiled GPU kernels | 10-50 MB | Stage code is compact |
| Runtime scheduler | 5-10 MB | Lightweight coordinator |
| Memory pools | 50-100 MB | Pre-allocated buffers for data |
| **Language overhead total** | **~65-160 MB** | Leaves 840+ MB for user data |

The LLM frontend runs as a separate phase (pre-compilation) and does not consume GPU memory during program execution. If using a local LLM for the frontend, it can run on CPU or use GPU temporarily then release memory before the compiled program executes.

---

## 11. Annexes (Separate Documents)

| Annex | Content | Status |
|-------|---------|--------|
| **Annex A** | Layer 2: Language Specification — core dataflow primitives, vanilla compute library, standard modules (14 modules), extended module ecosystem (databases, web, GUI, mobile, ML, finance, games, system), **bridge & interop system** (compilation targets: WASM/JS/Python/Rust/C/shared lib, language bridges, protocol bridges), module system architecture, failsafe system, LLM composition flywheel, reactive extensions roadmap | Draft v0.2 |
| **Annex B** | Programming Model & Language Semantics — safety-first design, complete concept map (23 concepts), records/enums (no classes), let/var, stage/fn (no methods), pipes/match/if/for, modules/visibility, Result/Option error handling, type system, collections, strings, I/O model, compilation output (.flow → .fgb), project structure, package manager, REPL, testing, LLM & GUI builder implications | Draft v0.1 |
| **Annex C** | Layer 1: Compiler Internals — SPIR-V-based architecture, minimal dependencies, 10-milestone implementation plan with checkpoints, custom SPIR-V emitter, Vulkan Compute dispatch, Cranelift CPU backend, runtime with OOM prevention | Draft v0.1 |
| **Annex D** | Runtime Specification — scheduler design, memory manager, transfer coordinator, hardware discovery | Planned |
| **Annex E** | Implementation Roadmap — phased development plan, MVP scope, milestones | Planned |
| **Annex F** | Research Survey — detailed analysis of existing GPU languages, LLM code generation, MLIR ecosystem | Planned |

---

*This is a living document. Architecture decisions are subject to revision as design work progresses through the annexes.*
