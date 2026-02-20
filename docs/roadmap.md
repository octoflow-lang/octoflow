# OctoFlow — Implementation Roadmap & Validation Strategy

**Parent Document:** OctoFlow Blueprint & Architecture
**Purpose:** Step-by-step implementation plan for AI-assisted development (Claude Code)
**Status:** Phase 83e COMPLETE — Native video/GIF/AVI decoders, JPEG chroma fix, video_open/video_frame builtins, bit_shl/bit_shr/bit_xor builtins
**Version:** 0.83.1
**Date:** February 20, 2026

---

## Table of Contents

1. Implementation Philosophy
2. Development Environment
3. Phase 0: Prove SPIR-V + Vulkan (Days 1-2) -- DONE
4. Phase 1: GPU Pattern Library (Days 3-7) -- DONE
5. Phase 2: Minimal Language End-to-End (Days 8-10) -- DONE
6. Phase 3: Multi-Stage Pipelines & Cost Model (Days 11-15) -- DONE
7. Phase 4: Vanilla Operations Library (Weeks 3-4) -- DONE
8. Phase 5: Pre-Flight Safety System (Week 5) -- DONE
9. Phase 6: Image I/O + Channel Ops (Week 6) -- DONE
10. Phase 7: OctoMedia CLI (Week 7) -- DONE
11. Phase 8: Conditionals + Comparisons -- DONE
12. Phase 9: Print Interpolation -- DONE
13. Phase 10: Source Locations & Error Diagnostics -- DONE
14. Phase 11: Program Parameterization -- DONE
15. Phase 12: Scalar Functions + Count Reduce -- DONE
16. Phase 13: Watch Mode (Hot Reload) -- DONE
17. Phase 14: Vec Types (vec2/vec3/vec4) -- DONE
18. Phase 15: Struct Types -- DONE
19. Phase 16: Array/List Operations -- DONE
20. Phase 17: Mutable State -- DONE
21. Phase 18: .octo Binary Format -- DONE
22. Phase 19: While Loops -- DONE
23. Phase 20: For Loops -- DONE
24. Phase 21: Nested Loops -- DONE
25. Phase 22: Break/Continue -- DONE
26. Phase 23: If/Elif/Else Statement Blocks -- DONE
27. Phase 24: User-Defined Scalar Functions -- DONE
28. Phase 25: Random Number Generation -- DONE
29. Phase 26: String Type -- DONE
30. Phase 26b: REPL -- DONE
31. Phase 27: Module State -- DONE
32. Phase 28: For-Each Loops -- DONE
33. Phase 29: Array Mutation -- DONE
34. Phase 30a: Stdlib + Array Parameter Passing -- DONE
35. Phase 30b: HashMap Builtin -- DONE
36. Phase 31: File I/O -- DONE
37. Phase 32: String Operations + Type Conversion -- DONE
38. Phase 33: Advanced Array Operations -- DONE
39. Phase 34: Error Handling -- DONE
40. Phase 35: HTTP Client -- DONE
41. Phase 36: JSON I/O -- DONE
42. Phase 37: Environment & OctoData -- DONE
43. Phase 38: Closures/Lambdas + HashMap Bracket Access -- DONE
44. Phase 39: Structured CSV + Value::Map -- DONE
45. Phase 40: Command Execution -- DONE
46. Phase 41: Bitwise Operations -- DONE
47. Phase 42: Regex Operations -- DONE
48. Phase 43: Self-Hosting Foundation -- DONE
49. Phase 44-46: eval.flow (OctoFlow meta-interpreter) -- DONE
50. Phase 47: parser.flow (recursive descent in OctoFlow) -- DONE
51. Phase 48: preflight.flow (static analyzer in OctoFlow) -- DONE
52. Phase 49: codegen.flow (GLSL shader emitter in OctoFlow) -- DONE
53. Phase 50: Bootstrap Verified (eval.flow = Rust runtime, 7/7) -- DONE
54. Phase 51: Rust OS-Boundary Audit (74% replaceable, 18815/25290 lines) -- DONE
55. Phase 52: GPU Benchmark (6-line sigmoid pipeline, LOC vs Python+CUDA) -- DONE
56. Phases 53-65: Bootstrap Verification + Stage 6 -- DONE
57. Phases 66-68: SPIR-V Emitters + GPU-Authored Shaders -- DONE
58. Phases 69-82: GPU-Native GPL Full Stack -- DONE
59. Phase 83e: Native Video Decoders + JPEG Fix -- DONE
60. Phase 83: Public Release v0.83.1 -- SHIPPED
61. What's Next — Prioritized Roadmap
62. Future: GPU-Accelerated Compilation
63. Future: Fractal Compression (Phase 90-93)
64. Future: Homeostasis (Phase 95-100)
17. Validation Gates & Decision Points
17. Risk Registry
18. Document Index

---

## 1. Implementation Philosophy

### 1.1 Validate Riskiest Assumption First

The riskiest assumption in the entire project is:

> "We can emit SPIR-V bytecode from Rust and execute it on a real GPU via Vulkan Compute, without depending on MLIR, LLVM, or any vendor-specific toolkit."

If this doesn't work, the architecture is wrong and we need to reconsider. So we test this FIRST — before building a parser, type system, or anything else.

### 1.2 AI-Assisted Development Model

The implementation model is:

```
Human (architect):
  - Reviews outputs at validation gates
  - Makes design decisions at decision points
  - Does NOT write code

Claude Code (implementor):
  - Reads the design documents (this repo)
  - Implements each phase
  - Writes tests
  - Reports results at validation gates

Documents (contract):
  - Blueprint: overall architecture
  - Annex A: language spec, modules, ecosystem
  - Annex B: programming model, safety, semantics
  - Annex C: compiler internals, milestones
  - This document: implementation sequence and validation criteria
```

### 1.3 Implementation Language

**Rust** for the OS-boundary bootstrap + **.flow** for the compiler itself (48% self-hosted).

**Zero external Rust dependencies.** Only system libraries:
- `vulkan-1.dll` / `libvulkan.so` — GPU compute (raw bindings via `vk_sys.rs`, no `ash` crate)
- `ws2_32.dll` — Windows sockets

Originally planned `ash` (Vulkan) and `cranelift` (CPU codegen) crates — both eliminated. Raw Vulkan bindings since Phase 49. CPU codegen unnecessary — the interpreter handles all CPU work.

### 1.4 Repository Structure

```
flowgpu/
├── README.md                    # Project overview, links to docs
├── docs/                        # All design documents
│   ├── blueprint.md             # Master architecture
│   ├── annex-a-language-spec.md # Language specification
│   ├── annex-b-programming-model.md # Programming model
│   ├── annex-c-compiler-internals.md # Compiler internals
│   └── roadmap.md               # This document
├── arms/
│   ├── flowgpu-spirv/           # Phase 0-1: SPIR-V emitter
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs           # SPIR-V module builder
│   │   │   ├── opcodes.rs       # SPIR-V opcode definitions
│   │   │   ├── types.rs         # SPIR-V type emission
│   │   │   └── patterns/        # GPU computation patterns
│   │   │       ├── map.rs       # element-wise parallel
│   │   │       ├── reduce.rs    # parallel reduction
│   │   │       ├── temporal.rs  # temporal pipeline
│   │   │       └── fused.rs     # multi-stage fusion
│   │   └── tests/
│   ├── flowgpu-vulkan/          # Phase 0-1: Vulkan compute runtime
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs           # Vulkan context, pipeline, dispatch
│   │   │   ├── device.rs        # hardware detection
│   │   │   ├── memory.rs        # buffer allocation, transfers
│   │   │   └── dispatch.rs      # compute shader dispatch
│   │   └── tests/
│   ├── flowgpu-parser/          # Phase 2: parser
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── flowgpu-compiler/        # Phase 3+: full compiler
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── graph.rs         # dataflow graph
│   │       ├── analyzer.rs      # static analysis
│   │       ├── cost_model.rs    # GPU/CPU cost estimation
│   │       ├── partitioner.rs   # graph partitioning
│   │       └── codegen.rs       # SPIR-V + CPU code generation
│   ├── flowgpu-runtime/         # Phase 3+: execution runtime
│   │   ├── Cargo.toml
│   │   └── src/
│   └── flowgpu-cli/             # Phase 2+: command line interface
│       ├── Cargo.toml
│       └── src/
├── vanilla/                     # Phase 4: vanilla operations
│   ├── operations/              # operation definitions
│   └── tests/                   # correctness test vectors
├── examples/                    # Example .flow programs
│   ├── hello.flow
│   ├── statistics.flow
│   ├── financial.flow
│   └── video_concept.flow
└── Cargo.toml                   # workspace root
```

---

## 2. Development Environment

### 2.1 Requirements

```
Hardware:
  - Any machine with a Vulkan-capable GPU
  - NVIDIA (any GTX/RTX), AMD (any RX/Radeon), Intel (Arc/integrated)
  - At minimum 2GB VRAM for development
  - CPU-only fallback for CI/testing

Software:
  - Rust toolchain (rustup, latest stable)
  - Vulkan SDK (for spirv-val validation tool)
  - GPU driver with Vulkan support
  - Linux recommended (best Vulkan support)
  - macOS possible via MoltenVK
  - Windows possible but less tested
```

### 2.2 Verification

Before starting Phase 0, verify the environment:

```bash
# Verify Rust
rustc --version

# Verify Vulkan
vulkaninfo | head -20

# Verify SPIR-V tools (optional but helpful)
spirv-val --version

# Verify GPU is detected
vulkaninfo | grep "GPU"
```

If `vulkaninfo` fails, the GPU driver doesn't support Vulkan. Fix this first.

---

## 3. Phase 0: Prove SPIR-V + Vulkan (Days 1-2) — DONE

### 3.1 Objective

Build a Rust program that constructs SPIR-V bytecode in memory, dispatches it as a Vulkan compute shader, and verifies correct GPU output. No parser, no language, no compiler — just proof that the GPU backend works.

### 3.2 Task for Claude Code

```
Build a Rust program in arms/flowgpu-spirv and arms/flowgpu-vulkan that:

1. SPIR-V Emitter (flowgpu-spirv):
   - Constructs a SPIR-V compute shader module in memory (no files, no external tools)
   - The shader multiplies each element of a float32 buffer by 2.0
   - SPIR-V is constructed byte-by-byte following the SPIR-V 1.6 spec
   - Output: Vec<u8> containing valid SPIR-V binary
   - The emitter should be structured as a library with a SpirVModule builder
     that can be extended later for other operations

2. Vulkan Runtime (flowgpu-vulkan):
   - Creates a Vulkan instance and selects a compute-capable device
   - Creates a compute pipeline from the SPIR-V bytes
   - Allocates GPU buffers (input + output)
   - Uploads input data: [1.0, 2.0, 3.0, 4.0, 5.0, ..., 1024.0]
   - Dispatches the compute shader with appropriate workgroup size
   - Downloads results back to CPU
   - Verifies output matches expected values (each element × 2.0)

3. Uses the `ash` crate for Vulkan bindings (add to Cargo.toml)

4. Includes a test that runs the full pipeline and asserts correctness

5. If spirv-val is available on the system, validate the SPIR-V before dispatch

Key constraints:
- NO external SPIR-V generation libraries (we emit bytes directly)
- NO MLIR, LLVM, or CUDA dependencies
- The SPIR-V emitter must be reusable (it will become the foundation 
  of the compiler's GPU code generator)
- Error handling with Result types (no unwrap in library code)

Reference documents:
- docs/annex-c-compiler-internals.md, Section 9 (Milestone 6: SPIR-V Code Generator)
- SPIR-V spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
- Vulkan compute tutorial concepts (but implement in Rust with ash)
```

### 3.3 Expected Output

```
$ cargo test -p flowgpu-spirv
running 2 tests
test test_emit_valid_spirv ... ok
test test_spirv_validates ... ok

$ cargo test -p flowgpu-vulkan
running 1 test
test test_element_wise_multiply_gpu ... ok

$ cargo run --example phase0_demo
GPU: NVIDIA GeForce RTX XXXX
Input:  [1.0, 2.0, 3.0, ..., 1024.0]
Output: [2.0, 4.0, 6.0, ..., 2048.0]
PASS: All 1024 elements correct
Time: 0.XXms (including transfer)
```

### 3.4 Validation Gate

| Check | Pass Criteria |
|-------|--------------|
| SPIR-V emission | `spirv-val` accepts the generated binary |
| Vulkan pipeline creation | No Vulkan validation layer errors |
| GPU computation | Output matches expected for all 1024 elements |
| Error handling | Graceful failure message if no GPU available |

**GATE DECISION:**
- **PASS** → Proceed to Phase 1
- **FAIL (SPIR-V invalid)** → Fix emitter, retry
- **FAIL (Vulkan errors)** → Debug Vulkan setup, may need different ash usage patterns
- **FAIL (wrong results)** → SPIR-V logic error, debug with spirv-dis
- **FAIL (no Vulkan)** → Environment issue, need different hardware/driver

---

## 4. Phase 1: GPU Pattern Library (Days 3-7) — DONE

### 4.1 Objective

Extend the SPIR-V emitter to generate all core GPU computation patterns that the compiler will eventually need. Each pattern is tested independently.

### 4.2 Patterns to Implement

#### Pattern 1: Parallel Map (already done in Phase 0)
Element-wise operation. Each GPU thread processes one element.
```
Input:  float[N]
Op:     any element-wise function (multiply, add, subtract, abs, sqrt, etc.)
Output: float[N]
```

#### Pattern 2: Parallel Reduction
Many-to-one operation using workgroup shared memory.
```
Input:  float[N]
Op:     sum, min, max, count
Output: float (single value)
Method: Two-phase tree reduction
  Phase 1: Each workgroup reduces its chunk via shared memory + barriers
  Phase 2: Second dispatch reduces workgroup results
```

#### Pattern 3: Parallel Scan (Prefix Sum)
Running accumulation, parallelized.
```
Input:  float[N]
Op:     cumulative sum, cumulative product
Output: float[N]
Method: Blelloch scan algorithm in shared memory
```

#### Pattern 4: Temporal Pipeline
Sequential on one dimension, parallel on another.
```
Input:  float[T, N]  (T time steps, N instruments)
Op:     EMA / decay: out[t] = alpha * in[t] + (1-alpha) * out[t-1]
Output: float[T, N]
Method: Each thread handles one instrument across all time steps
        Parallel across N, sequential across T
```

#### Pattern 5: Fused Multi-Stage
Multiple operations compiled into a single kernel.
```
Input:  float[N]
Ops:    subtract(min) then divide(range) — normalization
Output: float[N]
Method: Single kernel that loads once, computes both ops, stores once
        Avoids intermediate buffer and extra kernel launch
```

### 4.3 Task for Claude Code

```
Extend flowgpu-spirv to support generating SPIR-V for each pattern above.

The SpirVModule builder should have methods like:
  fn emit_parallel_map(op: MapOp, input_type: DataType) -> SpirVModule
  fn emit_parallel_reduce(op: ReduceOp, input_type: DataType) -> SpirVModule  
  fn emit_parallel_scan(op: ScanOp, input_type: DataType) -> SpirVModule
  fn emit_temporal(op: TemporalOp, time_dim: usize, parallel_dim: usize) -> SpirVModule
  fn emit_fused(ops: Vec<MapOp>, input_type: DataType) -> SpirVModule

Each pattern should:
  - Generate valid SPIR-V (passes spirv-val)
  - Be dispatched via the Vulkan runtime
  - Produce correct results verified against CPU reference implementation

Write tests for each pattern with multiple input sizes:
  - Small: 16 elements (edge case, single workgroup)
  - Medium: 1024 elements (multiple workgroups)
  - Large: 1,000,000 elements (realistic workload)

Reference documents:
- docs/annex-c-compiler-internals.md, Section 9 (Milestone 6)
- SPIR-V spec for OpControlBarrier, Workgroup memory, SubgroupSize
```

### 4.4 Validation Gate

| Check | Pass Criteria |
|-------|--------------|
| All 5 patterns generate valid SPIR-V | `spirv-val` passes for all |
| Parallel map | Correct for multiply, add, subtract, abs, sqrt |
| Parallel reduce | sum([1..1000]) = 500500, min/max correct |
| Parallel scan | cumsum matches CPU reference within float32 tolerance |
| Temporal (EMA) | Matches CPU reference for 100 instruments × 1000 time steps |
| Fused pipeline | normalize() matches CPU reference |
| Large input (1M) | All patterns correct and complete in <100ms |

**GATE DECISION:**
- **PASS** → All GPU patterns work. The SPIR-V emitter is production-foundation quality.
- **PARTIAL** → Some patterns work, others don't. Identify failures, fix, retry.
- **FAIL (reduction wrong)** → Shared memory / barrier logic incorrect. Most common bug.
- **FAIL (temporal wrong)** → Thread indexing for 2D dispatch incorrect.

---

## 5. Phase 2: Minimal Language End-to-End (Days 8-10) — DONE

### 5.1 Objective

Build the thinnest possible OctoFlow parser and connect it to the SPIR-V backend. Prove that source code → GPU execution works end-to-end.

### 5.2 The v0.0.1 Language

The entire language for Phase 2:

```flow
// Only these constructs exist in v0.0.1:
stream data = tap("input.csv")
stream result = data |> multiply(2.0)
emit(result, "output.csv")
```

Supported operations: `multiply`, `add`, `subtract`, `divide`, `abs`, `sqrt`
Input/output: CSV files only (single column of float32 values)

### 5.3 Task for Claude Code

```
Build arms/flowgpu-parser and arms/flowgpu-cli:

Parser (flowgpu-parser):
  - Hand-written recursive descent parser (no parser generator dependency)
  - Parses the v0.0.1 language subset:
    - stream declarations with tap()
    - single-stage pipe chains (one |> operation)  
    - emit statements
  - Produces an AST with: TapNode, StageNode, EmitNode, PipeChain
  - Error messages with line numbers for invalid syntax

CLI (flowgpu-cli):
  - `flow run program.flow` command
  - Reads .flow file → parses → maps stage to SPIR-V pattern → dispatches → writes output
  - Reads input CSV (single column of floats)
  - Writes output CSV

This connects:
  flowgpu-parser (source → AST)
  → stage lookup (AST → SPIR-V pattern selection)  
  → flowgpu-spirv (pattern → SPIR-V bytes)
  → flowgpu-vulkan (SPIR-V → GPU dispatch)
  → output

Write example programs in examples/:
  - examples/double.flow: multiply by 2
  - examples/normalize.flow: subtract(min) then divide(range) — use fused pattern

Reference documents:
- docs/annex-c-compiler-internals.md, Section 4 (Milestone 1: Parser)
- docs/annex-b-programming-model.md, Section 5 (stage & fn)
```

### 5.4 Validation Gate

| Check | Pass Criteria |
|-------|--------------|
| Parser | Parses all example programs, rejects invalid syntax with clear errors |
| End-to-end | `flow run double.flow` reads CSV, computes on GPU, writes correct CSV |
| Error handling | Missing file → clear error. Invalid syntax → line number + message. |
| No-GPU mode | If no GPU available, runs scalar ops on CPU with a warning message |

**GATE DECISION:**
- **PASS** → End-to-end works. This is the first runnable OctoFlow program.
- **FAIL** → Identify where the chain breaks (parser? codegen? dispatch? I/O?)

---

## 6. Phase 3: Multi-Stage Pipelines & Cost Model (Days 11-15) — DONE

### 6.1 Objective

Support pipe chains with multiple stages. Implement basic GPU/CPU classification and data transfer.

### 6.2 Extended Language

```flow
// v0.0.2: multi-stage pipes, temporal, print
stream data = tap("input.csv")
stream cleaned = data |> abs()
stream normalized = cleaned |> subtract(min) |> divide(range)
stream smoothed = normalized |> temporal decay(0.94)
emit(smoothed, "output.csv")
print("Done!")
```

### 6.3 Task for Claude Code

```
Extend the compiler to handle multi-stage pipelines:

1. Dataflow Graph Builder (flowgpu-compiler/graph.rs):
   - Convert AST pipe chains into a dataflow graph (nodes + edges)
   - Each stage is a node, each pipe is an edge
   
2. Static Analyzer (flowgpu-compiler/analyzer.rs):
   - Classify each node: GPU-capable or CPU-only
   - For Phase 3, simple rules:
     - Math operations (multiply, add, subtract, etc.) → GPU
     - temporal operations → GPU (parallel across N, sequential across T)
     - I/O operations (tap, emit, print) → CPU
   
3. Graph Partitioner (flowgpu-compiler/partitioner.rs):
   - Group adjacent GPU stages into GPU subgraphs
   - Group adjacent CPU stages into CPU subgraphs
   - Insert transfer nodes at GPU↔CPU boundaries
   - Fuse adjacent GPU stages into single dispatches where possible

4. Transfer Generator:
   - CPU→GPU buffer upload before GPU subgraphs
   - GPU→CPU buffer download after GPU subgraphs
   - Using existing flowgpu-vulkan transfer functions

5. Extend CLI:
   - `flow run` now handles multi-stage programs
   - `flow graph program.flow` dumps the dataflow graph in DOT format
   - `flow partition program.flow` shows GPU/CPU assignment

Reference documents:
- docs/annex-c-compiler-internals.md, Sections 5-8 (Milestones 2-5)
```

### 6.4 Validation Gate

| Check | Pass Criteria |
|-------|--------------|
| Multi-stage | `a \|> b \|> c` compiles and runs correctly |
| GPU fusion | Adjacent GPU stages produce one kernel dispatch (check via logging) |
| CPU stages | `print()` works, outputs to stdout |
| Temporal | EMA across instruments produces correct values |
| Transfer | GPU→CPU and CPU→GPU transfers are automatic and correct |
| Graph dump | `flow graph` produces valid DOT file, viewable in Graphviz |

---

## 7. Phase 4: Vanilla Operations Library (Weeks 3-4) — DONE

### 7.1 Objective

Implement the core vanilla operations. Each operation maps to a SPIR-V pattern and is tested against a CPU reference implementation.

### 7.2 Operations to Implement

**Priority 1 — Arithmetic & Element-wise (GPU map pattern):**
```
add, subtract, multiply, divide
abs, negate, sqrt, pow, exp, log
min (element), max (element)
floor, ceil, round
```

**Priority 2 — Reductions (GPU reduce pattern):**
```
sum, mean, min, max, count
std, variance
```

**Priority 3 — Array Operations (mixed patterns):**
```
sort (GPU parallel sort)
filter (GPU stream compaction)
slice, concat
map (with inline stage)
```

**Priority 4 — Temporal (GPU temporal pattern):**
```
rolling (window)
lag, lead, diff
cumsum, cumprod
decay (EMA)
```

**Priority 5 — Data Handling (CPU):**
```
drop_null, fill_null
cast (type conversion)
group_by
```

### 7.3 Task for Claude Code

```
For each operation:

1. Define in vanilla/operations/:
   - Type signature
   - CPU reference implementation (Rust, simple, correct)
   - SPIR-V pattern mapping
   - Memory formula
   - Test vectors (known input → expected output)

2. Implement SPIR-V generation in flowgpu-spirv:
   - Extend patterns/ for any new patterns needed
   - Each operation maps to an existing pattern with specific parameters

3. Register in the compiler:
   - Parser recognizes the operation name
   - Compiler selects correct SPIR-V pattern
   - Static analyzer knows the parallel profile

4. Test:
   - GPU output matches CPU reference within float32 tolerance (1e-6 relative error)
   - Test with small (16), medium (1K), and large (1M) inputs
   - Edge cases: empty input, single element, NaN in input, infinity

Reference documents:
- docs/annex-a-language-spec.md, Section 6 (Vanilla Standard Library)
```

### 7.4 Validation Gate — ALPHA Milestone

| Check | Pass Criteria |
|-------|--------------|
| All Priority 1 ops | Correct on GPU, match CPU reference |
| All Priority 2 ops | Correct on GPU, match CPU reference |
| Sorting | GPU parallel sort correct for 1M elements |
| Rolling window | rolling(50) on 10K elements matches reference |
| EMA / decay | temporal decay on 100 instruments × 1000 steps correct |
| Blueprint examples | Financial signal example compiles and runs correctly |
| Blueprint examples | Statistics example compiles and runs correctly |

**This is the ALPHA milestone.** After Phase 4, OctoFlow can run real-world data processing pipelines with automatic GPU acceleration.

---

## 8. Phase 5: Pre-Flight Safety System (Week 5) -- DONE

### 8.1 Objective

Implement pre-flight validation, range analysis, and dead code lint. Every program is checked before compilation.

### 8.2 What Was Implemented

- **Pre-flight validation** (`preflight.rs`): Checks undefined streams/scalars, unknown ops, wrong argument counts, typo suggestions (Levenshtein distance). Module `use` imports resolved from filesystem.
- **Range tracker** (`range_tracker.rs`): Tracks value ranges through pipeline stages. Warns about `sqrt(negative)`, `exp(overflow)`, `divide(0)`.
- **Dead code lint** (`lint.rs`): Detects unused streams, unused scalars, unused functions, unused imports, and redundancy patterns (`multiply(1)`, `add(0)`, `negate |> negate`).
- **CLI integration**: `flowgpu-cli check` runs all three analyses without GPU. `flowgpu-cli run` runs pre-flight before execution (fail = no execution).

### 8.3 Validation Gate -- PASSED

| Check | Result |
|-------|--------|
| Undefined refs | Caught with clear error messages |
| Unknown ops | Caught with typo suggestions (Levenshtein) |
| Arity checking | Wrong arg count caught |
| Range warnings | sqrt(negative), exp(overflow), div(0) flagged |
| Dead code | Unused streams/functions/imports detected |
| `flow check` | Works without executing program |

---

## 9. Phase 6: Image I/O + Channel Ops (Week 6) -- DONE

### 9.1 Objective

Add image support so OctoFlow can process photos (PNG/JPEG), not just CSV data. Add per-channel RGB operations for color grading.

### 9.2 What Was Implemented

- **Image I/O** (`image_io.rs`): `tap("photo.png")` loads pixels as flat RGB f32 [0-255], `emit(result, "output.png")` writes back. Routes by file extension (image vs CSV). Supports PNG and JPEG via the `image` crate.
- **Image dimension tracking**: `HashMap<String, (u32, u32)>` propagates width/height through the pipeline. `find_source_dims()` walks expression trees to resolve dimensions at emit time.
- **Channel-aware ops** (CPU, stride-3): `warm(amount)` -- R+amount, B-amount. `cool(amount)` -- R-amount, B+amount. `tint(r_shift, b_shift)` -- independent R/B shifts.
- **Filter library** (`examples/filters.flow`): Reusable functions -- brightness, contrast, gamma, invert, warm_tone, cool_tone.
- **Demo**: `examples/cinematic_photo.flow` applies cinematic color grading to photos.

### 9.3 Validation Gate -- PASSED

| Check | Result |
|-------|--------|
| PNG read/write | Roundtrip preserves pixels within float32 tolerance |
| JPEG read | Loads correctly (lossy, no roundtrip test) |
| Channel ops | warm/cool/tint produce correct RGB shifts |
| Dimension tracking | Width/height propagated through pipes and refs |
| Extension routing | Image extensions -> image I/O, CSV extensions -> CSV I/O |
| Real photo | Cinematic filter on mascot image produces visible color grading |

---

## 10. Phase 7: OctoMedia CLI (Week 7) -- DONE

### 10.1 Objective

Package GPU-accelerated image processing as a standalone CLI tool with built-in presets. No `.flow` file needed for common operations.

### 10.2 What Was Implemented

- **`octo-media` binary** (`octo_media.rs`): Second binary in flowgpu-cli crate, shares all modules via `lib.rs`.
- **`apply` subcommand**: `octo-media apply <preset|filter.flow> <input> [-o output]`. Supports single file and batch mode (multiple inputs + output directory).
- **7 built-in presets**: cinematic, brighten, darken, high_contrast, warm, cool, invert. Each is an embedded OctoFlow program with `INPUT`/`OUTPUT` placeholders.
- **AST rewriting**: `rewrite_io_paths()` swaps tap/emit paths in the parsed AST per-image.
- **Auto-naming**: `photo.jpg` -> `photo_filtered.png` when no `-o` specified.
- **`presets` subcommand**: Lists all available presets with descriptions.
- **Lib refactor**: Extracted shared modules (compiler, image_io, csv_io, preflight, range_tracker, lint) into `lib.rs` for two-binary crate.

### 10.3 Validation Gate -- PASSED

| Check | Result |
|-------|--------|
| All 7 presets | Produce correct output on test image |
| Custom .flow filter | `octo-media apply filter.flow photo.jpg` works |
| Batch mode | Multiple inputs processed to output directory |
| Auto-naming | Generates `_filtered.png` suffix when no `-o` |
| Preset listing | `octo-media presets` shows all 7 with descriptions |
| Error handling | Missing file, bad format, no inputs all caught |

---

## 11. Phase 8: Conditionals + Comparisons -- DONE

### 11.1 Objective

Add conditional expressions, comparison operators, and boolean logic to the OctoFlow scalar expression system.

### 11.2 What Was Implemented

- **Lexer** (13 new tokens): `if`, `then`, `else`, `true`, `false`, `>`, `<`, `>=`, `<=`, `==`, `!=`, `&&`, `||`
- **AST** (5 new ScalarExpr variants): `If`, `Compare`, `And`, `Or`, `Bool` + `CompareOp` enum
- **Parser**: Restructured from 2-level to 6-level operator precedence hierarchy: if/then/else > `||` > `&&` > comparisons > additive > multiplicative > atoms. Added parenthesized scalar expressions and `true`/`false` literals.
- **Compiler**: `eval_scalar()` extended with short-circuit `&&`/`||`, float equality tolerance (1e-6), conditional branching on `cond != 0.0`.
- **Preflight/Range/Lint**: All safety tools updated to traverse new AST variants.

### 11.3 Security Hardening (Ship Blockers, Annex O §16)

Implemented immediately after Phase 8:
- **Path traversal prevention**: `resolve_path()` rejects `..` components, confines .flow I/O to subtree
- **Image input limits**: 100 MB file size, 16384x16384 max dimensions
- **CSV input limits**: 50 MB file size, 10M value cap
- **`Security` error type**: Distinct from I/O errors for security-specific failures

### 11.4 Validation Gate -- PASSED

| Check | Result |
|-------|--------|
| if/then/else | Works as expression, returns f32, nests correctly |
| Comparisons | All 6 operators: `< > <= >= == !=` |
| Boolean logic | Short-circuit `&&` and `||` |
| Parenthesized | `(mx + mn) / 2.0` works |
| Security | Path traversal blocked, input limits enforced |
| Tests | 132 → 141 (9 security tests added) |

---

## 12. Phase 9: Print Interpolation -- DONE

### 12.1 Objective

Enable print statements to display computed scalar values with interpolation and precision formatting.

### 12.2 What Was Implemented

- **AST** (`PrintSegment` enum): `Literal(String)` and `Scalar { name, precision: Option<usize> }`. `Statement::Print` changed from `{ message: String }` to `{ segments: Vec<PrintSegment> }`.
- **Parser** (`parse_interpolation()`): Splits strings into segments. `{name}` for scalar refs, `{name:.N}` for precision, `{{`/`}}` for escaped braces. Error on unclosed/empty braces.
- **Compiler**: Iterates segments at runtime, substitutes scalars with optional precision formatting via `format!("{:.prec$}", value)`.
- **Preflight**: Validates interpolated scalar refs against `defined_scalars` set.
- **Lint**: Recognizes scalar usage in print segments (prevents false D002 dead code warnings).
- **Example**: `examples/interpolation.flow` demonstrates all features.

### 12.3 Validation Gate -- PASSED

| Check | Result |
|-------|--------|
| Scalar substitution | `{name}` replaced with scalar value at runtime |
| Precision formatting | `{name:.2}` formats to 2 decimal places |
| Escaped braces | `{{` → `{`, `}}` → `}` |
| Preflight validation | Undefined scalars in print caught with suggestions |
| Lint integration | Scalars used in print not flagged as dead code |
| Tests | 141 → 147 (4 parser + 2 preflight tests added) |

---

## 13. Phase 10: Source Locations & Error Diagnostics -- DONE

### 13.1 Objective

Add source locations (line numbers) to AST nodes so that all error and warning messages include precise line references. Foundation for all future diagnostics.

### 13.2 What Was Implemented

- **AST** (`ast.rs`): Added `Span { line, col }` struct. Changed `Program::statements` from `Vec<Statement>` to `Vec<(Statement, Span)>`.
- **Parser** (`lib.rs`): Captures span (line/col) at start of each statement via `current_span()`. All 30+ test assertions updated for tuple destructuring.
- **Preflight** (`preflight.rs`): Added `line: usize` to `PreflightError` and `PreflightWarning`. Display shows `line N:` prefix. All validation loops propagate line through `check_expr()`, `check_stage()`, `check_scalar_expr()`.
- **Lint** (`lint.rs`): Added `line: usize` to `LintWarning`. Display shows `[code] line N:` format. `Definitions` changed from `HashSet<String>` to `HashMap<String, usize>` to track definition lines.
- **Compiler** (`compiler.rs`): All statement loops destructure `(stmt, _span)`.
- **Range tracker** (`range_tracker.rs`): All statement loops destructure `(stmt, _span)`.
- **OctoMedia** (`octo_media.rs`): `rewrite_io_paths()` preserves spans when constructing rewritten programs.

### 13.3 Validation Gate -- PASSED

| Check | Result |
|-------|--------|
| Preflight errors include line | `line 4: Unknown operation 'frobnicate'` |
| Preflight symbol errors | `line 2: Undefined stream 'bogus'` |
| Lint warnings include line | `[D001] line 2: Unused stream 'unused'` |
| OctoMedia span preservation | Rewritten programs keep original spans |
| Tests | 147 → 149 (2 line-number verification tests added) |

---

## 14. Phase 11: Program Parameterization -- DONE

### 14.1 Objective

Make `.flow` programs reusable CLI tools by allowing runtime overrides of scalar values and I/O paths without editing source code.

### 14.2 What Was Implemented

- **`Overrides` struct** (`lib.rs`): Central type holding `scalars: HashMap<String, f32>`, `input_path: Option<String>`, `output_path: Option<String>`.
- **`--set name=value` flag**: Overrides `let` scalar declarations at runtime. When `--set` is present, the original expression is skipped entirely (no side effects from the overridden evaluation).
- **`-i path` flag**: Overrides all `tap()` paths via AST rewriting (`apply_path_overrides`). Recursively rewrites Tap nodes inside Pipe expressions.
- **`-o path` flag**: Overrides all `emit()` paths via AST rewriting.
- **CLI arg parser** (`main.rs`): `parse_run_args()` extracts flow file + overrides from args. `parse_set_arg()` validates `name=value` format with numeric parsing.
- **Updated usage**: Extended help text shows `--set`, `-i`, `-o` options.
- **Override logging**: When overrides are active, logs them to stderr before execution.
- **OctoMedia passthrough**: OctoMedia passes `Overrides::default()` (its own path rewriting is independent).

### 14.3 Validation Gate -- PASSED

| Check | Result |
|-------|--------|
| `--set` overrides let | `--set factor=10.0` replaces `let factor = 2.0`, multiplies by 10 |
| `-i` overrides tap | Input path rewritten in AST including nested Pipe expressions |
| `-o` overrides emit | Output path rewritten in AST |
| Combined overrides | `--set` + `-i` + `-o` work together |
| Error handling | Bad format, missing value, unknown flags produce clear errors |
| OctoMedia unaffected | Passes default overrides, own rewriting still works |
| Tests | 149 → 163 (4 override + 10 arg parsing tests added) |

---

## 15. Phase 12: Scalar Functions + Count Reduce -- DONE

### Goal
Complete the scalar expression system by adding function calls (`abs(x)`, `sqrt(range)`, `pow(x, 2.0)`, `clamp(val, lo, hi)`) and `count(stream)` reduce.

### Tasks
- [x] Add `ScalarExpr::FnCall { name, args }` AST variant
- [x] Parser: whitelist reduce ops (`min`, `max`, `sum`, `count`); parse everything else as `FnCall`
- [x] Compiler: `eval_scalar_fn()` for 11 scalar functions; `count` reduce (CPU-only, returns `data.len()`)
- [x] Preflight: validate scalar function names (`SCALAR_FNS_1/2/3`) and arg counts
- [x] Lint: traverse `FnCall` args in `collect_scalar_usages()`
- [x] Range tracker: propagate ranges through scalar functions and `count`
- [x] Plan: `scalar_to_string()` and `plan_scalar()` support FnCall
- [x] Example: `examples/scalar_fns.flow`
- [x] Tests: 20 new tests (7 parser, 8 compiler, 5 preflight)

### Scalar Functions
| Function | Args | Description |
|----------|------|-------------|
| `abs(x)` | 1 | Absolute value |
| `sqrt(x)` | 1 | Square root |
| `exp(x)` | 1 | Natural exponential |
| `log(x)` | 1 | Natural logarithm |
| `sin(x)` | 1 | Sine |
| `cos(x)` | 1 | Cosine |
| `floor(x)` | 1 | Floor |
| `ceil(x)` | 1 | Ceiling |
| `round(x)` | 1 | Round to nearest |
| `pow(x, y)` | 2 | Exponentiation |
| `clamp(x, lo, hi)` | 3 | Clamp to range |

### Validation Gate

| Criterion | Result |
|-----------|--------|
| All scalar functions evaluate correctly | **PASS** — end-to-end verified |
| `count(stream)` returns correct element count | **PASS** — verified: count of [1,4,9,16,25] = 5 |
| Nested function calls work | **PASS** — `sqrt(abs(mn - mx))` |
| Preflight catches unknown scalar functions | **PASS** — "Unknown scalar function 'frobnicate'" |
| Preflight catches wrong arg count | **PASS** — "abs() requires 1 argument, got 2" |
| Range tracker propagates through scalar functions | **PASS** — correct intervals |
| All existing tests still pass | **PASS** — 163 original tests unchanged |
| Tests | 163 → 183 (7 parser + 8 compiler + 5 preflight tests added) |

---

## 16. Phase 13: Watch Mode (Hot Reload) -- DONE

### Goal

Add `--watch` / `-w` flag to `flowgpu-cli run` for automatic re-compilation and re-execution when `.flow` source files change. This is the #1 priority for gaming preparation (Annex Q §18.2) and benefits all OctoFlow development workflows.

### Tasks

- [x] Add `--watch` / `-w` flag to `parse_run_args()` — returns 3-tuple `(file, overrides, watch)`
- [x] Extract `run_once()` from `run()` — clean separation of single-run logic
- [x] Implement `watch_and_run()` — infinite loop: run, collect watched files, poll, re-run
- [x] Implement `collect_watched_files()` — parses main `.flow` file to find `use` imports
- [x] Implement `get_modification_times()` — polls `std::fs::metadata().modified()` every 500ms
- [x] Zero new dependencies — uses `std::fs` + `std::thread::sleep` (no `notify` crate)
- [x] Graceful error handling — errors are printed but don't exit watch mode
- [x] Tests: 8 new tests (watch flag parsing, file collection, modification time detection)

### Implementation Details

| Component | Description |
|-----------|-------------|
| `--watch` / `-w` | Boolean flag, combinable with `--set`, `-i`, `-o` |
| File polling | 500ms interval via `std::thread::sleep` |
| Import tracking | Parses `UseDecl` statements to find module `.flow` files |
| Error recovery | Compile/runtime errors printed, watch continues |
| Dependencies | Zero — pure `std` library (keeps minimal dependency philosophy) |

### Validation Gate

| Criterion | Result |
|-----------|--------|
| `--watch` flag parsed correctly | **PASS** — long and short forms |
| Watch flag combines with all existing options | **PASS** — `--set`, `-i`, `-o` |
| Watched files include `use` imports | **PASS** — cinematic_photo.flow collects filters.flow |
| Modification time detection works | **PASS** — real files return Some, missing return None |
| Error recovery in watch mode | **PASS** — errors printed, loop continues |
| All existing tests still pass | **PASS** — 183 original tests unchanged |
| Tests | 183 → 191 (8 new CLI main tests) |

---

## 17. Phase 14: Vec Types (vec2/vec3/vec4) -- DONE

### Goal

Add vec2/vec3/vec4 constructors with scalar-decomposed component access. This is the #2 priority for gaming preparation (Annex Q §18.2) — enables 3D position, UV coordinates, and RGBA color representation.

### Tasks

- [x] Parser: dotted scalar ref parsing (`v.x`, `v.y`, `v.z`, `v.w`) in `parse_scalar_atom()`
- [x] Compiler: vec constructor detection in `execute()` — validates component count, stores `name.x`/`name.y`/etc.
- [x] Compiler: `plan()` — `[VEC]` label for dispatch plan output
- [x] Preflight: vec constructor validation — correct arity, register component scalars
- [x] Preflight: vec2/vec3/vec4 added to known scalar functions (no false "unknown function" errors)
- [x] Lint: vec component definition tracking — registers `name.x`/etc. instead of vec name
- [x] Range tracker: vec component range propagation — individual component ranges
- [x] Example: `examples/vec_types.flow`
- [x] Tests: 16 new tests (6 parser, 6 compiler, 4 preflight)

### Design Decision: Scalar-Decomposed Vectors

`vec3(1,2,3)` creates three separate f32 scalars (`v.x`, `v.y`, `v.z`) in the scalars HashMap. No new AST variant needed — vec constructors parse as `ScalarExpr::FnCall` naturally. Dotted refs `v.x` parse as `ScalarExpr::Ref("v.x")`. Print interpolation `{v.x}` already works since `.` in the ref name is before the `:.` precision separator.

This avoids introducing a polymorphic type system while still enabling vector math for positions, colors, and coordinates.

### Validation Gate

| Criterion | Result |
|-----------|--------|
| vec2/vec3/vec4 constructors create component scalars | **PASS** — `vec3(1,2,3)` creates `.x`, `.y`, `.z` |
| Dotted ref access works in expressions | **PASS** — `pos.x * pos.x + pos.y * pos.y` |
| Wrong component count caught | **PASS** — `vec3(1,2)` → error |
| Print interpolation with components | **PASS** — `{pos.x:.2}` formats correctly |
| Preflight validates component existence | **PASS** — `pos.w` on vec3 → undefined |
| End-to-end example runs correctly | **PASS** — distance calculation matches expected value |
| All existing tests still pass | **PASS** — 191 original tests unchanged |
| Tests | 191 → 207 (6 parser + 6 compiler + 4 preflight tests added) |

---

## 18. Phase 15: Struct Types -- DONE

### Goal

Add user-defined struct types with named fields. This is the #3 priority for gaming preparation (Annex Q §18.2) — enables entity representation (players, enemies, game objects) with meaningful field names instead of generic x/y/z.

### Tasks

- [x] Lexer: `Token::Struct` keyword
- [x] AST: `Statement::StructDef { name, fields }` variant
- [x] Parser: `struct Name(field1, field2, ...)` syntax
- [x] Compiler: struct def storage, struct constructor decomposition in `execute()` and `plan()`
- [x] Preflight: struct arity validation, field registration, forward reference collection
- [x] Lint: struct field definition tracking, unused field detection
- [x] Range tracker: struct field range propagation
- [x] Example: `examples/struct_types.flow`
- [x] Tests: 11 new tests (4 parser, 4 compiler, 3 preflight)

### Design Decision: Scalar-Decomposed Structs

Structs follow the same scalar-decomposition pattern as vec types. `struct Entity(x, y, health)` defines a named record type, and `let e = Entity(1,2,100)` creates three f32 scalars (`e.x`, `e.y`, `e.health`). Struct names act as constructors — they parse as `ScalarExpr::FnCall` and are resolved against the struct definition table during execution.

Key differences from vec types:
- User-defined field names (not fixed x/y/z/w)
- Arbitrary number of fields (not limited to 2-4)
- Forward references work — struct definitions collected in first pass

### Validation Gate

| Criterion | Result |
|-----------|--------|
| Struct definition parses correctly | **PASS** — `struct Name(field1, field2, ...)` |
| Struct constructor creates field scalars | **PASS** — `Entity(1,2,3)` creates `.x`, `.y`, `.health` |
| Wrong field count caught | **PASS** — `Point(1,2)` on 3-field struct → error |
| Field access via dotted refs | **PASS** — `player.health` works in expressions and print |
| Undefined field caught | **PASS** — `p.z` on 2-field struct → undefined |
| Lint detects unused fields | **PASS** — `player.speed` flagged as unused D002 |
| End-to-end example runs correctly | **PASS** — entity math, field access, vec interop |
| All existing tests still pass | **PASS** — 207 original tests unchanged |
| Tests | 207 → 218 (4 parser + 4 compiler + 3 preflight tests added) |

---

## 19. Phase 16: Array/List Operations -- DONE

### Goal

Add array literals, index access, and `len()` function. This is the #4 and final HIGH priority item from Annex Q §18.2 — enables lookup tables, weight vectors, threshold lists, and dynamic parameterization for game data.

### Tasks

- [x] Lexer: `Token::LBracket` and `Token::RBracket`
- [x] AST: `Statement::ArrayDecl { name, elements }`, `ScalarExpr::Index { array, index }`
- [x] Parser: array literal detection in `parse_let_decl()`, index access in `parse_scalar_atom()`
- [x] Compiler: `arrays: HashMap<String, Vec<f32>>`, `ArrayDecl` execution, `eval_scalar` updated for `Index` and `len()`
- [x] Preflight: `ArrayDecl` validation, `len` as known scalar function, `Index` expression validation
- [x] Lint: `ArrayDecl` definition/usage tracking, `Index` scalar usage collection
- [x] Range tracker: array element range union, `Index` range propagation
- [x] Plan: `[ARRAY]` label for dispatch plan output, `Index` display in scalar_to_string
- [x] Example: `examples/array_ops.flow`
- [x] Tests: 12 new tests (5 parser, 4 compiler, 3 preflight)

### Design Decision: Separate Array Storage

Arrays are stored in a separate `HashMap<String, Vec<f32>>` rather than scalar-decomposing them (like vec/struct). This enables dynamic indexing (`arr[i]` where i is computed at runtime) and `len()` introspection. Array elements support full scalar expressions. Index access truncates f32 to usize.

### Validation Gate

| Criterion | Result |
|-----------|--------|
| Array literals with scalar expressions | **PASS** — `let arr = [a, b, a + b]` works |
| Index access with computed index | **PASS** — `arr[i + 1.0]` works |
| Out-of-bounds error | **PASS** — `arr[5.0]` on 2-element array → clear error |
| `len()` returns correct count | **PASS** — `len([1,2,3,4,5])` = 5.0 |
| Preflight validates array elements | **PASS** — undefined refs in elements caught |
| Dispatch plan shows arrays | **PASS** — `[ARRAY]` label in graph output |
| All existing tests still pass | **PASS** — 218 original tests unchanged |
| Tests | 218 → 230 (5 parser + 4 compiler + 3 preflight tests added) |

---

## 20. Phase 17: Mutable State -- DONE

### Goal

Add `let mut` mutable variable declarations and `name = expr` reassignment. This is the foundational piece needed for gaming (Annex Q) -- mutable state + frame loop + rendering unlocks playable demos. Once you have `let mut score = 0` and a way to modify it per frame, you're writing games.

### Tasks

- [x] Lexer: `Token::Mut` keyword
- [x] AST: `mutable: bool` field on `LetDecl`, new `Statement::Assign { name, value }` variant
- [x] Parser: `let mut` detection, bare `Ident = expr` assignment parsing with lookahead to distinguish from `==`
- [x] Compiler: `mutable_scalars: HashSet<String>` tracking, `Assign` handler with mutability check
- [x] Preflight: mutable variable tracking, assignment validation (target must be mutable or undefined)
- [x] Lint: `Assign` in definitions (no new names) and usages (target + value traversal)
- [x] Range tracker: `Assign` handler updates scalar range
- [x] Plan: `mut` prefix display for mutable declarations
- [x] Example: `examples/mutable_state.flow`
- [x] Tests: 12 new tests (5 parser, 4 compiler, 3 preflight)

### Design Decision: Mutability as Opt-In

Variables are immutable by default (`let x = 5.0`). Only `let mut x = 5.0` permits reassignment. This prevents accidental modification of constants while enabling state accumulation for game loops and counters. The mutability flag is enforced at both preflight (static check) and runtime (dynamic check).

Assignment parsing uses lookahead: bare `Ident` followed by `=` (but not `==`) triggers assignment. This cleanly distinguishes `x = 10.0` (assignment) from `x == 10.0` (comparison).

### Validation Gate

| Criterion | Result |
|-----------|--------|
| `let mut` declares mutable variable | **PASS** -- tracked in `mutable_scalars` set |
| Reassignment updates value | **PASS** -- `score = score + 10.0` works |
| Immutable assignment rejected | **PASS** -- preflight and compiler both catch |
| Undefined variable assignment rejected | **PASS** -- preflight with "did you mean?" suggestion |
| Conditional mutable init works | **PASS** -- `let mut label = if ... then ... else ...` |
| Accumulator pattern works | **PASS** -- `total = total + 1.0` repeated |
| Dispatch plan shows `mut` prefix | **PASS** -- graph mode annotates mutable declarations |
| All existing tests still pass | **PASS** -- 230 original tests unchanged |
| Tests | 230 → 242 (5 parser + 4 compiler + 3 preflight tests added) |

---

## 21. Phase 18: .octo Binary Format -- DONE

### Goal

Implement a GPU-friendly binary columnar storage format. `.octo` files store f32 data in a layout designed for zero-copy GPU buffer upload — no parsing, no string conversion, just binary f32 straight to VRAM. This is the storage foundation for OctoDB and all high-performance data workflows.

### Tasks

- [x] New module: `octo_io.rs` — reader/writer for .octo binary format
- [x] Binary format: magic header, column descriptors, aligned data sections, footer stats
- [x] Raw encoding (0x00): direct f32 little-endian — zero overhead
- [x] Delta encoding (0x01): base value + f32 deltas — ideal for time-series
- [x] Multi-column support: write_octo_columns() for OHLCV and multi-field data
- [x] Metadata API: octo_info() reads header without loading all data
- [x] Compiler integration: tap()/emit() auto-route by `.octo` extension
- [x] Security: MAX_OCTO_FILE_BYTES (500 MB) limit, magic validation
- [x] Examples: `octo_format.flow` (write), `octo_readback.flow` (read roundtrip)
- [x] Tests: 11 new tests (roundtrip, multi-column, delta, metadata, error handling)

### Design Decision: Storage Format Not Database

The `.octo` format is a file format, not a database. Key design:
- **Header**: Magic "OCTO" + version(u16) + column_count(u32) + row_count(u64)
- **Column descriptors**: name + dtype + encoding + data_offset + data_size
- **Data alignment**: 16-byte aligned column data (Vulkan buffer compatible)
- **Footer**: per-column min/max statistics + sentinel
- **Delta encoding**: same byte count as raw (f32 deltas), but foundation for future zstd integration where small deltas compress 3-10x
- **Multi-column**: format supports N columns, initial API wraps single-column for stream compatibility

Performance advantage: reading 1M f32 values from CSV requires parsing 1M strings (~300ms). Reading from .octo requires reading 4MB of bytes (~2ms). **100-150x speedup** for data loading.

### Validation Gate

| Criterion | Result |
|-----------|--------|
| Raw roundtrip exact | **PASS** — bit-perfect f32 preservation |
| Delta roundtrip within tolerance | **PASS** — < 1e-4 error from accumulated deltas |
| Multi-column write/read | **PASS** — 4 columns (OHLC), mixed encodings |
| Metadata without full read | **PASS** — octo_info() returns version, columns, rows |
| Bad magic rejected | **PASS** — clear error message |
| Column out of range rejected | **PASS** — clear error message |
| 100K element dataset | **PASS** — first, middle, last values verified |
| tap() auto-detects .octo | **PASS** — reads normalized data (0.0 to 1.0) |
| emit() writes .octo | **PASS** — write from pipeline, read back verified |
| All existing tests still pass | **PASS** — 242 original tests unchanged |
| Tests | 242 → 253 (11 new octo_io tests) |

---

## 22. Phase 19: While Loops -- DONE

### Goal

Add `while <condition>` ... `end` loop construct with safety limit. Combined with `let mut` from Phase 17, this enables iteration patterns: countdowns, accumulators, search loops, and Newton's method convergence. The while loop is CPU-only — loop body executes scalar statements, not stream pipelines.

### Tasks

- [x] Lexer: `Token::While` and `Token::End` keywords
- [x] AST: `Statement::WhileLoop { condition: ScalarExpr, body: Vec<(Statement, Span)> }` variant
- [x] Parser: `parse_while()` — parses `while <condition> NEWLINE <body> end NEWLINE`
- [x] Compiler: WhileLoop handler with MAX_WHILE_ITERATIONS = 10,000 safety limit
- [x] Compiler: body supports LetDecl (including vec/struct), Assign, Print, ArrayDecl
- [x] Compiler: `plan()` — `[WHILE]`/`[END]` labels with indented body
- [x] Compiler: `apply_path_overrides()` — traverses while body for -i/-o rewrites
- [x] Preflight: condition validation, recursive body statement validation
- [x] Lint: WhileLoop traversal in `collect_definitions()` and `collect_usages()`
- [x] Range tracker: WhileLoop body traversal for range propagation
- [x] Example: `examples/while_loop.flow`
- [x] Tests: 15 new tests (1 lexer, 5 parser, 5 compiler, 4 preflight)

### Design Decision: CPU-Only Loop with Safety Limit

While loops execute entirely on CPU. The body supports scalar statements (let, assign, print, array) but not stream operations. This is consistent with while loops being inherently sequential — they depend on mutable state that changes each iteration.

The 10,000 iteration safety limit prevents infinite loops from hanging the process. Programs that legitimately need more iterations can be restructured into nested loops or batch processing.

The `end` keyword (not `}` or indentation) matches the existing `if/then/else` pattern and is LLM-friendly — clear block boundaries without bracket matching.

### Validation Gate

| Criterion | Result |
|-----------|--------|
| While loop executes correctly | **PASS** — countdown, factorial, power-of-2 search |
| False condition skips body | **PASS** — zero iterations when condition is 0.0 |
| Safety limit triggers | **PASS** — `while 1.0` stops after 10,000 iterations with error |
| Print inside loop works | **PASS** — prints each iteration |
| Let/assign inside loop work | **PASS** — mutable accumulator pattern |
| Preflight validates condition | **PASS** — undefined refs in condition caught |
| Preflight catches immutable assign | **PASS** — assign to non-mut inside loop rejected |
| Graph mode shows loop structure | **PASS** — `[WHILE]`/`[END]` with indented body |
| All existing tests still pass | **PASS** — 253 original tests unchanged |
| Tests | 253 → 268 (1 lexer + 5 parser + 5 compiler + 4 preflight tests added) |

---

## 23. Phase 20: For Loops -- DONE

**Goal:** Add counted iteration with `for i in range(start, end)` ... `end` syntax, complementing while loops (condition-based) with index-based iteration.

### Tasks

- [x] **Lexer** — Add `Token::For` and `Token::In` keywords
- [x] **AST** — Add `Statement::ForLoop { var, start, end, body }` variant
- [x] **Parser** — `parse_for()` method: `for <var> in range(<start>, <end>)` with body until `end`
- [x] **Compiler** — Evaluate start/end, iterate `start..end` (Python-style exclusive), MAX_FOR_ITERATIONS=10,000
- [x] **Preflight** — Validate start/end expressions, register loop variable as defined scalar in body scope
- [x] **Lint** — Traverse ForLoop in collect_definitions and collect_usages
- [x] **Range tracker** — Combined `WhileLoop | ForLoop` arm for body traversal
- [x] **Graph** — `[FOR]`/`[END]` labels in dispatch plan
- [x] **Tests** — 15 new tests: 1 lexer + 5 parser + 5 compiler + 4 preflight
- [x] **Example** — `for_loop.flow`: sum, sum-of-squares, empty range, factorial

### Design Decision

**Python-style exclusive upper bound:** `range(0, 5)` iterates 0,1,2,3,4 (not including 5). This matches Python conventions and makes `range(0, n)` produce exactly `n` iterations. Start and end are evaluated as i64 for clean integer iteration.

**Loop variable scoping:** The loop variable is inserted into scalars at each iteration and removed after the loop ends. It is accessible inside the body but not after the loop.

### Validation Gate

```
Gate 20: "Can programs iterate a known number of times?"
- for i in range(0, 5): accumulates sum = 0+1+2+3+4 = 10           PASS
- range(5, 3): empty range, zero iterations, no error               PASS
- range(0, 10001): exceeds MAX_FOR_ITERATIONS, compile error         PASS
- Loop variable accessible in body (let doubled = i * 2.0)          PASS
- Preflight catches undefined range refs and immutable assignment    PASS
- Check mode: READY, 0 warnings across all passes                   PASS
- Graph mode: [FOR]/[END] labels, body indented                     PASS
- 283 tests passing (152 CLI lib + 18 main + 87 parser + 11 SPIR-V + 15 GPU)
```

---

## 24. Phase 21: Nested Loops -- DONE

**Goal:** Enable for/while loops inside other for/while loop bodies at arbitrary nesting depth, unifying loop body handling via shared recursive helpers.

### Tasks

- [x] **Compiler** — Extract `execute_loop_body()` helper with recursive WhileLoop/ForLoop handling
- [x] **Preflight** — Extract `validate_loop_body()` helper with recursive nesting validation
- [x] **Lint** — Extract `collect_loop_body_defs()` and `collect_loop_body_usages()` recursive helpers
- [x] **Graph** — `plan_loop_body()` recursive helper with depth-based indentation
- [x] **Tests** — 9 new tests: 4 compiler (for-in-for, while-in-for, for-in-while, while-in-while) + 5 preflight (3 valid nesting, undefined ref in inner, immutable assign in inner)
- [x] **Example** — Updated `for_loop.flow`: nested for-in-for (3x3 table), while-in-for (countdown)

### Design Decision

**Recursive helper extraction:** Rather than duplicating body handling code, shared helper functions (`execute_loop_body`, `validate_loop_body`, `collect_loop_body_defs`, `collect_loop_body_usages`, `plan_loop_body`) recursively process loop bodies. This eliminates code duplication and naturally supports arbitrary nesting depth. The parser already handled nesting via recursive `parse_for`/`parse_while` — this phase extended the compiler, preflight, lint, and graph passes to match.

### Validation Gate

```
Gate 21: "Can loops be nested inside other loops?"
- for-in-for: 3x3 multiplication table sum = 36                     PASS
- while-in-for: countdown inside each iteration = 6                  PASS
- for-in-while: inner for sum inside while loop                      PASS
- while-in-while: nested countdown, 2*2 = 4 iterations              PASS
- Preflight catches errors in nested loop bodies                     PASS
- Graph shows indented nested [FOR]/[WHILE]/[END] blocks             PASS
- 292 tests passing (161 CLI lib + 18 main + 87 parser + 11 SPIR-V + 15 GPU)
```

---

## 25. Phase 22: Break/Continue -- DONE

**Goal:** Add `break` and `continue` statements for loop control flow — `break` exits the innermost loop immediately, `continue` skips remaining body and proceeds to the next iteration.

### Tasks

- [x] **Lexer** — `Token::Break` and `Token::Continue` keyword tokens
- [x] **AST** — `Statement::Break` and `Statement::Continue` variants
- [x] **Parser** — Parse break/continue as standalone statements (consume keyword + newline)
- [x] **Compiler** — `LoopControl` enum (Normal/Break/Continue), signal propagation from `execute_loop_body`, top-level error for break/continue outside loops
- [x] **Preflight** — `Break | Continue => {}` in `validate_loop_body` (valid inside loops), error at top level
- [x] **Lint** — Added Break/Continue to ignore patterns in `collect_definitions` and `collect_usages`
- [x] **Graph** — `[BREAK]` and `[CONTINUE]` labels in `plan_loop_body` output
- [x] **Tests** — 17 new tests: 7 compiler (break/continue in while/for/nested, top-level error) + 5 preflight (valid in loops, rejected at top level, nested) + 5 parser (parse break/continue, mixed, standalone)
- [x] **Example** — Updated `for_loop.flow`: break in for, continue in for, while+break, nested break

### Design Decision

**LoopControl enum signal propagation:** `execute_loop_body` returns `Result<LoopControl, CliError>` where `LoopControl` is `Normal | Break | Continue`. When `Break` or `Continue` is encountered, it returns immediately — the caller matches on the signal to `break` or `continue` the Rust loop. This naturally handles nested loops because each loop level matches its own `execute_loop_body` result independently. A `Break` in an inner loop only exits that inner loop; the outer loop's `execute_loop_body` returns `Normal` for that iteration.

### Validation Gate

```
Gate 22: "Can programs exit loops early or skip iterations?"
- break in while: exits immediately, no infinite loop              PASS
- break in for: exits on first iteration, body after break skipped PASS
- continue in for: skips body after continue, loop completes       PASS
- continue in while: skips body after continue, converges          PASS
- break in nested loop: only inner loop exits                      PASS
- break at top level: compile error                                PASS
- continue at top level: compile error                             PASS
- graph shows [BREAK] and [CONTINUE] labels                       PASS
- 309 tests passing (173 CLI lib + 18 main + 92 parser + 11 SPIR-V + 15 GPU)
```

---

## 26. Phase 23: If/Elif/Else Statement Blocks -- DONE

**Goal:** Add block-form `if`/`elif`/`else`/`end` statement blocks for multi-statement conditional branching, distinct from the existing expression-form `if/then/else`.

### Tasks

- [x] **Lexer** — `Token::Elif` keyword
- [x] **AST** — `Statement::IfBlock` with condition, body, elif_branches, else_body
- [x] **Parser** — `parse_if_block()` dispatched from `parse_statement()` on `Token::If`, `parse_block_body()` helper for terminator-aware body parsing
- [x] **Compiler** — Top-level `IfBlock` execution, `execute_block_stmt()` for nested if-block bodies, IfBlock in `execute_loop_body` with LoopControl propagation (break/continue pass through)
- [x] **Preflight** — `validate_if_body()` for top-level if-blocks, IfBlock in `validate_loop_body` for loops
- [x] **Lint** — IfBlock handling in `collect_definitions`, `collect_usages`, `collect_loop_body_defs`, `collect_loop_body_usages`
- [x] **Graph** — `[IF]`, `[ELIF]`, `[ELSE]`, `[END]` labels with depth-based indentation
- [x] **Tests** — 19 new tests: 7 compiler (if true/false/else/elif/in-loop/conditional-continue/nested) + 5 preflight (valid if/else/elif, undefined ref, if-in-loop) + 7 parser (simple/else/elif/multi-stmt/missing-end/in-loop/elif-keyword)
- [x] **Example** — `examples/if_blocks.flow`: branching, conditional break/continue, nested ifs

### Design Decision

**Dual if-form:** The language now has two forms of conditional:
1. **Expression-form**: `let x = if cond then A else B` — inline, returns a value, requires both `then` and `else`
2. **Block-form**: `if cond` / `elif cond` / `else` / `end` — multi-statement, no return value, optional elif/else

The parser distinguishes them by context: `if` at statement position → block-form; `if` inside scalar expression → expression-form. This mirrors Python's `x if cond else y` vs `if cond:` distinction.

**LoopControl propagation through if-blocks:** When an if-block appears inside a loop, break/continue inside the if body must propagate to the enclosing loop. This is handled by `execute_loop_body` calling itself recursively for the chosen if-branch body, and propagating Break/Continue signals upward.

### Validation Gate

```
Gate 23: "Can programs conditionally execute blocks of statements?"
- if true: body executes, result = 42                              PASS
- if false: body skipped, result unchanged                         PASS
- if/else: else branch executes when condition false               PASS
- if/elif/else: correct branch chosen by first matching condition  PASS
- if + break in loop: conditional early exit                       PASS
- if + continue in loop: conditional skip                          PASS
- nested if blocks: inner if inside outer if                       PASS
- graph shows [IF]/[ELIF]/[ELSE]/[END] labels                     PASS
- 328 tests passing (185 CLI lib + 18 main + 99 parser + 11 SPIR-V + 15 GPU)
```

---

## 27. Phase 24: User-Defined Scalar Functions -- DONE

**Goal:** Add user-defined scalar functions with `fn name(params)` ... `return expr` ... `end` syntax — imperative body with local variables, loops, conditionals, and function-calling-function support.

### Tasks

- [x] **Lexer** — `Token::Return` keyword
- [x] **AST** — `Statement::ScalarFnDecl { name, params, body }` and `Statement::Return { value }` variants
- [x] **Parser** — Disambiguate `fn name(params):` (pipeline fn) vs `fn name(params)\n...end` (scalar fn), parse `return <expr>`
- [x] **Compiler** — `ScalarFnDef` struct, `LoopControl::Return(f32)` variant, `execute_user_fn()` with local scope + param binding, return propagation through if/while/for
- [x] **Preflight** — `defined_fns` set threaded through `check_scalar_expr`, user fn names skip "unknown function" errors
- [x] **Lint** — Scalar fn definitions tracked in `collect_definitions`, fn call names tracked in `collect_scalar_usages` (prevents false unused-function warnings)
- [x] **Graph** — `[FN]`/`[RETURN]`/`[END]` labels in dispatch plan
- [x] **Tests** — 8 new compiler tests (double, locals, conditional return, loop, fn-calls-fn, no-return error, return-outside-fn error, fn in for loop) + 7 parser tests
- [x] **Example** — `examples/scalar_functions.flow`: double, distance, my_abs, factorial, perimeter, fibonacci, fn-in-for-loop

### Design Decision

**Reuse execute_loop_body for fn bodies:** Rather than duplicating statement execution logic, user-defined functions call `execute_loop_body` on their body statements. The `LoopControl::Return(f32)` variant carries the return value back up through nested control structures. `execute_user_fn()` creates a local scope (separate scalars/mutables/arrays HashMap), binds parameters, executes the body, and catches the Return signal.

**Parser disambiguation:** After parsing `fn name(params)`, if `:` follows → existing pipeline fn (FnDecl); if newline follows → scalar fn body until `end` (ScalarFnDecl). This preserves backward compatibility with pipeline functions.

### Validation Gate

```
Gate 24: "Can programs define and call user-written functions?"
- Simple fn (double): return x * 2.0                              PASS
- Local variables: let dx = x2 - x1 inside fn body                PASS
- Conditional return: if x >= 0 return x, else return -x           PASS
- Loop in fn: factorial via while loop                              PASS
- Fn calling fn: perimeter calls distance                           PASS
- No return error: fn without return → clear error                  PASS
- Return outside fn: top-level return → clear error                 PASS
- Fn used in for loop: sum of factorials(1..5)                      PASS
- Pipeline fn still works: fn name(params): body (backward compat)  PASS
- Lint: no false "unused function" warnings                         PASS
- 336 tests passing (193 CLI lib + 18 main + 106 parser + 11 SPIR-V + 15 GPU)
```

---

## 28. Phase 25: Random Number Generation -- DONE

**Status:** Complete
**Tests added:** 8 (6 compiler + 2 preflight)
**Total tests:** 351

### What was implemented

- `random()` — returns f32 in [0.0, 1.0) using xorshift64* algorithm
- Deterministic seeding via `--set seed=N`; time-based default if no seed
- RNG state threaded through all execution paths via `Cell<u64>` for interior mutability
- Preflight validation: `random()` recognized as 0-arg scalar function
- Range tracker: `random()` returns known range [0.0, 1.0]
- Example: `examples/random_demo.flow` — dice rolls, coin flips, user-defined random functions

### Key design decisions

- **xorshift64\***: Simple, fast, no dependencies, good quality for games/simulations
- **`Cell<u64>`**: Interior mutability through shared reference — adds only one parameter to function signatures
- **Seed override**: Reuses existing `--set seed=N` mechanism, no new CLI flags needed

---

## 29. Phase 26: String Type -- DONE

### What Was Built

First-class string support: the OctoFlow runtime goes from pure f32 to dual-type (f32 + String).

**Value enum**: `Value::Float(f32)` and `Value::Str(String)` replace bare `f32` throughout the compiler. Helper methods `as_float()`, `as_str()`, `is_float()`, `is_str()` provide safe extraction.

**String operations (MVP)**:
- **Literals**: `"hello world"` in scalar expressions
- **Concatenation**: `+` when both operands are strings → string result
- **Comparison**: `==`, `!=` for string vs string (ordered comparisons are type errors)
- **len(s)**: Returns character count as float (works on both arrays and strings)
- **contains(haystack, needle)**: Returns 1.0 if found, 0.0 otherwise
- **Print interpolation**: `{s}` works for both float and string values

**Strict typing**: `"hello" + 1.0` → type error. `"hello" == 1.0` → type error. This catches bugs early rather than silently coercing.

**CLI**: `--set name=World` passes string overrides (falls back to string when value isn't a valid number).

### Files Changed
- `arms/flowgpu-parser/src/ast.rs` — `StringLiteral(String)` variant in ScalarExpr
- `arms/flowgpu-parser/src/lib.rs` — Parse `Token::StringLit` as scalar atom
- `arms/flowgpu-cli/src/lib.rs` — `Value` enum, `Overrides` uses `HashMap<String, Value>`
- `arms/flowgpu-cli/src/compiler.rs` — eval_scalar returns Value, string ops, ~45 call sites
- `arms/flowgpu-cli/src/preflight.rs` — StringLiteral validation, contains() registration
- `arms/flowgpu-cli/src/range_tracker.rs` — StringLiteral range, len/contains ranges
- `arms/flowgpu-cli/src/lint.rs` — StringLiteral arm in usage collection
- `arms/flowgpu-cli/src/main.rs` — parse_set_arg string fallback

### Tests: 363 (213 CLI lib + 18 CLI main + 106 parser + 11 SPIR-V + 15 GPU)
12 new tests: string literal, concatenation, len, contains, equality, inequality, arithmetic error, cross-type error, ordered compare error, user fn strings, loop strings, --set string override.

### Key Decisions
- **CPU-only**: Strings are CPU-side plumbing, no GPU path needed
- **Strict typing**: No implicit coercion between string and float
- **LoopControl::Return(Value)**: Return values can now be strings

---

## 30. Phase 26b: REPL -- DONE

### What Was Built

Interactive shell for OctoFlow — `flowgpu-cli repl`. Persistent context across inputs, auto-print for bare expressions, multi-line detection for blocks/functions/arrays, special commands for introspection.

### Architecture Decisions

- **No new AST variant** — REPL tries `parse()` for statements, then `parse_expr()` for bare expressions
- **Context struct in compiler.rs** — Holds all persistent state, delegates to existing `eval_scalar`/`execute_loop_body`/etc.
- **Subcommand on existing binary** — `flowgpu-cli repl`, not a new binary
- **Simple multi-line detection** — Count block-openers vs `end`, unmatched `[`, unclosed `"`

### Auto-Print Rules

| Input type | Behavior |
|---|---|
| Bare expression (`2+2`, `x`, `greet("hi")`) | Evaluate and print result |
| `let` declaration | Execute silently |
| `print(...)` | Explicit print via stdout |
| `fn`/`struct` definition | Define silently, confirm: `fn <name> defined` |
| `stream`/`emit` | Execute, print summary |

### Special Commands

| Command | Action |
|---|---|
| `:help` | Syntax reference |
| `:vars` | List all variables with types and values |
| `:fns` | List all user-defined functions |
| `:type <name>` | Show type and value of a variable |
| `:reset` | Clear all state |
| `:load <file>` | Load and execute a .flow file |
| `exit` / `quit` | Exit the REPL |

### Files Modified/Created

| File | Changes |
|---|---|
| `arms/flowgpu-parser/src/lib.rs` | Added `parse_expr()` public entry point |
| `arms/flowgpu-cli/src/compiler.rs` | Added `Context` struct, `StmtResult` enum, `format_repl_value()` |
| `arms/flowgpu-cli/src/repl.rs` | **NEW** — REPL loop, multi-line detection, special commands, 13 tests |
| `arms/flowgpu-cli/src/lib.rs` | Added `pub mod repl` |
| `arms/flowgpu-cli/src/main.rs` | Added `"repl"` subcommand dispatch |

### Test Results

13 new tests in `repl.rs`:
- `test_context_arithmetic` — eval_expression(2+2) → Float(4)
- `test_context_variable_persist` — let x, then eval x+1
- `test_context_string_ops` — concat, len, contains
- `test_context_multiline_fn` — define fn across statements, then call
- `test_context_struct` — struct def + instantiate + field access
- `test_context_array` — array create, index
- `test_context_vec` — vec3 create, field access
- `test_context_reset` — reset clears all state
- `test_needs_continuation_fn` — "fn foo()" → true
- `test_needs_continuation_complete` — "let x = 5" → false
- `test_needs_continuation_bracket` — "[1, 2" → true
- `test_format_repl_float` — 4.0 → "4", 3.14 → "3.14"
- `test_format_repl_string` — Value::Str("hi") → `"hi"`

Total: 401 tests (251 CLI lib + 18 CLI main + 106 parser + 11 SPIR-V + 15 GPU)

---

## 31. Phase 27: Module State — DONE

Extended `use` imports to bring in ALL module-level definitions, not just pipeline functions.

**What's new:**
- `use module` now imports: pipeline fns, scalar fns, structs, constants (`let`), and arrays
- Dual registration: `name` and `module.name` both work (e.g., `PI` and `mathlib.PI`)
- Dotted function calls: parser handles `module.fn(args)` syntax in scalar expressions
- Module constants evaluate in module-local context (self-contained, no access to importer state)
- Module `let` declarations import as **immutable** regardless of `let mut` in source
- No transitive imports: nested `use` in modules doesn't leak to the importer
- Preflight validates imported scalar fns, structs, constants, and arrays
- Lint D004 (unused import) detects usage of all imported types, not just pipeline fns
- `plan()` shows detailed import summary: `[USE] use mathlib (3 scalar fns + 1 struct + 2 consts)`

**Files modified:** compiler.rs, preflight.rs, lint.rs, parser lib.rs (dotted fn calls)
**Example module:** `examples/mathlib.flow` (scalar fns, struct, constants)
**Tests:** 25 new (12 compiler + 5 preflight + 4 lint + 4 REPL)

---

## 32. Phase 28: For-Each Loops -- DONE

**Goal:** Add `for x in arr ... end` syntax to iterate over array elements.

**Completed:**
- AST: `ForEachLoop { var, iterable, body }` variant
- Parser: `parse_for()` disambiguates `range(...)` (ForLoop) vs identifier (ForEachLoop)
- Compiler: ForEachLoop handlers in `execute()`, `Context::eval_statement()`, `execute_loop_body()`, `apply_path_overrides()`, `plan()`, `plan_loop_body()`
- Preflight: validates iterable is a defined array, loop var registered as scalar
- Lint: ForEachLoop tracked in definitions and usages (iterable counted as array usage)
- break/continue work inside for-each bodies
- Loop var cleaned up after loop (no leak)
- 24 new tests (6 parser + 8 compiler + 4 preflight + 2 lint + 4 REPL)
- Example: `examples/foreach_demo.flow`

**Syntax:**
```flow
let scores = [85.0, 92.0, 78.0, 95.0]
let mut total = 0.0
for s in scores
  total = total + s
end
print("Total: {total}")
```

**Test count:** 425 (269 CLI lib + 18 CLI main + 112 parser + 11 SPIR-V + 15 GPU)

## 33. Phase 29: Array Mutation -- DONE

**Goal:** Add mutable array operations — element assignment, push, and pop.

**Completed:**
- AST: `ArrayAssign { array, index, value }`, `ArrayPush { array, value }` variants, `mutable: bool` on `ArrayDecl`
- Parser: `parse_array_assign()` for `arr[i] = val`, `parse_push()` for `push(arr, val)`, mutable flag threaded through `ArrayDecl`
- Compiler: `pop(arr)` as eval_scalar builtin (mutates array, returns value); `ArrayAssign`/`ArrayPush` handlers in all 6 execution contexts; `eval_scalar` signature changed to `arrays: &mut HashMap` for pop support
- Preflight: `pop()` recognized as builtin (arity 1, arg-skipped like `len()`); `ArrayAssign`/`ArrayPush` validated in top-level, loop body, if body contexts; mutable tracking
- Lint: `ArrayAssign`/`ArrayPush` tracked in usages and definitions
- Mutability enforcement: `let mut arr = [...]` required for assignment/push/pop; clear error messages
- 27 new tests (6 parser + 12 compiler + 4 REPL + 6 preflight - some overlap with existing)
- Example: `examples/array_mutation_demo.flow`

**Syntax:**
```flow
let mut scores = [85.0, 92.0, 78.0]
scores[1] = 99.0           // element assignment
push(scores, 100.0)        // append element
let last = pop(scores)     // remove and return last
```

**Design decisions:**
- `pop()` is an expression (returns value) implemented in `eval_scalar`'s FnCall handler, unlike `push()` which is a statement
- `eval_scalar` takes `&mut HashMap` for arrays to support `pop()` mutation
- Mutable arrays tracked via existing `mutable_scalars` HashSet (no name collisions since scalars and arrays use separate maps)
- Immutable arrays reject assignment/push/pop with clear "use `let mut`" error messages

**Test count:** 452 (290 CLI lib + 18 CLI main + 118 parser + 11 SPIR-V + 15 GPU)

## 34. Phase 30a: Stdlib + Array Parameter Passing -- DONE

**Goal:** Enable functions to receive arrays as parameters. Write standard library modules in pure .flow — the self-hosting milestone.

**Completed:**
- **Array parameter passing**: When calling a user function with an array name, the array is cloned into the function's local scope under the parameter name. Mutability inherited from caller. Mutated copies written back on return.
- **eval_scalar signature**: Added `mutable_scalars` parameter for threading mutability context through function calls
- **execute_user_fn**: New signature with `scalar_args`, `array_bindings`, `caller_arrays`, `caller_mutable` for proper array lifecycle management
- **Preflight**: `check_scalar_expr` accepts `defined_arrays` parameter; `ScalarFnDecl` body validation registers params as both scalars AND arrays (type-agnostic)
- **stdlib/math.flow**: `min_of`, `max_of`, `clamp_val`, `lerp`, `map_range`, `sign`, `deg_to_rad`, `rad_to_deg`
- **stdlib/array_utils.flow**: `arr_contains`, `index_of`, `arr_sum`, `arr_avg`, `arr_min`, `arr_max`, `swap`, `reverse`, `fill`, `binary_search`
- **stdlib/sort.flow**: `insertion_sort`, `bubble_sort` (in-place on mutable arrays)
- 19 new tests (15 compiler + 2 preflight + 2 REPL)
- Example: `examples/stdlib_demo.flow`

**Key design decisions:**
- Arrays passed by clone-in/copy-back — function gets its own copy, mutations propagated back to caller on return
- Mutability inherited from caller: if original array is `let mut`, function can mutate it; if `let`, mutation errors
- Preflight registers function params as both scalars and arrays (since type is unknown at definition time)
- No parser changes needed — array detection happens at runtime in eval_scalar's FnCall handler

**Test count:** 471 (309 CLI lib + 18 CLI main + 118 parser + 11 SPIR-V + 15 GPU)

## 35. Phase 30b: HashMap Builtin -- DONE

**Goal:** First-class hashmap data structure for key-value storage — every real application needs associative data.

**Completed:**
- **AST**: `Statement::MapDecl { name, mutable }`, `Statement::MapInsert { map, key, value }`
- **Parser**: `let m = map()` constructor detection, `map_set(m, key, value)` statement parsing
- **Compiler**: `hashmaps: HashMap<String, HashMap<String, Value>>` storage alongside arrays; `map_set` as statement, `map_get`/`map_has`/`map_remove`/`map_keys` as expression-level FnCall handlers; `len()` extended for maps; parameter threaded through all functions (execute_loop_body, execute_block_stmt, eval_scalar, execute_user_fn)
- **Preflight**: `defined_maps` tracking, MapDecl/MapInsert validation in main loop + validate_loop_body + validate_if_body; map_get/map_has/map_remove/map_keys arity checking; first-arg skip for map function names
- **Lint**: MapDecl tracked as scalar definition (D002 unused detection); MapInsert tracks map usage; collect_loop_body_defs/usages updated
- **Plan output**: `[MAP]` and `[CPU] map_set(...)` step labels
- **REPL**: hashmaps in Context struct, list_variables shows maps, reset() clears maps
- 25 new tests (15 compiler + 4 parser + 6 preflight)
- Example: `examples/hashmap_demo.flow`

**API:**
| Function | Type | Description |
|----------|------|-------------|
| `map()` | constructor | Create empty hashmap: `let mut m = map()` |
| `map_set(m, key, val)` | statement | Insert/update key (requires `let mut`) |
| `map_get(m, key)` | expression | Retrieve value (errors if key missing) |
| `map_has(m, key)` | expression | Returns 1.0 (exists) or 0.0 (missing) |
| `map_remove(m, key)` | expression | Remove and return value |
| `map_keys(m)` | expression | Sorted comma-separated string of all keys |
| `len(m)` | expression | Number of entries |

**Key design decisions:**
- Storage: `HashMap<String, HashMap<String, Value>>` — parallel to arrays `HashMap<String, Vec<f32>>`
- `map_set` as statement (like `push`) since it's a mutation, not a value-producing expression
- `map_get`/`map_has`/`map_remove`/`map_keys` as expression-level FnCall handlers
- String keys only — float keys require explicit `str()` conversion (future)
- Values can be `Value::Float(f32)` or `Value::Str(String)` — same dual-type system as scalars
- `map_keys` returns sorted string (not array) since arrays are `Vec<f32>` and keys are strings
- Mutability enforcement: same pattern as arrays via `mutable_scalars` set

**Test count:** 496 (330 CLI lib + 18 CLI main + 122 parser + 11 SPIR-V + 15 GPU)

## 36. Phase 31: File I/O — DONE

**Goal:** File system access with Deno-style security — read/write files, directory listing, path utilities, string splitting. Prerequisite: heterogeneous arrays (`Vec<f32>` → `Vec<Value>`) to support arrays of strings.

**What shipped:**
- Heterogeneous array upgrade: arrays hold `Value` (Float or Str) instead of only `f32`
- Security model: `--allow-read` / `--allow-write` CLI flags, deny-by-default, REPL defaults to allow
- Thread-local security flags (`Cell<bool>`) — zero signature changes to `eval_scalar`
- 13 new built-in functions across 4 categories
- `write_file` / `append_file` as AST statement variants (like `map_set`)
- `eval_array_fn()` helper for array-returning functions (`read_lines`, `list_dir`, `split`)
- 34 new tests (20 compiler + 4 parser + 10 preflight)
- Example: `examples/file_io_demo.flow`

**API:**
| Function | Type | Security | Description |
|----------|------|----------|-------------|
| `read_file(path)` | expression → Str | `--allow-read` | Read entire file as string |
| `write_file(path, content)` | statement | `--allow-write` | Overwrite file with content |
| `append_file(path, content)` | statement | `--allow-write` | Append content to file |
| `read_lines(path)` | expression → array | `--allow-read` | Read file lines into string array |
| `list_dir(path)` | expression → array | `--allow-read` | List directory entries as string array |
| `file_exists(path)` | expression → Float | `--allow-read` | 1.0 if exists, 0.0 otherwise |
| `file_size(path)` | expression → Float | `--allow-read` | File size in bytes |
| `is_directory(path)` | expression → Float | `--allow-read` | 1.0 if directory, 0.0 otherwise |
| `file_ext(path)` | expression → Str | none (pure) | Extract file extension |
| `file_name(path)` | expression → Str | none (pure) | Extract filename from path |
| `file_dir(path)` | expression → Str | none (pure) | Extract directory from path |
| `path_join(dir, file)` | expression → Str | none (pure) | Join path components |
| `split(str, delim)` | expression → array | none (pure) | Split string into array |

**Key design decisions:**
- Security via thread-local `Cell<bool>` flags — avoids threading 2 extra params through 82 `eval_scalar` call sites
- `write_file`/`append_file` as statements (like `map_set`) since they're mutations, not value-producing
- Array-returning functions detected at `LetDecl` level via `eval_array_fn()` — returns `Option<Vec<Value>>`
- Path utilities (`file_ext`, `file_name`, `file_dir`, `path_join`) are pure string ops — no security check needed
- Heterogeneous arrays: `Vec<Value>` enables `["XAUUSD", "EURUSD"]`, `[1.0, "hello"]`, `split("a,b", ",")`

**Test count:** 530 (360 CLI lib + 18 CLI main + 126 parser + 11 SPIR-V + 15 GPU)

## 37. Phase 32: String Operations + Type Conversion — DONE

**Goal:** Type conversion functions (`str`, `float`, `int`) and comprehensive string manipulation (`substr`, `replace`, `trim`, `to_upper`, `to_lower`, `starts_with`, `ends_with`, `index_of`, `char_at`, `repeat`). Bridges the gap between float and string types, enabling CSV field parsing, string building, and text processing.

**What shipped:**
- 3 type conversion functions: `str()`, `float()`, `int()`
- 10 string manipulation functions
- All implemented as FnCall handlers — zero new AST nodes
- 31 new tests (25 compiler + 6 preflight)
- Example: `examples/string_ops_demo.flow`

**API:**
| Function | Arity | Returns | Description |
|----------|-------|---------|-------------|
| `str(val)` | 1 | Str | Convert float to string (3.0→"3", 3.14→"3.14"); strings pass through |
| `float(val)` | 1 | Float | Parse string to float; floats pass through; errors on invalid |
| `int(val)` | 1 | Float | Truncate to integer (toward zero); also parses strings |
| `substr(s, start, len)` | 3 | Str | Extract substring (char-based, bounds-clamped) |
| `replace(s, old, new)` | 3 | Str | Replace all occurrences |
| `trim(s)` | 1 | Str | Strip leading/trailing whitespace |
| `to_upper(s)` | 1 | Str | Convert to uppercase |
| `to_lower(s)` | 1 | Str | Convert to lowercase |
| `starts_with(s, prefix)` | 2 | Float | 1.0 if string starts with prefix, else 0.0 |
| `ends_with(s, suffix)` | 2 | Float | 1.0 if string ends with suffix, else 0.0 |
| `index_of(s, needle)` | 2 | Float | Position of first match, or -1.0 if not found |
| `char_at(s, index)` | 2 | Str | Single character at position (errors on out-of-bounds) |
| `repeat(s, count)` | 2 | Str | String repeated N times |

**Key design decisions:**
- `str(3.0)` produces `"3"` not `"3.0"` — clean integer formatting for whole numbers
- `int()` truncates toward zero (like Rust's `trunc()`) — `int(-3.7)` = `-3.0`
- `substr` is char-based and bounds-clamped — no panics on out-of-range
- All functions are expression-level FnCall handlers — no parser changes needed
- Builtins take priority over user-defined functions with same name

**Test count:** 561 (391 CLI lib + 18 CLI main + 126 parser + 11 SPIR-V + 15 GPU)

## 38. Phase 33: Advanced Array Operations — DONE

**Goal:** Comprehensive array manipulation — aggregation (`sum`/`min`/`max`/`count` on arrays, not just streams), search (`find`, `first`, `last`), transformation (`reverse`, `slice`, `sort_array`, `unique`), construction (`range_array`), and string conversion (`join`). Plus `type_of` for runtime type introspection.

**What shipped:**
- 12 new array operations + `type_of` introspection
- `sum`/`min`/`max`/`count` extended to work on arrays via Reduce path (check arrays before streams)
- Array-returning functions via `eval_array_fn`: `reverse`, `slice`, `sort_array`, `unique`, `range_array`
- Scalar-returning array functions via `eval_scalar` FnCall: `join`, `find`, `first`, `last`, `min_val`, `max_val`
- Edge-case proof: empty arrays produce errors (not panics), bounds are clamped, duplicates preserved in order
- 44 new tests (38 compiler + 6 preflight)
- Example: `examples/array_ops_advanced.flow`

**API:**
| Function | Arity | Returns | Description |
|----------|-------|---------|-------------|
| `join(arr, delim)` | 2 | Str | Join array elements into string with delimiter |
| `find(arr, value)` | 2 | Float | Index of first match (-1.0 if not found); uses 1e-6 tolerance for floats |
| `first(arr)` | 1 | Value | First element (error on empty) |
| `last(arr)` | 1 | Value | Last element (error on empty) |
| `reverse(arr)` | 1 | Array | New reversed array |
| `slice(arr, start, end)` | 3 | Array | Subarray [start, end), bounds-clamped |
| `sort_array(arr)` | 1 | Array | New sorted array (floats ascending, strings alphabetical) |
| `unique(arr)` | 1 | Array | Remove duplicates, preserving first-seen order |
| `range_array(start, end)` | 2 | Array | Generate [start, start+1, ..., end-1] as float array |
| `type_of(val)` | 1 | Str | Returns "float" or "string" |
| `min_val(arr)` | 1 | Float | Minimum float value in array (error on empty) |
| `max_val(arr)` | 1 | Float | Maximum float value in array (error on empty) |
| `sum(arr)` | 1 | Float | Sum of float array (also works on streams) |
| `min(arr)` | 1 | Float | Min of float array (also works on streams) |
| `max(arr)` | 1 | Float | Max of float array (also works on streams) |
| `count(arr)` | 1 | Float | Element count (also works on streams) |

**Key design decisions:**
- `sum`/`min`/`max`/`count` dual-dispatch: parser produces `ScalarExpr::Reduce`, handler checks arrays first then falls through to streams
- `sort_array` not `sort` to avoid collision with stdlib bubble_sort user fn
- `find` uses 1e-6 tolerance for float comparison (matches `==` operator behavior)
- Empty arrays: `first`/`last`/`min_val`/`max_val` error cleanly; `reverse`/`sort_array`/`unique` return empty
- `slice` clamps bounds to [0, len] — no panics on out-of-range

**Test count:** 605 (435 CLI lib + 18 CLI main + 126 parser + 11 SPIR-V + 15 GPU)

## 39. Phase 34: Error Handling — DONE

**Goal:** Add `try()` error handling so runtime errors can be caught and inspected instead of crashing.

### Syntax

```flow
let r = try(read_file("config.json"))
if r.ok == 0.0
  print("Error: {r.error}")
end
print("Value: {r.value}")
```

### API

| Expression | Result |
|---|---|
| `let r = try(expr)` | Evaluates `expr`, catches any `CliError` |
| `r.value` | Result value on success, empty string `""` on error |
| `r.ok` | `1.0` on success, `0.0` on error |
| `r.error` | Empty string `""` on success, error message string on error |

### Design Decisions

- **LetDecl-level detection** — `try(expr)` parsed as `FnCall`, detected alongside `vec3()`/struct constructors at the LetDecl level. No AST/parser/lexer changes needed.
- **Scalar decomposition** — Same pattern as `vec3` creating `v.x`, `v.y`, `v.z`. The `try()` creates `r.value`, `r.ok`, `r.error` as separate scalars.
- **Bare try() rejected** — `try()` without `let r = ...` produces a compile error (caught by eval_scalar guard).
- **Catches all CliError variants** — Io, Csv, Parse, Compile, Gpu, Security, UndefinedStream, UndefinedScalar, UnknownOperation.
- **Works in all contexts** — Top-level, loops (while/for/for-each), user-defined scalar functions, modules, REPL.
- **Nested try()** — `try(try(x))` correctly catches inner error (bare try in eval_scalar).

### Validation Gate

- Preflight: registers `.value`, `.ok`, `.error` fields; validates 1-arg arity
- Lint: tracks decomposed fields for unused-variable detection (D002)
- Plan: shows `[TRY]` tag with field decomposition

**Test count:** 629 (459 CLI lib + 18 CLI main + 126 parser + 11 SPIR-V + 15 GPU)

## 40. Phase 35: HTTP Client -- DONE

**Status:** Complete
**Tests added:** 28 (18 compiler + 8 preflight + 1 main + 2 lint, minus 1 overlap)
**Total tests:** 657 (486 CLI lib + 19 CLI main + 126 parser + 11 SPIR-V + 15 GPU)

### What was implemented

- `http_get(url)` — GET request, 1 arg
- `http_post(url, body)` — POST with application/json content-type, 2 args
- `http_put(url, body)` — PUT with application/json content-type, 2 args
- `http_delete(url)` — DELETE request, 1 arg

### Response decomposition (LetDecl-level, 4 fields)

| Field | Type | Description |
|-------|------|-------------|
| `.status` | Float | HTTP status code (200.0, 404.0) — 0.0 on network error |
| `.body` | Str | Response body text |
| `.ok` | Float | 1.0 if status 200-299, 0.0 otherwise |
| `.error` | Str | Error message on failure, empty on success |

### Security

- `ALLOW_NET` thread-local `Cell<bool>` — same pattern as ALLOW_READ/ALLOW_WRITE
- `--allow-net` CLI flag required for network access
- REPL defaults to allowing network access
- Without flag: `CliError::Security("network access not permitted — use --allow-net")`

### Key design decisions

- **LetDecl-level decomposition**: Like try()/vec/struct, decomposes into named scalar fields
- **`ureq` crate**: Synchronous HTTP client, ~15 deps (vs reqwest's 100+), no async runtime needed
- **Non-throwing**: HTTP errors captured in `.ok`/`.error` fields — never throws, always safe
- **7 execution sites**: REPL, execute(), execute_loop_body(), execute_block_stmt(), import_module(), plan(), plan_loop_body()

### Files modified

| File | Changes |
|------|---------|
| `Cargo.toml` | Added `ureq = "2"` dependency |
| `src/lib.rs` | Added `allow_net: bool` to Overrides |
| `src/main.rs` | Added `--allow-net` CLI flag parsing |
| `src/compiler.rs` | ALLOW_NET flag, check_net_permission(), do_http_request() helper, 5 LetDecl handlers, eval_scalar guard, 2 plan handlers |
| `src/preflight.rs` | 4 locations: arity validation + field registration |
| `src/lint.rs` | 2 locations: dead-code tracking for 4 fields |

### Example

`examples/http_demo.flow` — GET/POST/error handling demo

---

## 41. Phase 36: JSON I/O -- DONE

**Status:** Complete
**Tests added:** 22 (16 compiler + 4 preflight + 2 lint)
**Total tests:** 679 (508 CLI lib + 19 CLI main + 126 parser + 11 SPIR-V + 15 GPU)

### What was implemented

- `json_parse(str)` — Parse JSON object → HashMap with dot-notation flattening for nested objects
- `json_parse_array(str)` — Parse JSON array → Vec<Value>
- `json_stringify(name)` — Serialize hashmap (with unflattening) or array to JSON string

### json_parse behavior

Parses JSON object into OctoFlow hashmap. Nested objects flattened with dot-notation:
- `{"user": {"city": "NYC"}}` → key `"user.city"` = `Str("NYC")`
- Numbers → Float, strings → Str, true → Float(1.0), false → Float(0.0), null → Str("")
- Nested arrays → Str(json_string)

### json_stringify behavior

Serializes hashmap or array back to JSON. Detects type by checking hashmaps first, then arrays.
Unflattens dot-notation keys back into nested JSON objects.

### Key design decisions

- **Dot-notation flattening**: Matches OctoFlow's flat scalar decomposition pattern (like HTTP/try fields)
- **LetDecl-level**: json_parse at hashmap level, json_parse_array at array level — same pattern as other constructors
- **`serde_json` crate**: Standard JSON library, mature, efficient
- **eval_hashmap_fn helper**: New pattern for hashmap-returning functions (parallel to eval_array_fn)
- **6 execution sites**: REPL, execute(), execute_loop_body(), execute_block_stmt(), import_module() + eval_scalar guard

### Files modified

| File | Changes |
|------|---------|
| `Cargo.toml` | Added `serde_json = "1"` dependency |
| `src/compiler.rs` | flatten/unflatten/value_to_json helpers, eval_hashmap_fn(), json_parse_array in eval_array_fn, json_stringify in eval_scalar, 5 LetDecl handlers, 2 plan handlers, eval_scalar guard, import_module hashmaps param |
| `src/preflight.rs` | 4 locations: arity + registration for json_parse/json_parse_array/json_stringify, collect_module_fns updated |
| `src/lint.rs` | 2 locations: dead-code tracking for json_parse/json_parse_array results |

### Example

`examples/json_demo.flow` — Build/serialize/parse/roundtrip JSON demo

---

## 42. Phase 37: Environment & OctoData -- DONE

**Status:** Complete
**Tests added:** 22 (15 compiler + 4 preflight + 1 lint + 2 parser)
**Total tests:** 701 (528 CLI lib + 19 CLI main + 128 parser + 11 SPIR-V + 15 GPU)

### What was implemented

**System utilities (eval_scalar):**
- `time()` — Unix epoch seconds as Float
- `env(name)` — Environment variable lookup, returns "" if unset
- `os_name()` — Returns "windows", "linux", or "macos"

**OctoData format (.od):**
- `load_data(path)` — Parse .od file into hashmap (LetDecl-level, requires --allow-read)
- `save_data(path, map)` — Serialize hashmap to .od file (statement-level, requires --allow-write)
- .od files are a pure-data subset of .flow syntax (only `let` declarations allowed)
- Expressions evaluated in sandboxed context (prior values visible)
- Array declarations stored in arrays namespace

### .od format spec

```
// config.od — pure data, no logic
let name = "Alice"
let age = 30.0
let items = [1.0, 2.0, 3.0]
let derived = age + 5.0
```

- Parsed by `flowgpu_parser::parse()`, then filtered to LetDecl/ArrayDecl only
- Print/if/while/for/fn/use etc. → error
- serialize_od() writes sorted keys with proper Float/Str formatting

### Key design decisions

- **Reuse existing parser**: .od files ARE valid .flow subsets — no new parser needed
- **Sandboxed evaluator**: load_data uses isolated scope — no GPU, no streams, no hashmaps
- **AST variant**: `SaveData { path: ScalarExpr, map_name: String }` — statement-level like WriteFile
- **Security**: load_data requires --allow-read, save_data requires --allow-write

### Files modified

| File | Changes |
|------|---------|
| `parser/ast.rs` | Added `SaveData` variant |
| `parser/lib.rs` | `parse_save_data()` method, statement dispatch, 2 tests |
| `src/compiler.rs` | time/env/os_name in eval_scalar, load_data in eval_hashmap_fn, serialize_od helper, SaveData at 4 execution sites + 2 plan sites, load_data guard |
| `src/preflight.rs` | time/os_name in SCALAR_FNS_0, env in SCALAR_FNS_1, load_data at 4 LetDecl sites, SaveData at 3 statement sites, collect_module_fns |
| `src/lint.rs` | load_data at 2 definition sites, SaveData at 3 usage/no-op sites |

### Example

`examples/od_demo.flow` — System utilities + save/load OctoData roundtrip

---

## 43. Phase 38: Closures/Lambdas + HashMap Bracket Access -- DONE

**Status:** Complete
**Tests added:** 36 (24 compiler + 5 preflight + 3 lint + 4 parser)
**Total tests:** 737 (560 CLI lib + 19 CLI main + 132 parser + 11 SPIR-V + 15 GPU)

### What was implemented

**Lambda expressions (ScalarExpr::Lambda):**
- Inline lambda syntax: `fn(params) expr end`
- Single expression body, 1+ parameters
- Snapshot capture: lambda body sees all outer scalars (cloned at invocation)
- No `Value::Lambda` — lambdas stay as AST nodes, only valid as arguments to higher-order builtins
- Bare lambda expression → error with descriptive message

**Higher-order builtins:**
- `filter(arr, fn(x) cond end)` — keep elements where condition is truthy (eval_array_fn)
- `map_each(arr, fn(x) expr end)` — transform each element (eval_array_fn)
- `sort_by(arr, fn(x) key end)` — sort by key function result (eval_array_fn)
- `reduce(arr, init, fn(acc, x) expr end)` — fold array to single value (eval_scalar)

**HashMap bracket access:**
- `map["key"]` and `map[var_name]` — reuses ScalarExpr::Index
- Runtime disambiguation: check arrays first, then hashmaps fallback
- String key evaluation from index expression

**Print interpolation fix (pre-existing bug):**
- Fixed `print("{arr_name}")` to render arrays as `[item1, item2, ...]`
- Fixed `print("{map_name}")` to render hashmaps as `{key1=val1, key2=val2}`
- Updated all 4 runtime print handler sites (REPL, execute_loop_body ×2, execute_block_stmt)
- Updated 3 preflight print handler sites to also check defined_arrays/defined_maps

### Key design decisions

- **No Value::Lambda**: Lambdas are AST-only — avoids runtime type complexity, lambdas can only appear inline as builtin args
- **Clone-before-iterate**: Source array cloned before iteration to release borrow, allowing &mut in eval_scalar during lambda invocation
- **invoke_lambda helper**: Clones outer scalars, binds params (shadowing), evaluates body in isolated scope
- **extract_lambda helper**: Pattern-matches Lambda variant from FnCall args with clean error message
- **Index disambiguation**: Array lookup takes precedence over hashmap — consistent with existing array-first semantics

### Files modified

| File | Changes |
|------|---------|
| `parser/ast.rs` | Added `Lambda { params, body }` variant to ScalarExpr |
| `parser/lib.rs` | Lambda parsing in parse_scalar_atom (Token::Fn arm), 4 tests |
| `src/compiler.rs` | extract_lambda + invoke_lambda helpers, filter/map_each/sort_by in eval_array_fn, reduce in eval_scalar, Index hashmap fallback, Lambda error arm, print interpolation for arrays/maps at 4 sites, 24 tests |
| `src/preflight.rs` | Lambda arm in check_scalar_expr, arity for filter/map_each/sort_by/reduce, skip-first-arg list, array-fn registration at 4 LetDecl sites, print array/map checks, 5 tests |
| `src/lint.rs` | Lambda arm in collect_scalar_usages, filter/map_each/sort_by at definition sites, 3 tests |
| `src/range_tracker.rs` | Lambda arm → Range::UNKNOWN |

### Example

`examples/lambda_demo.flow` — filter, map_each, sort_by, reduce, hashmap bracket access, snapshot capture, chaining, lambda with user function

---

## 44. Phase 39: Structured CSV + Value::Map -- DONE

**Goal:** Add `Value::Map` as third value variant, enabling structured CSV processing with header-based row access.

### Implemented

- **Value::Map(HashMap<String, Value>)** — third variant in the Value enum
  - `is_map()`, Display (`{key=val, ...}`), type_of → "map", str() conversion
  - All existing match arms updated (print, sort, filter truthiness, type conversion, serialization)
- **read_csv(path)** — parse CSV with headers into array of maps
  - Auto-detect types: parseable as f32 → Float, otherwise → Str
  - `parse_csv_line()` handles quoted fields, double-quote escaping
  - File size and row count limits (security)
  - Requires `--allow-read`
- **write_csv(path, arr)** — serialize array of maps to CSV
  - Header row from first map's keys (sorted alphabetically)
  - CSV quoting for fields containing commas/quotes
  - Requires `--allow-write`
- **Index on Value::Map scalars** — `row["field"]` when row is Value::Map
  - Priority: arrays → top-level hashmaps → Value::Map scalars → error
- **For-each with Value::Map** — works naturally (loop var becomes Value::Map scalar)
- **value_to_json Map arm** — json_stringify works on arrays of maps
- **Preflight, Lint, Plan** — all updated for read_csv + WriteCsv

### Tests: 765 (586 CLI lib + 19 main + 134 parser + 11 SPIR-V + 15 GPU)

### Example

`examples/csv_demo.flow` — read CSV, field access, filter, map_each, reduce, write filtered results

---

## 46. Phases 41+: Domain Foundation Roadmap

**Context:** Phase 40 complete (777 tests). Domain audit reveals OctoFlow has **strong general-purpose foundation** across 14 domains. Next phases focus on **critical missing primitives** that unlock high-value domains for LLM-generated libraries.

See `docs/domain-audit-2026.md` and `docs/phase-41-42-pain-points.md` for comprehensive analysis.

### Phase 41: Core Utilities Extension (NEXT - IMMEDIATE) ✅

**Rationale:** Unlock 8/14 domains to 9-10/10 readiness with ~350 lines
**Scope:** 18 tests, ~795 total

**Features:**
1. **Statistics builtins** — mean, median, stddev, variance, quantile, correlation
   - Unblocks: Data Science (10/10), Finance (8/10), Scientific (6/10)
   - GPU-accelerated where possible
2. **Path operations** — join_path, dirname, basename, file_exists, is_file, is_dir, canonicalize_path
   - Unblocks: DevOps (10/10), Systems (9/10)
   - Security: canonicalize prevents path traversal
3. **Base64/hex encoding** — encode, decode for both
   - Unblocks: Web (7/10), Security (5/10)
   - Enables: Auth headers, binary data handling
4. **File metadata extensions** — list_dir(path, detailed=true) → [{name, size, mtime, is_dir}, ...]
   - Unblocks: Systems (9/10), DevOps (10/10)
   - Solves: Metadata bottlenecks at scale

**Pain Points Solved:**
- Pandas CSV performance collapse (60x slowdowns)
- Bash filename-with-spaces disasters (#1 DevOps failure)
- Web API auth header encoding (no library import needed)
- Metadata bottlenecks on 100M+ file datasets

### Phase 42: Date/Time Operations (CRITICAL) ✅

**Rationale:** Unblocks Finance + Data Science (high-value domains)
**Scope:** 12 tests, ~807 total

**Features:**
1. **Timestamp parsing** — timestamp(iso_string), timestamp_from_unix, now()
2. **Formatting** — format_datetime(ts, fmt)
3. **Arithmetic** — add_seconds/minutes/hours/days, diff_seconds/days
4. **Ranges** — date_range(start, end, step) for backtesting
5. **Timezone basics** — tz_convert(ts, from_tz, to_tz)

**Pain Points Solved:**
- JavaScript Date timezone disasters (DST bugs, silent failures)
- Backtesting date range logic (pandas import overhead)
- Finance "last business day" calculations
- No standard timestamp comparison across timezones

**Domain Impact:**
- Finance: 7/10 → **10/10**
- Data Science: 10/10 (already strong, now complete)
- DevOps: 10/10 (timestamp-based log analysis)

### Phase 43: Security & Crypto Primitives (⚠️ Expert Review Required)

**Rationale:** Unblock Security domain, enable production-grade auth
**Scope:** ~400 lines, 15 tests, ~822 total

**Features:**
- Hashing: SHA256, BLAKE3
- Encryption: AES-GCM, ChaCha20-Poly1305
- Crypto-safe RNG: random_bytes(n)
- Decimal type: For finance money math (no float errors)

**CRITICAL:** Security audit before merge. Use battle-tested libraries only (ring, sodiumoxide).

### Phase 44-45: TCP/UDP Sockets + HTTP Server

**Rationale:** Complete web stack (client exists, add server)
**Scope:** ~1100 lines total, 35 tests

**Features:**
- TCP server/client (listen, accept, connect, send, recv)
- UDP sockets (send_to, recv_from)
- HTTP server (routing, request/response)

**Domain Impact:**
- Web: 7/10 → **10/10**
- Distributed: 3/10 → **7/10**

### Phase 43: Bitwise Operators + Regex (FOUNDATION)

**Rationale:** Enable encoding algorithms and text processing for zero-dependency migration. Bitwise ops enable SPIR-V byte building and base64 in .flow. Regex enables log parsing, ISO8601 parsing, data validation.

**Self-hosting milestone:** Bitwise ops + regex are critical for moving stdlib to .flow.

**Revised scope** (enum/match + FFI moved to Phase 44 for proper design):
- Bitwise operators (<<, >>, &, |, ^) — for SPIR-V byte building, encoding
- Regex operations (regex_match, regex_find, regex_replace, regex_split, is_match, capture_groups)

**Deferred to Phase 44:** enum/match (needs type system design), extern FFI (needs careful design for zero dependencies)

See `docs/zero-dependency-vision.md`, `docs/ffi-design.md`

### Phase 44: Enum/Match + extern FFI + Named Args (TYPE SYSTEM + FFI)

**Rationale:** Comprehensive type system update. Enum/match enables proper state machines and error variants. extern FFI enables zero-dependency vision (Vulkan bindings in .flow). Named args improve API ergonomics.

**Self-hosting milestone:** extern FFI is the critical path to zero dependencies.

- enum type declarations + match expressions (pattern matching)
- extern "library" { fn ... } blocks (C FFI support)
- Pointer type (Value::Ptr or addresses as floats)
- Named/keyword arguments: `fn(name=value)` syntax
- Basic C struct layout support

**Crypto moved to Phase 45** (after FFI foundation is solid)

### Phases 45-46: TCP/UDP Sockets + HTTP Server

- Phase 45: TCP server/client, UDP sockets, --allow-listen
- Phase 46: HTTP server, routing, request/response

### Phases 44-55: Self-Hosting Compiler (100% .flow) — CURRENT TRACK ✅

**Strategic pivot:** User directive — make the compiler 100% OctoFlow. Rust becomes thin OS-boundary layer only (~500 lines: file I/O, sockets, GPU dispatch). All compiler logic written in .flow.

**Why 100% self-hosting beats 70% .flow / 30% Rust:**
- Proves OctoFlow is a real systems language, not just a scripting DSL
- Compiler is the largest OctoFlow program — dogfooding at maximum scale
- Enables three-compiler bootstrap to verify correctness (v1→v2→v3)

**Phase 44: eval.flow — Lexer + Meta-Interpreter (DONE — Phase 54 internal)**
- Inline lexer + token-walking evaluator for .flow programs
- Handles: let, fn, if/elif/else, while, print, push, map_set, user fns
- **Milestone:** eval.flow interprets lexer.flow → produces 1203 tokens (matches native)

**Phase 45: eval.flow Feature Completion (DONE — Phase 63 internal)**
- `for item in array` loops, `break`, `continue`, `arr[idx] = val`, `use` skip
- Phases 53-56: 19 bootstrap programs PASS; split→for-each + nested user fns + ||/contains/starts_with/ends_with in if
- Preflight fix: split/regex_split/read_lines/list_dir/capture_groups now register as arrays in all 3 pass contexts
- Phase 62: **STAGE 3** — eval.flow interprets lexer.flow (SEP delimiter fix, 28 programs PASS)
- Phase 63: **STAGE 4** — eval.flow interprets parser.flow (EVAL_PROG_PATH decoupling, while/if/elif arr[idx] op, Assign arr[idx] rhs, 29 programs PASS)
- Phase 64: **STAGE 5** — eval.flow interprets preflight.flow (char_at in if/while cond, map_has negation, float(arr[idx]), arithmetic-LHS in elif &&; 30 programs PASS)
- Phase 65: **STAGE 6 ACHIEVED** — eval.flow meta-interprets eval.flow which runs test_hello.flow → prints "hello OctoFlow"; 3-layer execution (Rust → OUTER eval.flow → INNER eval.flow → test program); root cause fix: elif condition evaluator was missing `contains()` branch (also added `starts_with`, `ends_with`, `char_at` for completeness); previous diagnosis "character-by-character lexer too complex" was incorrect — the line-based lexer worked fine, only the elif branch dispatch was incomplete
- All 895 tests pass; eval.flow now covers 99% of OctoFlow semantics
- Phase 66: **Pure OctoFlow SPIR-V Emitter** — `spirv_emit.flow` generates valid SPIR-V compute shader (output[gid]=input[gid]*2.0), validated by `spirv-val`; new builtins: `float_to_bits(f)`, `bits_to_float(n)`, `float_byte(f,idx)`, `write_bytes(path,arr)` Statement; f32 precision solved via byte-level emission; 901 tests
- Phase 67: **Full .flow-to-GPU Pipeline + GPU Test Battery** — `gpu_compute(spv_path, array_name)` builtin dispatches .flow-generated SPIR-V on the GPU; StorageBuffer storage class; 10 GLSL reference shaders (branching, nested cond, loops, large dispatch 65K, edge alignment 10 sizes, int cast, chain ops, negative values, zero handling); 3 .flow emitter validation tests (spirv-val + patched opcodes add/sub/div + cross-emitter comparison); randomized property test; key finding: GLSL sign(0.0)=0 vs Rust signum(0.0)=1; 916 tests
- Phase 68: **Structured Control Flow SPIR-V Emitters** — Credibility barrier crossed: `spirv_emit_branch.flow` emits OpSelectionMerge + OpBranchConditional + OpPhi (if/else with SSA merge, cross-validated against GLSL 02_branch.spv); `spirv_emit_loop.flow` emits OpLoopMerge + dual OpPhi (float accumulator + uint counter) + back-edge SSA + OpConvertFToU/UToF + OpSelect (branchless clamp), cross-validated against GLSL 04_loop.spv; `spirv_emit_nested.flow` nested stress test (outer loop > branch > inner loop, 5 OpPhi across 3 scopes, 10+ basic blocks, inner loop bound depends on outer counter, cross-validated against GLSL 11_nested_loop_branch.spv); `spirv_emit_param.flow` parametric emitter for 5 arithmetic ops via env vars, all spirv-val validated; status: GPU-authored (not just GPU-targeting); current approach is AST-direct lowering, no explicit CFG/SSA IR yet; 919 tests

**Phase 46: eval.flow — Remaining Builtins (~200 lines)**
- Chained string concat: `a = a + b + c` (currently only handles `a + b`)
- map_has, map_get as standalone conditions (not just in if)
- Array-returning builtins in interpreted context: read_lines, list_dir
- Math: abs, floor, ceil, min, max, sqrt, pow
- Missing string ops: trim, replace, starts_with, ends_with, upper, lower

**Phase 47: parser.flow — DONE (commit d452956)**
- Full recursive descent parser → AST as 9 parallel arrays (nd_kind, nd_lhs, nd_op, nd_rhs, nd_rhs2, nd_child, nd_next, nd_line, nd_mut)
- Inline lexer + iterative DFS dump; block tracking via push/pop stacks
- Three bugs fixed: elif_pnd post-pop access; nd_rhs2 DFS overloading; len() byte vs char_at() Unicode char mismatch
- **Milestones:** lexer.flow=152 nodes ✓, eval.flow=1547 nodes ✓, self-parse=768 nodes ✓

**Phase 48: preflight.flow — DONE (commit 01a09e9)**
- Inline lexer (copied from parser.flow), 4796 tokens on itself, 12275 on eval.flow
- Map-based symbol table (sym_kind_map/sym_line_map/sym_used_map) — avoids OctoFlow snapshot array limitation in fn bodies
- Checks: E001 (undefined assign), E002 (immutability), E003 (undefined fn call), D002 (unused var), D003 (unused fn)
- `_`-prefix exemption for D002 (intentionally unused convention)
- String template scanning: `{ident}` inside format strings marks vars as used
- Self-test: preflight.flow on itself → OK, no issues
- Raises while/for iteration limit from 10K → 1M (real programs need it; test uses max_iters=100 override)

**Phase 49: codegen.flow — DONE (commit pending)**
- OctoFlow → GLSL compute shader transpiler (~540 lines pure OctoFlow)
- Inline lexer + pipeline pattern extractor (stream → map kernel detection)
- Output: `.comp` GLSL file compilable with `glslc`/`glslangValidator` → `.spv`
- Handles: arithmetic ops, math builtins (exp/sqrt/sin/cos/...), scalar constants
- Two-input kernels supported (fn x, y: expr → two input buffer bindings)
- `chr(10)` pattern: OctoFlow has no escape sequences → use `let NL = chr(10)`
- Preflight-clean: preflight.flow reports 0 issues on codegen.flow
- Examples: gpu_double.flow → doubles each element; gpu_sigmoid.flow → sigmoid activation

**Phase 50: Three-Compiler Bootstrap — DONE (commit pending)**
- bootstrap.flow: orchestrates v1 (Rust runtime) vs v2 (eval.flow) output comparison
- 7/7 tests pass: test_hello, test_fib, test_foreach, test_fn + parser/preflight/codegen self-tests
- BOOTSTRAP VERDICT: VERIFIED — eval.flow produces identical output to Rust runtime
- Key fix: OctoFlow's `"\n"` is literal backslash-n; use `chr(10)` = NL for real newlines
- Key fix: `split(str, NL)` splits on real newline chr(10), then `trim()` removes CRLF \r
- **SELF-HOSTING MILESTONE**: OctoFlow compiler written in OctoFlow is correct

**Phase 51: Rust OS-Boundary Reduction (~500 lines target)**
- Strip Rust compiler to: file I/O, socket I/O, GPU dispatch, process spawn
- Remove: parser, preflight, lint, codegen from Rust (replaced by .flow)
- Result: `octoflow` binary is a tiny loader + runtime; brain is 100% .flow

**Phase 52: Python+CUDA vs OctoFlow BENCHMARK (GRAND MILESTONE)**
- Head-to-head benchmark: Python+CUDA vs OctoFlow for GPU compute tasks
- Test cases: matrix multiply, image blur, N-body, neural net layer forward pass
- Metrics: lines of code, setup time, compile time, execution time, memory
- Expected outcome: OctoFlow wins on LOC (no boilerplate), competitive on perf
- **This is the proof:** A self-hosted, GPU-native language as fast as Python+CUDA with 10x less code

### Phases 53+: AI/ML Foundation → Public Release ✅

**Rationale:** Build HyperGraphDB (Annex L) + Neural Networks in OctoFlow (Annex M) to prove the language is production-ready for AI/ML. Public release at Phase 55 when 9/14 domains are at 7-10/10 and someone can build a GNN, train it, and query a knowledge graph — all in pure .flow.

**Note:** Phases 47-49 were the zero-dependency elimination sprint (regex NFA, image codec, Vulkan raw bindings). AI/ML phases renumbered 50-55.

**See:** `docs/annex-l-hypergraphdb.md`, `docs/annex-m-neural-networks.md`, `docs/annex-n-octoparallel.md`, `docs/domain-foundation-map.md`

**The Vision:**
- In Python: SQLite (simple DB) + PyTorch (neural networks)
- In OctoFlow: OctoDB (Annex R) + HyperGraphDB (hypergraph DB + neural networks unified)

#### Phase 50: Self-Hosting Foundation ✅ COMPLETE
- ✅ `lexer.flow` (212 lines): Full tokenizer — 25 keywords, 2-char operators, string/number/ident scanning
- ✅ `eval.flow` (7,964 lines): Tree-walking interpreter — 80+ builtins, module imports, FFI, arrays, maps, all control flow
- ✅ `ord(c)` / `chr(n)` builtins for character comparison without escape sequences
- ✅ Key rule: top-level mutable cursor state; functions get snapshot (all position state must be inline)

**Self-hosting:** Tokenizer + interpreter in `stdlib/compiler/` (~8,200 lines)

#### Phase 47-49: Parser + Preflight + Codegen ✅ COMPLETE
- ✅ `parser.flow` (927 lines): Inline lexer + recursive descent parser → 9 parallel nd_* arrays
- ✅ `preflight.flow` (763 lines): Semantic analysis — E001 (undefined var), E002 (immutability), E003 (undefined fn), D002/D003 diagnostics
- ✅ `codegen.flow` (504 lines): GLSL compute shader emission from stream/map patterns

**Self-hosting:** Full front-end (lex → parse → analyze → codegen) in `stdlib/compiler/` (~10,400 lines)

#### Phase 53-65: Bootstrap Verification ✅ COMPLETE — STAGE 6 ACHIEVED
- ✅ `bootstrap.flow` (395 lines): 25 test programs + 4 toolchain self-tests
- ✅ Stage 3: eval.flow interprets lexer.flow → 1,203 tokens match native (Phase 62)
- ✅ Stage 4: eval.flow interprets parser.flow → AST generation correct (Phase 63)
- ✅ Stage 5: eval.flow interprets preflight.flow → static analysis works (Phase 64)
- ✅ **Stage 6: eval.flow meta-interprets eval.flow → 3-layer execution verified** (Phase 65)
- ✅ 7 Rust integration tests: comprehensive, hardening, meta, nested, selfhost patterns (Phase 73)

**Self-hosting MILESTONE ACHIEVED:** Three-layer meta-circular interpretation verified.
Compiler is **48% .flow** (22,128 lines) vs **52% Rust** (24,298 lines — OS boundary only).

#### Phases 50-55 AI/ML Stdlib ✅ DELIVERED VIA STDLIB MODULES

The original roadmap planned these as phases 50-55. They were delivered as stdlib modules:

- ✅ **Graph algorithms:** `stdlib/collections/graph.flow` — graph_create, add_node/edge, BFS, neighbors, weight
- ✅ **Matrix operations:** `stdlib/ml/linalg.flow` — mat_create, mat_mul, mat_transpose, mat_inverse, outer_product, determinant
- ✅ **Neural network layers:** `stdlib/ml/nn.flow` — dense_forward, relu, sigmoid, tanh, softmax, cross_entropy, SGD, batch_norm, dropout
- ✅ **ML preprocessing:** `stdlib/ml/preprocess.flow` — train_test_split, shuffle, minmax/zscore scaling, label encoding, imputation
- ✅ **ML metrics:** `stdlib/ml/metrics.flow` — accuracy, precision, recall, F1, MSE, RMSE, MAE, confusion matrix
- ✅ **Regression:** `stdlib/ml/regression.flow` — linear, ridge, gradient descent, R², predictions
- ✅ **Classification:** `stdlib/ml/classify.flow` — KNN, logistic regression, naive Bayes
- ✅ **Clustering:** `stdlib/ml/cluster.flow` — K-means, euclidean/manhattan distance, silhouette score
- ✅ **Statistics:** `stdlib/stats/` — descriptive, correlation, distribution, hypothesis testing, risk, timeseries (14 files)
- ✅ **Science:** `stdlib/science/` — calculus, constants, interpolation, matrix, optimize, physics, signal (7 files)

**1,045 tests passing.** 70+ stdlib modules across 15 domains.

### Phases 69-82: GPU-Native GPL — Full Stack ✅ COMPLETE

**Strategic pivot (Phase 68):** OctoFlow's identity is a **GPU-native general-purpose language**.
The GPU is the primary execution target. CPU is optional (I/O only). Rust handles OS boundary only.
See `docs/gpu-native-strategy.md` for full analysis.

**Achieved:**
- 5 .flow SPIR-V emitters (including nested loops+branches, 5,706 lines)
- 22,128 lines of .flow self-hosting compiler
- IR foundation with SSA, Phi nodes, and automated SPIR-V emission
- End-to-end: .flow source → kparse → lower → ir → SPIR-V → GPU dispatch
- Deferred dispatch: batched command buffers, 1 fence per chain
- GPU-resident buffers: zero PCIe between chained operations
- 1,045 tests, 0 failures

#### Phase 69: IR Foundation — CFG + SSA in .flow (~929 lines) ✅ COMPLETE
- ✅ IR data structures: basic blocks as parallel arrays, SSA instructions, Phi nodes (`stdlib/compiler/ir.flow`)
- ✅ IR → SPIR-V automated byte emission (replaces hand-crafted emitters) — `ir_emit_spirv()` builder API
- ✅ Test: hand-build IR for `output[gid] = input[gid] * 2.0 + 1.0` — spirv-val PASS + GPU dispatch (167 words, bound=30)
- ✅ Test: hand-build IR with loop (sum 0..9 = 45.0) — OpLoopMerge + OpPhi + multi-block CFG — spirv-val PASS + GPU dispatch (212 words, bound=38)
- ✅ Language additions: ExprStmt (bare function calls), function scope inheritance, module mutability import
- ✅ **Gate:** spirv-val PASS + GPU dispatch correct for BOTH tests (linear + loop)

#### Phase 70: AST → IR Lowering ✅ COMPLETE
- ✅ Expression parser with operator precedence (recursive descent, array-based shared state)
- ✅ Symbol table: variable name → IR instruction index (SSA references)
- ✅ Statement lowering: Let, Assign, ArrSet (output[gid]), If/Else → OpSelectionMerge
- ✅ Preflight: imported `let mut` arrays/scalars now properly tracked as mutable
- ✅ Key insight: function scope snapshot semantics for scalars → use arrays for shared mutable state
- ✅ Test: expression parser `input[gid]*3.0+1.0` → spirv-val PASS + GPU dispatch
- ✅ Test: variable references `let x; let y=x*2; let z=y+x; output=z` → spirv-val PASS + GPU dispatch
- ✅ Test: if/else branch `if x>0.5 then x*2 else 0` → OpSelectionMerge → spirv-val PASS + GPU dispatch
- ✅ Test: while loop `sum=0; i=0; while i<10: sum+=i; i+=1; output=sum` → OpLoopMerge + OpPhi → spirv-val PASS + GPU dispatch (45.0)
- ✅ **Gate:** arithmetic + branch + loop programs compile and run correctly (4 tests, all GPU-verified)

#### Phase 71: Full Automated Pipeline ✅ COMPLETE (~810 lines)
- ✅ `kparse.flow` (~330 lines): Kernel-subset parser — tokenizes source text, builds nd_* AST
  - Tokenizer: number/ident/keyword/operator scanning with array-based position state
  - Parser: recursive descent — let/mut, assign, output[gid]=, if/elif/else/end, while/end
  - Block stack for nested control flow (push/pop parallel arrays)
  - Entry point: `kparse_source(src_text)` → nd_* arrays ready for `lower_to_ir()`
- ✅ End-to-end pipeline: .flow source text → kparse → lower → ir → SPIR-V binary → GPU dispatch
- ✅ 3 GPU-verified compile tests: arithmetic, if/else branching, while loop
- ✅ Fixed tokenizer bug: replaced broken force-exit position hack with `break`
- ✅ **MILESTONE:** `compile_and_run.flow` — single .flow file compiles kernel source AND dispatches on GPU
- ✅ `compile_and_run_loop.flow` — while-loop kernel compiled + dispatched from .flow (256 elements verified)
- ✅ **Gate:** 5 end-to-end tests pass (3 compile + 2 compile+run), 930 total tests, 0 failures

#### Phase 72a: Raw Memory Builtins + FFI Fixes ✅ COMPLETE (~310 lines impl + 200 lines tests)
- ✅ MEM_TABLE handle system: thread-local `Vec<Option<MemBlock>>`, 16-byte aligned, arena cleanup
- ✅ 10 mem_* builtins: `mem_alloc`, `mem_free`, `mem_size`, `mem_set_u32`, `mem_set_f32`, `mem_set_ptr`, `mem_get_u32`, `mem_get_f32`, `mem_get_ptr`, `mem_copy`
- ✅ FFI arg encoding fix: type-aware encoding (integer cast for u32/i32/etc, MEM_TABLE lookup for ptr/handle)
- ✅ FFI return value fix: `"ptr"/"handle"` returns handle (not null-terminated string read)
- ✅ Extend call_fn_ptr from 4 to 8 args (Vulkan functions need up to 8)
- ✅ All builtins require `--allow-ffi`, bounds-checked, `write_unaligned`/`read_unaligned`
- ✅ Gate: 943 tests (731+19+139+11+43), 0 failures, 13 new mem_* tests

#### Phase 72b: Vulkan Dispatch in .flow ✅ COMPLETE (~980 lines)
- ✅ `stdlib/gpu/vk.flow`: Vulkan FFI module — 40 extern declarations, 19 VK constants, utility helpers
- ✅ `stdlib/gpu/test_vk_dispatch.flow`: Full 20-step Vulkan compute dispatch test (480 lines)
- ✅ 5 new builtins: `mem_set_u8`, `mem_get_u8`, `mem_set_u64`, `mem_get_u64`, `mem_from_str`
- ✅ `read_bytes(path)` array-returning builtin for loading .spv files
- ✅ LIB_CACHE: thread-local DLL handle cache (avoids load/free per FFI call)
- ✅ `bit_and`, `bit_or`, `bit_test` Rust builtins (required for memory type flag checking)
- ✅ ExternBlock handling in `import_module` + `collect_module_fns` (module system fix)
- ✅ Pure .flow dispatches 01_double.spv on NVIDIA GTX 1660 SUPER, verifies 256 elements
- ✅ **Gate:** 948 tests (735+19+139+11+44), 0 failures; .flow program dispatches SPIR-V on GPU without Rust in call path

#### Phase 75a: GPU Compute Builtins — Tier 1 (Identity) COMPLETE

The builtins that define OctoFlow as a GPU-native language.

**Element-wise ops** (new binop SPIR-V pattern + existing MapOp):
- mat_mul(a, b, m, n, k) — tiled matrix multiply (16x16 workgroups) DONE
- gpu_add(a, b), gpu_sub(a, b), gpu_mul(a, b), gpu_div(a, b) — element-wise binary
- gpu_scale(arr, s) — scalar multiply every element
- gpu_abs(arr), gpu_sqrt(arr), gpu_exp(arr), gpu_log(arr) — element-wise math
- gpu_clamp(arr, lo, hi) — element-wise clamp
- gpu_where(cond, a, b) — conditional select (new SPIR-V)

**Reductions** (reuse ReduceOp):
- gpu_sum(arr), gpu_min(arr), gpu_max(arr), gpu_mean(arr)

**Scans** (reuse scan SPIR-V):
- gpu_cumsum(arr) — prefix sum

**Linear algebra:**
- dot(a, b) — inner product (map+reduce)
- mat_transpose(a, rows, cols) — matrix transpose
- norm(a), normalize(a) — L2 norm / unit vector

**Utility:**
- gpu_fill(val, size), gpu_range(start, end, step), gpu_reverse(arr)
- gpu_info() — GPU device properties (returns map)

**Gate:** All Tier 1 builtins working with GPU dispatch (CPU for no-GPU targets) ✅

#### Phase 76: gpu_run — Universal GPU Dispatch COMPLETE

The last significant Rust addition for GPU compute. After this, every new GPU operation is a .flow module + .spv kernel.

**Core:**
- `gpu_run(spv_path, arr1, arr2, ..., scalar_params...)` — universal Vulkan compute dispatch
- Pipeline caching: per (device, spirv_hash, binding_count, push_constant_size) tuple
- Push constant support: trailing scalar args passed as push constants to shader
- Output buffer: always the last binding (N = num_inputs)
- Auto workgroup calculation: ceil(output_size / 256) for 1D

**Pre-built kernels** (stdlib/gpu/kernels/):
- abs.spv, sqrt.spv, exp.spv, log.spv, negate.spv, floor.spv, ceil.spv, round.spv
- sin.spv, cos.spv, double.spv
- add.spv, sub.spv, mul.spv, div.spv
- where.spv (conditional select)
- reduce_sum.spv, reduce_min.spv, reduce_max.spv

**.flow wrappers** (stdlib/gpu/ops.flow): Architecture C — .flow functions calling gpu_run()
- gpu_abs_run, gpu_sqrt_run, gpu_exp_run, ... (suffixed _run to avoid name collision with Rust builtins)
- gpu_add_run, gpu_sub_run, gpu_mul_run, gpu_div_run
- gpu_where_run

**Gate:** 975 tests (768+19+139+16+52), 0 failures; gpu_run dispatches any .spv kernel from .flow

#### Phase 75b: GPU Compute Builtins — Tier 2 (Utility) — FUTURE

Data movement + domain-specific GPU ops. Available via `gpu_run()` + custom .spv kernels today. Native builtins would improve ergonomics.

**Candidates:**
- More reductions: gpu_product, gpu_any, gpu_all, gpu_argmin, gpu_argmax, gpu_variance, gpu_stddev
- More scans: gpu_cumprod, gpu_cummax, gpu_cummin
- Data movement: gpu_filter, gpu_scatter, gpu_gather, gpu_concat
- Sorting: gpu_sort (bitonic/radix), gpu_argsort
- Signal/DSP: gpu_fft, gpu_ifft, gpu_convolve, gpu_convolve2d
- Threading: spawn/join (CPU OS boundary)

#### Phase 75c: GPU Compute Builtins — Tier 3 (Ecosystem) — FUTURE

Advanced builtins for mature GPU programming: mat_inverse, mat_det, gpu_unique, gpu_topk, gpu_resize, gpu_blur, gpu_edge_detect, gpu_shuffle.

#### Phase 77: eval.flow Builtins — Launch Runtime COMPLETE

45+ builtins implemented in eval.flow meta-interpreter (+401 lines, 6975 total):

**Statistics**: mean, median, variance (sample), stddev, quantile, correlation
**Array queries**: find, first, last, min_val, max_val, slice, range_array
**Path utilities**: file_ext, file_name, file_dir, dirname, basename, path_join, file_exists, file_size, is_directory
**Linear algebra**: dot, norm, normalize, mat_transpose
**Encoding**: hex_encode, hex_decode, base64_encode, base64_decode
**Date arithmetic**: add_seconds, add_minutes, add_hours, add_days, diff_seconds, diff_hours, diff_days
**I/O**: append_file, write_bytes, json_parse, json_parse_array
**Regex**: capture_groups, regex_split

Also added to ncall_fn (nested calls) and afn (arithmetic context) dispatch chains.

**Preflight fixes**: array-returning function detection for slice/range_array; print validation now handles `{arr[i]}` syntax.

**Test count**: 975 (768 CLI + 139 parser + 16 SPIR-V + 52 GPU)

#### Phase 78: GPU-Native Array Storage — Zero-Copy Dispatch Path (~250 lines) ✅ COMPLETE

**Priority: LAUNCH-BLOCKING.** A GPU-native language cannot be 8x slower than its own dispatch layer in its own domain. The interpreter's Value enum (56 bytes/element) forces 560MB allocation for what should be 40MB of f32 data. This is the single biggest performance gap.

**Problem:** `gpu_add(a, b)` on 10M floats takes 230ms end-to-end but only 30ms at the Rust dispatch level. The 200ms gap:
- `extract_array_arg`: Vec<Value> → Vec<f32> conversion (~30ms, iterates 10M values)
- `gpu_fill` / result wrapping: 10M × Value::Float construction (~100-150ms, 560MB alloc)
- These dominate even after HOST_CACHED readback fix (183ms → 6ms) and BufferSpec zero-copy

**Solution:** GPU-typed array storage alongside the existing Vec<Value> arrays:
- `gpu_arrays: HashMap<String, Vec<f32>>` in the interpreter — stores raw f32 data
- `gpu_fill()`, `gpu_add()`, `gpu_mul()`, etc. produce and consume gpu_arrays directly
- No Value enum wrapping/unwrapping on the GPU path — zero-copy dispatch
- `len()`, `print()`, array indexing transparently check gpu_arrays first
- Conversion only happens when a gpu_array is used in a non-GPU context (lazy materialization)

**Implementation:**
- Add `gpu_arrays: HashMap<String, Vec<f32>>` to interpreter state (compiler.rs)
- `gpu_fill(val, n)` → stores directly into gpu_arrays (skip Vec<Value>)
- `gpu_add/sub/mul/div(a, b)` → reads from gpu_arrays, dispatches, stores result in gpu_arrays
- `extract_array_arg()` checks gpu_arrays first → zero-copy &[f32] reference
- Array indexing `arr[i]` checks gpu_arrays, returns Value::Float on single access
- `len(arr)` checks gpu_arrays
- `print("{arr[i]}")` works transparently
- Only materialize to Vec<Value> when array enters non-GPU operations (push, for-each, etc.)

**Expected result:** 10M gpu_add end-to-end: 230ms → ~35ms (matching Rust dispatch layer)

**Gate:** `examples/benchmark_gpu_10m.flow` shows < 50ms for gpu_add 10M. NumPy parity or better.

**Result:** gpu_add 10M: 46ms (gate: <50ms ✅). gpu_fill 10M: 7ms. 5x speedup from Phase 77.
Thread-local GPU_ARRAYS + ArrayResult enum + lazy materialization. 978 tests pass.

#### Phase 79: GPU-Resident Buffer Storage — Zero PCIe Chaining COMPLETE

Data born on GPU stays on GPU. `GpuBuffer` (owned VkBuffer+VkDeviceMemory) and `GpuBufferRef`
(Copy handle) types. `GpuArrayStorage::Cpu|Resident` enum in GPU_ARRAYS. Chained operations
(gpu_fill → gpu_add → gpu_mul) bind directly from VRAM — zero PCIe between ops.
`download_buffer_fast()` uses HOST_CACHED staging for 20 GB/s CPU reads (vs 200 MB/s direct).

- dispatch_resident() — binds existing VkBuffers, allocates output, single fence
- Resident fast-paths for all binary/unary/select GPU ops
- gpu_fill/gpu_range auto-upload to Resident when GPU available
- **Gate:** 10M 4-step chain <100ms ✅ (95ms actual, down from 260ms)
- 963 tests pass

#### Phase 80: GPU Filesystem — Binary + CSV I/O COMPLETE

GPU-native file I/O: data moves between disk and VRAM without CPU array intermediation.

- `gpu_load_binary(path)` → GpuBuffer (raw f32, ~890 MB/s read)
- `gpu_save_binary(arr, path)` → raw f32 to disk (~625 MB/s write)
- `gpu_load_csv(path)` → GpuBuffer (text parse, ~100 MB/s)
- `gpu_save_csv(arr, path)` → text to disk (~28 MB/s)
- **Gate:** 10M save+load roundtrip <200ms ✅ (109ms binary actual)
- 970 tests pass

#### Phase 81: GPU Advanced Builtins — Random, MatMul, EMA COMPLETE

Three high-value GPU compute primitives for data science and ML workflows.

- `gpu_random(n, lo, hi)` — GPU-parallel RNG (lowbias32 hash, well-distributed)
- `gpu_matmul(a, b, m, n, k)` — matrix multiplication (naive, workgroup-per-row)
- `gpu_ema(arr, alpha)` — exponential moving average (sequential scan on GPU)
- **Gate:** All three correct + 10M random in <50ms ✅
- 969 tests pass

#### Phase 82: Public Release Prep ✅ COMPLETE

Repurposed from original I/O-in-.flow plan to public release preparation.

- ✅ **Phase 82a:** Deferred GPU dispatch (batched command buffers, 1 fence per chain), REPL polish, binary rename `octoflow`, 70+ stdlib modules (1,045 tests)
- ✅ **Phase 82b:** REPL polish, CLI help system, auto-timing, working examples
- ✅ **Phase 82c:** Complete reference documentation (CODING-GUIDE.md, stdlib-signatures.md)
- ✅ **Phase 82d:** GPU Showcase — Mandelbrot zoom in ANSI truecolor (realtime animation)
- ✅ **Phase 82e:** GPU fractal PPM image, 10K CSV benchmark, LLM docs (5-file context package), recording scripts

**Gate:** Public-facing docs, demos, and tooling ready. 1,045 tests, 0 failures.

---

#### Phase 83e: Native Video Decoders + JPEG Chroma Fix ✅ COMPLETE

Native media decoding pipeline — zero external dependencies, no ffmpeg.

- **JPEG chroma subsampling fix:** `decode_ecs()` handles 4:2:0, 4:2:2, 4:4:4 subsampling with per-component buffers and proper upsampling. DRI marker parsing added. RST marker range corrected to 0xD0-0xD7. NASA GIBS satellite tiles now decode correctly.
- **GIF decoder:** ~250 lines pure Rust. LZW decompression (variable-width codes up to 12 bits), GIF87a/GIF89a, frame compositing with disposal methods (0-3), interlace support, transparency via Graphics Control Extension. Returns `GifData { width, height, frames: Vec<GifFrame> }`.
- **AVI/MJPEG parser:** ~150 lines pure Rust. RIFF chunk walker, AVI header parsing (avih, strl, strf), frame offset extraction (00dc/00db chunks). Each frame decoded via existing JPEG decoder.
- **video_open/video_frame builtins:** `video_open(byte_array)` → vid.width, vid.height, vid.frames, vid.fps (handle-based). `video_frame(handle, index)` → f.r, f.g, f.b (GPU arrays). Auto-detects GIF/AVI format. GIF: eager decode; AVI: lazy decode. 7 LetDecl sites + 4 preflight sites each. Thread-local VIDEO_HANDLES storage.
- **showcase3.flow rewritten:** Native GIF decoder, downloads animated octopus GIF from Giphy, GPU-enhanced playback with contrast boost + ocean blue tint, frame timing from GIF delay values.

**Files:** `image_io.rs` (JPEG fix + gif/avi modules), `compiler.rs` (7 LetDecl sites each), `preflight.rs` (4 sites each)

---

#### Phase 83: Public Release v0.83.1 ✅ SHIPPED

OctoFlow is publicly available.

- **Public repo:** https://github.com/octoflow-lang/octoflow (51 stdlib modules, 19 examples, docs)
- **GitHub Pages:** https://octoflow-lang.github.io/octoflow/ (landing page, install scripts)
- **Install scripts:** PowerShell one-liner (Windows) + shell one-liner (Linux)
- **Binary release:** v0.83.1 Windows x64 (2.2 MB, zero deps)
- **First community interaction:** 3 issues from OpenDGPS — all resolved
- **GPU showcase examples:** population_gpu.flow, geo_track.flow, batch_decode_gpu.flow
- **Bit shift builtins:** bit_shl, bit_shr, bit_xor (foundation for .flow binary decoders)
- **SO_REUSEADDR fix:** TCP socket port rebind after Ctrl-C

**Repos:**
- `C:\FlowGPU` — private dev workspace (full compiler source, 143 stdlib modules, 1,058 tests)
- `G:\octoflow` — staging repo (clean, curated, push to GitHub from here)

**Gate:** Binary downloadable, examples run, first external user filed issues and got responses.

---

### What's Next — Version Roadmap

#### v0.84 — Terminal Graphics & Native Media

**TAGLINE:** "GPU pixels in your terminal. Native video decoding. Still 2.2 MB. Still zero dependencies."

**Ship what makes people say "holy shit" in a terminal. GUI can wait.**

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | **Linux binary** | ✅ done | Cross-compiled |
| 2 | **Windows binary** | ✅ done | v0.83.1 shipped |
| 3 | **term_image** | ✅ done | Kitty/Sixel/halfblock protocols — GPU-rendered images in terminal |
| 4 | **gif.flow + avi.flow** | ✅ done | Pure .flow decoders (self-hosting direction — less Rust) |
| 5 | **MP4/H.264 native decoder** | ✅ done | mp4.flow demuxer + h264.flow baseline I-frame decoder (42 tests) |
| 6 | **showcase.flow** | ✅ done | N-body physics simulation — GPU-computed gravity |
| 7 | **showcase2.flow** | ✅ done | NASA satellite tile — GPU image processing |
| 8 | **showcase3.flow** | ✅ done | Video playback in terminal — native GIF/AVI decode |

Each showcase creates content for weekly posts. Every item is terminal-visible.

#### v0.85 — GUI (ext.ui)

Window, canvas, input events. Unlock gaming domain (4/10 → 7/10). Infrastructure investment — not a launch moment, so it follows the terminal-visible phase.

#### v0.90 — Fractal Codec

Fractal compression with Holder Pattern (~2,100 lines). GPU-accelerated IFS search, .odc format, streaming decompression. See Phase 90-93 details below.

#### v0.95 — Homeostasis

GPU compute self-regulation (~2,200 lines). VRAM pressure, dispatch routing, thermal awareness, contention detection. See Phase 95-100 details below.

#### Beyond

- Self-hosting compiler on GPU (compiler as compute shader)
- Rust elimination — thin loader < 200 lines
- Platform products: OctoMedia, OctoMark, OctoEngine, OctoShell

**Standing priority:** Self-hosting direction. No new Rust unless OS-boundary. Future format decoders in .flow.

---

### Self-Hosting Achievement Summary ✅ VERIFIED

**The OctoFlow compiler is self-hosting.** The compiler chain — written entirely in OctoFlow — can tokenize, parse, analyze, and execute OctoFlow programs, including itself.

**Self-hosting compiler chain (22,128 lines of .flow):**

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Lexer | lexer.flow | 212 | COMPLETE |
| Parser | parser.flow | 927 | COMPLETE |
| Preflight | preflight.flow | 763 | COMPLETE |
| Evaluator | eval.flow | 7,964 | COMPLETE (80+ builtins, module imports, FFI) |
| Codegen | codegen.flow | 504 | COMPLETE (GLSL shader emission) |
| IR | ir.flow | 950 | COMPLETE (43 opcodes, SSA, CFG, Phi nodes) |
| Kernel Parser | kparse.flow | 476 | COMPLETE (GPU kernel subset) |
| AST→IR Lower | lower.flow | 479 | COMPLETE (expression + control flow) |
| SPIR-V Emitters | spirv_emit*.flow | 5,706 | COMPLETE (5 emitters, nested loops+branches) |
| Bootstrap | bootstrap.flow | 395 | COMPLETE (25 programs, 6 stages verified) |

**Bootstrap stages achieved:**
- **Stage 3:** eval.flow interprets lexer.flow → 1,203 tokens (matches native Rust)
- **Stage 4:** eval.flow interprets parser.flow → correct AST generation
- **Stage 5:** eval.flow interprets preflight.flow → static analysis works
- **Stage 6:** eval.flow meta-interprets eval.flow → **3-layer execution verified**

**Rust vs .flow split:**
- Rust (OS boundary): 24,298 lines — parser, Vulkan bindings, SPIR-V patterns, sockets
- .flow (self-hosted): 22,128 lines — compiler chain + 70+ stdlib modules
- **Ratio: 48% .flow** — approaching parity

**What Rust remains for (per std lib principle):**
- OS-boundary concerns: file I/O, sockets, Vulkan FFI, memory allocation
- These are inherently non-portable and require system-level bindings
- Not a self-hosting gap — this is the correct architecture

**End-to-end GPU compilation in .flow:**
- `compile_and_run.flow`: source text → kparse → lower → ir → SPIR-V binary → GPU dispatch
- Pure .flow program compiles a kernel and dispatches it on NVIDIA GPU hardware

---

### Future: GPU-Accelerated Compilation

#### Phase 83: Self-Hosting Compiler on GPU
- Compiler as compute shader: input = .flow source buffer, output = .spv buffer
- parse → lower → codegen all run as GPU kernels
- Three-compiler bootstrap: v1 (CPU) → v2 (GPU) → v3 (GPU), verify v2 == v3
- **Gate:** GPU-compiled .spv matches CPU-compiled .spv byte-for-byte

#### Phase 84: Rust Elimination — FINAL MILESTONE
- Rust bootstrap → thin loader (< 200 lines: read .spv, call vkDispatch)
- Eventually: pre-compiled binary replaces Rust source entirely
- **Proof:** OctoFlow is a zero-dependency, GPU-native, self-hosting GPL
- **Gate:** `cargo` is no longer needed to build or run OctoFlow

#### Phase 90–93: Fractal Compression with Holder Pattern
GPU-native compression/decompression pipeline. Holder Pattern pre-processing
separates data into layers of regularity before compression.

- **Phase 90: Holder Pattern Engine** (~800 lines)
  - Columnar separation, delta encoding, scale normalization, pattern tiling
  - GPU kernels for each Holder level (embarrassingly parallel)
  - .od data as input → Holder-decomposed layers as output
- **Phase 91: GPU Fractal Compression** (~600 lines)
  - IFS (Iterated Function System) search on GPU — find self-similar transforms
  - Holder layers compress independently (smooth layers → high ratio, noise → low ratio)
  - Adaptive: switch compression algorithm per Holder level
- **Phase 92: .odc Format** (~400 lines)
  - OctoData Compressed — binary format storing Holder metadata + compressed layers
  - Streaming decompression: GPU reads .odc → reconstructs in VRAM
  - `gpu_load_compressed("data.odc")` → GpuBuffer (decompresses on GPU)
- **Phase 93: Integration + Benchmarks** (~300 lines)
  - `gpu_save_compressed(buf, "out.odc")` / `gpu_load_compressed("in.odc")`
  - Benchmark: compression ratio vs LZ4/Zstd on structured data (.od files)
  - **Gate:** 10x compression on structured numeric data, decompress faster than disk read

#### Phase 95–100: Homeostasis — GPU Compute Self-Regulation
Runtime self-regulation system that keeps the GPU compute pipeline healthy
under varying workloads, memory pressure, and thermal conditions.

- **Phase 95: VRAM Pressure Manager** (~500 lines)
  - Track GpuBuffer allocations, total VRAM usage vs available
  - Eviction policy: LRU with pinning (active pipeline buffers never evicted)
  - Spill to HOST_VISIBLE when VRAM full, transparent re-upload on access
- **Phase 96: Dispatch Router** (~400 lines)
  - Heuristic: small arrays (< 1K elements) → CPU, large → GPU
  - Adaptive threshold based on measured dispatch overhead vs compute time
  - `gpu_auto(fn, data)` — runtime decides GPU vs CPU
- **Phase 97: Thermal Awareness** (~300 lines)
  - Query GPU temperature via Vulkan/NVML (if available)
  - Throttle dispatch rate when approaching thermal limit
  - Backpressure signal to .flow runtime (reduce batch size)
- **Phase 98: Contention Detection** (~300 lines)
  - Detect GPU queue saturation (fence wait time > threshold)
  - Multi-queue dispatch when available (compute + transfer queues)
  - Work-stealing between queues
- **Phase 99: Memory Defragmentation** (~400 lines)
  - Compact buffer pool: coalesce small freed buffers into large blocks
  - Background defrag during idle (between dispatches)
  - Metrics: fragmentation ratio, largest contiguous block
- **Phase 100: Homeostasis Dashboard** (~200 lines)
  - `gpu_health()` → map with VRAM usage, temperature, queue depth, fragmentation
  - `gpu_stats()` → cumulative dispatch count, total compute time, PCIe bytes
  - Print-friendly format for .flow scripts to self-monitor
  - **Gate:** System maintains stable performance under 1-hour stress test with varying workloads

### Post-GPU-Native: Platform Products

- **ext.ui core** (window, canvas, input) — Gaming/Media unlock
- **Video codec** (Vulkan Video) — OctoMedia full
- **Threading/async** (spawn, join, channels) — Distributed compute
- **Platform products:** OctoMedia, OctoMark, OctoEngine, OctoShell
- **Ecosystem:** Module registry, WASM target, oct:// protocol
- **AI/ML:** Graph core, sparse matrix, neural networks, autograd (Annex L/M)

### Open Source Architecture Split (Pre-Alpha milestone)

See `docs/annex-k-open-source-strategy.md` for full strategy.
- After Phase 44: refactor crate architecture for open/proprietary split
- Open: parser, preflight, lint, stdlib, SDK, docs, tests (Apache 2.0)
- Proprietary: compiler, SPIR-V codegen, Vulkan runtime, stage fusion

---

## 17. Validation Gates & Decision Points

### Gate Summary

```
Phase 0 --> Gate 0: "Can we emit valid SPIR-V and run it on GPU?"        PASSED
  |
Phase 1 --> Gate 1: "Do all 5 GPU patterns produce correct results?"     PASSED
  |
Phase 2 --> Gate 2: "Does source code -> GPU execution work end-to-end?" PASSED
  |
Phase 3 --> Gate 3: "Do multi-stage pipelines with automatic GPU/CPU?"   PASSED
  |
Phase 4 --> Gate 4: "Do real-world examples produce correct results?"    PASSED
  |         19 MapOps, fn declarations, use imports, 61 tests
  |
Phase 5 --> Gate 5: "Does pre-flight catch errors before execution?"     PASSED
  |         Preflight + range analysis + dead code lint, 82 tests
  |
Phase 6 --> Gate 6: "Can OctoFlow process real images (PNG/JPEG)?"        PASSED
  |         Image I/O, channel-aware ops (warm/cool/tint), 105 tests
  |
Phase 7 --> Gate 7: "Is there a packaged CLI for end users?"             PASSED
  |         OctoMedia CLI, 7 presets, batch mode, 117 tests
  |
Phase 8 --> Gate 8: "Does the language support conditionals/variables?"  PASSED
  |         if/then/else, comparisons, boolean logic, 132 tests
  |         Security hardening: path traversal, input limits, 141 tests
  |
Phase 9 --> Gate 9: "Can programs report computed values?"              PASSED
  |         Print interpolation, precision formatting, 147 tests
  |
Phase 10 --> Gate 10: "Do errors show where the problem is?"           PASSED
  |         Source locations in AST, line numbers in errors, 149 tests
  |
Phase 11 --> Gate 11: "Can programs be reused without editing?"        PASSED
  |         --set, -i, -o CLI overrides, 163 tests
  |
Phase 12 --> Gate 12: "Do scalar functions work in expressions?"     PASSED
  |         abs, sqrt, pow, clamp, count, 183 tests
  |
Phase 13 --> Gate 13: "Can developers iterate without restarting?"   PASSED
  |          --watch mode, file polling, import tracking, 191 tests
  |
Phase 14 --> Gate 14: "Can programs use vector math (positions, colors)?"  PASSED
  |          vec2/vec3/vec4, scalar-decomposed components, dotted refs, 207 tests
  |
Phase 15 --> Gate 15: "Can programs define entity types with named fields?"  PASSED
  |          user-defined struct types, field access, lint, 218 tests
  |
Phase 16 --> Gate 16: "Can programs use arrays with indexing and len()?"  PASSED
  |          array literals, index access, len(), 230 tests
  |
Phase 17 --> Gate 17: "Can programs use mutable variables?"  PASSED
  |          let mut, variable reassignment, accumulator pattern, 242 tests
  |
Phase 18 --> Gate 18: "Can programs use GPU-friendly binary storage?"  PASSED
  |          .octo format, raw + delta encoding, multi-column, 253 tests
  |
Phase 19 --> Gate 19: "Can programs use iterative loops?"  PASSED
  |          while loops, safety limit, mutable accumulation, 268 tests
  |
Phase 20 --> Gate 20: "Can programs iterate a known number of times?"  PASSED
             for loops, range(), Python-style exclusive, 283 tests
  |
Phase 21 --> Gate 21: "Can loops be nested inside other loops?"  PASSED
  |          recursive body helpers, arbitrary depth nesting, 292 tests
  |
Phase 22 --> Gate 22: "Can programs exit loops early or skip iterations?"  PASSED
  |          break/continue, LoopControl signal propagation, nested-safe, 309 tests
  |
Phase 23 --> Gate 23: "Can programs conditionally execute blocks of statements?"  PASSED
  |          if/elif/else/end blocks, break/continue propagation, nested-safe, 328 tests
  |
Phase 24 --> Gate 24: "Can programs define and call user-written functions?"  PASSED
  |          fn/return/end, local scope, fn-calls-fn, loop/if in body, 336 tests
  |
Phase 25 --> Gate 25: "Can programs generate random numbers?"  PASSED
  |          random(), xorshift64*, deterministic seeding, --set seed=N, 351 tests
  |
Phase 26 --> Gate 26: "Can programs manipulate string values?"  PASSED
  |          Value enum (f32+String), concat, len, contains, ==, !=, strict typing, 363 tests
  |
Phase 26b --> Gate 26b: "Can users interactively explore the language?"  PASSED
  |          REPL, persistent context, multi-line, auto-print, :help/:vars/:fns, 376 tests
  |
Phase 27 --> Gate 27: "Can modules share all definition types?"  PASSED
  |           use imports scalar fns, structs, constants, arrays; dual-name, dotted calls, 401 tests
  |
Phase 28 --> Gate 28: "Can code iterate over array elements?"    PASSED
  |           for x in arr ... end; break/continue; nested; 425 tests
  |
Phase 29 --> Gate 29: "Can code mutate arrays at runtime?"       PASSED
  |           arr[i] = val, push(arr, val), pop(arr); mutable tracking; 452 tests
  |
Phase 30a -> Gate 30a: "Can .flow write its own algorithms?"    PASSED
  |           array params, stdlib math/array/sort; self-hosting milestone; 471 tests
  |
Phase 30b -> Gate 30b: "Can programs use key-value data structures?" PASSED
  |           map/map_set/map_get/map_has/map_remove/map_keys; 496 tests
  |
Phase 31 --> Gate 31: "Can programs read/write files securely?"    PASSED
  |           read/write/append_file, read_lines, list_dir, split, path utils; --allow-read/--allow-write; 530 tests
  |
Phase 32 --> Gate 32: "Can programs convert types and manipulate strings?"  PASSED
  |           str/float/int, substr, replace, trim, upper/lower, starts/ends_with, index_of, char_at, repeat; 561 tests
  |
Phase 33 --> Gate 33: "Can programs fully manipulate arrays?"              PASSED
              join, find, first/last, reverse, slice, sort, unique, range_array, sum/min/max/count on arrays; 605 tests
  |
Phase 34 --> Gate 34: "Can programs handle errors gracefully?"              PASSED
  |           try() → .ok/.value/.error decomposition, LetDecl-level; 629 tests
  |
Phase 35 --> Gate 35: "Can programs make HTTP requests?"                   PASSED
  |           http_get/post/put/delete → .status/.body/.ok/.error, --allow-net security; 657 tests
  |
Phase 36 --> Gate 36: "Can programs parse and generate JSON?"              PASSED
              json_parse → hashmap (dot-notation flatten), json_parse_array → array, json_stringify → string; 679 tests
  |
Phase 37 --> Gate 37: "Can programs access environment and persist config data?"  PASSED
              time(), env(), os_name(); load_data/save_data .od format; 701 tests
  |
Phase 38 --> Gate 38: "Can programs use functional transforms and hashmap bracket access?"  PASSED
              filter/map_each/sort_by/reduce with inline lambdas; map["key"]; 737 tests
  |
Phase 39 --> Gate 39: "Can programs process structured CSV data with headers?"  PASSED
              Value::Map, read_csv → array of maps, write_csv, row["field"] access; 765 tests
  |
Phase 40 --> Gate 40: "Can programs execute external commands securely?"          PASSED
              exec(cmd, ...args) → .status/.output/.ok/.error, --allow-exec flag; 777 tests
  |
Phase 41-42 -> Gate 42: "Bitwise ops + regex?"                                  PASSED
              bit_and/or/test, regex_match/replace/find_all; 822 tests
  |
Phase 43-50 -> Gate 50: "Is the compiler self-hosting?"                         PASSED
              eval.flow = Rust runtime, 7/7 integration tests; Stage 6 (3-layer); 901 tests
  |
Phase 51-52 -> Gate 52: "Does OctoFlow beat Python+CUDA?"                       PASSED
              6-line sigmoid pipeline benchmark, LOC comparison; OS audit 74% replaceable
  |
Phase 53-65 -> Gate 65: "Does eval.flow meta-interpret itself?"                 PASSED
              Stage 6 — 3-layer execution verified; 22,128 lines .flow compiler
  |
Phase 66-68 -> Gate 68: "Can .flow emit SPIR-V and dispatch on GPU?"            PASSED
              spirv_emit.flow → spirv-val → GPU dispatch; structured loops+selection
  |
Phase 69-76 -> Gate 76: "Is GPU compute universally accessible from .flow?"     PASSED
              gpu_run, IR builder, CFG+SSA, raw Vulkan FFI from .flow; 1,005 tests
  |
Phase 77-82 -> Gate 82: "Is OctoFlow ready for public release?"                 PASSED
              Deferred dispatch, REPL, 143 stdlib, docs, examples; 1,045 tests
  |
Phase 83e --> Gate 83e: "Native media decoders + public release shipped?"        PASSED
              GIF/AVI/JPEG decoders, video builtins, v0.83.1 on GitHub; 1,058 tests
```

### Decisions Made

| Phase | Decision | Chosen |
|-------|----------|--------|
| 0 | SPIR-V emission approach | Byte-level construction (no external libs) |
| 1 | Workgroup size strategy | Fixed 256 |
| 2 | Parser approach | Hand-written recursive descent |
| 3 | Fusion strategy | Always fuse adjacent GPU stages |
| 4 | Float tolerance | 1e-6 relative error |
| 5 | Pre-flight strictness | Block on errors, warn on range/lint |
| 6 | Image format | Flat interleaved RGB f32 [0-255], image crate |
| 7 | CLI packaging | Two binaries in one crate via lib.rs |
| 8 | Operator precedence | 6-level hierarchy, non-chainable comparisons |
| 8 | Float equality | 1e-6 tolerance for == and != |
| 8 | Security model | Reject `..` traversal, allow absolute paths (CLI needs them) |
| 9 | Interpolation syntax | `{name}` and `{name:.N}`, `{{`/`}}` escape (Rust/Python convention) |
| 10 | Span representation | `(Statement, Span)` tuples — minimal, idiomatic, destructures cleanly |
| 11 | Override scope | `--set` skips eval entirely (no side effects from overridden expression) |
| 11 | Path override granularity | `-i`/`-o` override ALL tap/emit paths (simple programs, matches OctoMedia pattern) |
| 12 | Reduce vs FnCall parsing | Whitelist reduce ops (`min/max/sum/count`), everything else parses as FnCall |
| 13 | File watching approach | Polling via `std::fs::metadata` (500ms) — zero dependencies, no `notify` crate |
| 13 | Watch scope | Main file + `use`-imported modules (parses AST to find imports) |
| 14 | Vec type representation | Scalar-decomposed — `vec3(1,2,3)` creates 3 f32 scalars `v.x`/`v.y`/`v.z` (no polymorphic types) |
| 14 | Vec AST approach | Reuse `ScalarExpr::FnCall` + `ScalarExpr::Ref("v.x")` — zero new AST variants |
| 15 | Struct representation | Scalar-decomposed like vec — `Entity(1,2,3)` creates `e.x`, `e.y`, `e.health` |
| 15 | Struct syntax | `struct Name(field1, field2)` — parenthesized field list, positional constructors |
| 16 | Array storage | Separate `HashMap<String, Vec<f32>>` — enables dynamic indexing and `len()`, unlike scalar-decomposed vec/struct |
| 16 | Index type | f32 truncated to usize — consistent with all-f32 scalar system |
| 17 | Mutability model | Opt-in via `let mut` — immutable by default, prevents accidental modification |
| 17 | Assignment parsing | Lookahead: `Ident` + `=` (not `==`) triggers assignment — clean disambiguation from equality |
| 18 | Storage approach | File format (not database) — .octo binary columnar, foundation for future OctoDB |
| 18 | Data alignment | 16-byte aligned column data — compatible with Vulkan buffer requirements |
| 18 | Default encoding | Raw f32 LE for emit() — zero overhead, GPU-ready; delta available via API |
| 19 | Loop execution model | CPU-only — while loops are inherently sequential (depend on mutable state) |
| 19 | Block terminator | `end` keyword (not braces/indentation) — matches if/then/else, LLM-friendly |
| 19 | Safety limit | 10,000 max iterations — prevents infinite loops, restructure for larger workloads |
| 20 | Range semantics | Python-style exclusive upper bound — `range(0, n)` produces n iterations (0..n-1) |
| 20 | Loop variable type | Evaluated as i64 for clean integer iteration, inserted as f32 into scalars |
| 20 | Loop variable scope | Inserted each iteration, removed after loop — not accessible after end |
| 21 | Nested loop approach | Recursive helper extraction — shared execute/validate/lint/plan functions |
| 22 | Loop control flow model | LoopControl enum signal propagation — execute_loop_body returns Normal/Break/Continue, caller matches |
| 23 | Dual if-form design | Expression-form (`if/then/else` inline) vs block-form (`if/elif/else/end` multi-statement), context-disambiguated |
| 24 | Scalar fn execution | Reuse `execute_loop_body` + `LoopControl::Return(f32)` — no duplicated statement handling |
| 24 | Fn disambiguation | `:` after params → pipeline fn (FnDecl); newline → scalar fn body until `end` (ScalarFnDecl) |
| 26b | REPL architecture | Context struct wrapping execute() locals + parse/parse_expr fallback for bare expressions |
| 26b | Multi-line detection | Count fn/if/while/for vs end, unmatched `[`, unclosed `"` — simple heuristic, no parser involvement |
| 27 | Module constant evaluation | Module-local context — constants can only reference literals and prior module-level constants, no access to importer state |
| 27 | Module mutability | All imports immutable regardless of `let mut` in source — modules define shared vocabulary, not shared mutable state |
| 27 | Transitive imports | Disabled — nested `use` in modules doesn't propagate, preventing diamond dependency cycles |
| 30b | HashMap storage | `HashMap<String, HashMap<String, Value>>` — parallel to arrays' `HashMap<String, Vec<f32>>` |
| 30b | map_set syntax | Statement-level (like `push`) — mutation doesn't produce a value |
| 30b | map_keys return type | Sorted comma-separated string — arrays are `Vec<f32>` so can't hold string keys |
| 31 | Security model | Deno-style `--allow-read`/`--allow-write` CLI flags, deny-by-default, REPL defaults to allow |
| 31 | Security threading | Thread-local `Cell<bool>` flags — avoids threading 2 extra params through 82 `eval_scalar` call sites |
| 31 | File write syntax | `write_file`/`append_file` as AST statements (like `map_set`) — mutations, not value-producing |
| 31 | Array-returning fns | `eval_array_fn()` helper returning `Option<Vec<Value>>` — detected at `LetDecl` level |
| 31 | Heterogeneous arrays | `Vec<f32>` → `Vec<Value>` — arrays can hold strings, floats, or mixed types |
| 31 | Path utilities security | Pure string ops (`file_ext`, `file_name`, etc.) — no security check needed |
| 32 | str() integer formatting | `str(3.0)` produces `"3"` not `"3.0"` — clean formatting for whole numbers |
| 32 | int() truncation | Truncates toward zero (Rust `trunc()`) — `int(-3.7)` = `-3.0`, not `-4.0` |
| 32 | substr bounds | Char-based with clamping — no panics on out-of-range, returns shorter string |
| 32 | Builtin priority | Builtins override user-defined functions with same name — no shadowing allowed |
| 33 | Reduce dual-dispatch | `sum`/`min`/`max`/`count` check arrays first in Reduce handler, fall through to streams |
| 33 | sort_array naming | Named `sort_array` not `sort` to avoid collision with stdlib user-defined bubble_sort |
| 33 | find tolerance | Uses 1e-6 tolerance for float comparison, matching `==` operator behavior |
| 33 | Empty array errors | `first`/`last`/`min_val`/`max_val` error cleanly; `reverse`/`sort`/`unique` return empty |

---

## 17. Risk Registry

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| SPIR-V emission produces invalid bytecode | High | Medium | Phase 0 catches this immediately. Use spirv-val aggressively. |
| Vulkan compute not available on target hardware | High | Low | CPU scalar path for no-GPU targets from day one. Test on multiple GPU vendors. |
| Parallel reduction SPIR-V is wrong (shared memory bugs) | Medium | High | Most common GPU programming bug. Extensive testing with known values. |
| ~~ash crate Vulkan bindings have gaps~~ | ~~Medium~~ | ~~Low~~ | RESOLVED: ash eliminated Phase 49. Raw vk_sys.rs bindings — zero deps. |
| Performance is much worse than CUDA | Medium | Medium | Acceptable for v1 (correctness first). Optimization passes planned for later. |
| Float32 precision differences GPU vs CPU | Low | High | Expected. Use relative tolerance (1e-6). Document precision model. |
| ~~Cranelift CPU backend insufficient~~ | ~~Low~~ | ~~Low~~ | RESOLVED: Cranelift never needed. Interpreter handles all CPU work. |
| Module system too complex for LLMs | Medium | Medium | "Compiler infers everything" approach from Annex B minimizes author burden. |

---

## 18. Document Index

The following documents define the OctoFlow project. Claude Code should read all of them before starting implementation.

| Document | File | Content | Read Before |
|----------|------|---------|------------|
| **Roadmap** | `docs/roadmap.md` | This document. Implementation sequence, validation gates, risk registry | Always |
| **Annex C** | `docs/annex-c-compiler-internals.md` | SPIR-V architecture, Vulkan compute, Cranelift CPU backend, 10 compiler milestones | Phase 0+ |
| **Blueprint** | `docs/blueprint.md` | Master architecture, vision, three-layer overview, competitive landscape | Always |
| **Annex B** | `docs/annex-b-programming-model.md` | 23 language concepts, records/enums, let/var, stage/fn, pipes/match/if/for | Phase 2+ |
| **Annex A** | `docs/annex-a-language-spec.md` | Vanilla ops, modules, ecosystem, bridges, failsafe, LLM flywheel | Phase 4+ |
| **GPU Use Cases** | `docs/gpu-use-cases.md` | 150+ GPU use cases across 19 domains (market context) | Reference |
| **Coding Bible** | `docs/octoflow-coding-bible.md` | Style guide, patterns, anti-patterns, LLM generation rules | Before writing code |
| **Annex Q** | `docs/annex-q-octoengine.md` | OctoEngine gaming platform concept, bridge features, implementation roadmap | Gaming preparation |
| **Annex R** | `docs/annex-r-octodb.md` | OctoDB GPU-native append-only database, append-only model, version resolution, .octo format as storage layer | Reference |
| **Annex S** | `docs/annex-s-octomark.md` | OctoMark GPU-rendered Markdown, OctoOffice unified app, media recipe language, disruption strategy | Reference |
| **Annex T** | `docs/annex-t-disruption-vectors.md` | 25 disruption vectors (10 platform + 15 daily-use tools), FFmpeg playbook, wave-based build order, stack composition map | Strategy |

---

*This roadmap is a living document. Timelines and decisions will be adjusted based on validation gate outcomes.*
