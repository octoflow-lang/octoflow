# OctoFlow — Annex C: Layer 1 Compiler Internals

**Parent Document:** OctoFlow Blueprint & Architecture  
**Status:** Draft  
**Version:** 0.1  
**Date:** February 15, 2026  

---

## Table of Contents

1. Architecture Philosophy
2. Dependency Strategy
3. Compilation Pipeline Overview
4. Milestone 1: Parser
5. Milestone 2: Dataflow Graph Construction
6. Milestone 3: Static Analyzer
7. Milestone 4: Cost Model
8. Milestone 5: Graph Partitioner
9. Milestone 6: SPIR-V Code Generator (GPU)
10. Milestone 7: CPU Code Generator
11. Milestone 8: Transfer & Synchronization Generator
12. Milestone 9: Runtime
13. Milestone 10: End-to-End Integration
14. Testing Strategy
15. Future Optimization Passes
16. Open Questions

---

## 1. Architecture Philosophy

### 1.1 Core Principle: Minimal External Dependencies

The OctoFlow compiler is designed to own as much of its stack as possible. External dependencies are accepted only when:

- The dependency is a **stable specification** (not a moving-target library)
- The dependency is **driver-provided** (maintained by GPU vendors as part of their driver stack)
- Building the equivalent from scratch would take years with no meaningful advantage

Everything else is custom. This means more upfront work but dramatically less maintenance burden and zero risk of upstream breaking changes killing the project.

### 1.2 Why Not MLIR/LLVM

MLIR and LLVM are extraordinary infrastructure, but they come with costs:

| Concern | Impact |
|---------|--------|
| LLVM releases every ~6 months with API breaks | Constant maintenance to keep compiler building |
| MLIR C++ API is unstable and sparsely documented | Hard to build on, hard to debug |
| LLVM is ~30M lines of code | Massive build dependency, slow compile times |
| MLIR expertise is rare | Hard to find contributors or get help |
| Tight coupling to LLVM version | Can't update one without the other |

For a project that needs to be maintainable by a small team (or AI-assisted development), LLVM/MLIR is too heavy and too volatile.

### 1.3 The SPIR-V Alternative

SPIR-V (Standard Portable Intermediate Representation) is an open-standard binary IL maintained by Khronos Group. It is the universal GPU intermediate language:

- **Consumed by every major GPU vendor**: NVIDIA, AMD, Intel, ARM, Qualcomm
- **Specification-stable**: Backwards compatible across versions (current: v1.6, January 2025)
- **No SDK dependency**: SPIR-V is a binary format with a public spec — you emit bytes, not call a library
- **Driver-optimized**: GPU drivers perform their own optimization on SPIR-V input
- **Future-proof**: Microsoft adopting SPIR-V for Direct3D Shader Model 7 (announced September 2024)

OctoFlow compiles GPU stages to SPIR-V bytecode and submits them via the Vulkan Compute API. The GPU driver handles final compilation to hardware-native code.

### 1.4 Dependency Map

```
┌─────────────────────────────────────────────────┐
│  FULLY CUSTOM (no external dependency)           │
│                                                   │
│  • Parser                                         │
│  • Dataflow graph representation                  │
│  • Static analyzer                                │
│  • Cost model                                     │
│  • Graph partitioner                              │
│  • Optimization passes                            │
│  • Pre-flight / failsafe system                   │
├─────────────────────────────────────────────────┤
│  STABLE SPECIFICATION DEPENDENCIES               │
│                                                   │
│  • SPIR-V spec (GPU code emission)               │
│    - Public binary format, backwards compatible   │
│    - We emit bytes directly, no library needed    │
│    - spirv-val for optional validation            │
│                                                   │
│  • Vulkan Compute API (GPU execution)             │
│    - Driver-provided, backwards compatible        │
│    - Thin runtime interface for dispatch           │
│    - Available on all major OS + GPU combos       │
├─────────────────────────────────────────────────┤
│  LIGHTWEIGHT LIBRARY DEPENDENCIES                │
│                                                   │
│  • Cranelift (CPU code generation)                │
│    - Rust-native, much simpler than LLVM          │
│    - Used by Wasmtime (production-grade)          │
│    - ~200K lines vs LLVM's ~30M lines             │
│    - Stable API, infrequent breaking changes      │
│    OR                                             │
│  • QBE (alternative CPU backend)                  │
│    - ~12K lines of C, trivially stable            │
│    - Less optimized but nearly zero maintenance   │
│                                                   │
│  • Vulkan loader (libvulkan)                      │
│    - System library, ships with GPU drivers       │
│    - Stable C API                                 │
└─────────────────────────────────────────────────┘
```

**What we do NOT depend on:**
- ❌ MLIR (no churn from LLVM release cycle)
- ❌ LLVM (no 30M line dependency)
- ❌ CUDA Toolkit (no NVIDIA vendor lock-in)
- ❌ nvcc / PTX (no NVIDIA-specific IR)
- ❌ ROCm / HIP (no AMD-specific toolchain)
- ❌ Any vendor-specific compiler

---

## 2. Compilation Pipeline Overview

```
OctoFlow Source (.flow file)
    ↓
┌──────────────────┐
│  MILESTONE 1     │
│  Parser          │  Source text → Abstract Syntax Tree (AST)
└──────┬───────────┘
       ↓
┌──────────────────┐
│  MILESTONE 2     │
│  Graph Builder   │  AST → Dataflow Graph (nodes + edges + types)
└──────┬───────────┘
       ↓
┌──────────────────┐
│  MILESTONE 3     │
│  Static Analyzer │  Annotate nodes: purity, dimensions, intensity,
│                  │  dependency class, parallel profile
└──────┬───────────┘
       ↓
┌──────────────────┐
│  MILESTONE 4     │
│  Cost Model      │  Per-node: estimated GPU time, CPU time,
│                  │  transfer overhead → classification
└──────┬───────────┘
       ↓
┌──────────────────┐
│  MILESTONE 5     │
│  Graph           │  Partition graph into GPU/CPU subgraphs,
│  Partitioner     │  insert transfer nodes, fuse adjacent stages
└──────┬───────────┘
       ↓
  ┌────┴────┐
  ↓         ↓
┌──────┐ ┌──────┐
│  M6  │ │  M7  │
│ SPIRV│ │ CPU  │  GPU subgraphs → SPIR-V bytecode
│ Gen  │ │ Gen  │  CPU subgraphs → native machine code
└──┬───┘ └──┬───┘
   ↓         ↓
┌──────────────────┐
│  MILESTONE 8     │
│  Transfer Gen    │  Generate async CPU↔GPU data movement code
└──────┬───────────┘
       ↓
┌──────────────────┐
│  MILESTONE 9     │
│  Runtime         │  Scheduler, memory manager, Vulkan interface,
│                  │  hardware discovery, OOM prevention
└──────┬───────────┘
       ↓
┌──────────────────┐
│  MILESTONE 10    │
│  Integration     │  End-to-end: source → compiled → runs → output
└──────────────────┘
```

---

## 3. Implementation Language

**Rust** is the implementation language for the entire compiler and runtime.

Rationale:
- Memory safety without garbage collection (critical for a system that manages GPU memory)
- Cranelift is Rust-native (zero FFI overhead for CPU code gen)
- Vulkan bindings are mature in Rust (ash crate — thin, unsafe, stable)
- Strong type system catches compiler bugs at compile time
- Single binary output — no runtime dependency on Python/Node/JVM
- AI coding assistants (Claude Code) are strong at Rust

---

## 4. Milestone 1: Parser

### Goal
Convert OctoFlow source text into an Abstract Syntax Tree (AST).

### Input
```flow
stream prices = tap("market_feed")
stream ema = prices |> temporal decay(0.94)
stream signals = ema |> crossover(fast=12, slow=26)
emit(signals, "output")
```

### Output
AST data structure representing:
- Stream declarations with names and types
- Tap sources
- Pipe chains (stage connections)
- Stage invocations with arguments
- Temporal markers
- Emit sinks

### Implementation

Use a **PEG parser** (Parsing Expression Grammar) via the `pest` Rust crate, or hand-write a recursive descent parser for maximum control and zero dependency.

**Recommended: hand-written recursive descent parser.**

Rationale: The OctoFlow grammar is small (streams, pipes, stages, taps, annotations). A hand-written parser is ~500-1000 lines of Rust, has zero dependencies, produces the exact AST structure we need, and is trivially debuggable.

### Grammar Sketch

```
program      = statement*
statement    = stream_decl | emit_stmt | stage_def | module_decl | import_stmt
stream_decl  = "stream" IDENT (":" type)? "=" expr
emit_stmt    = "emit" "(" expr "," STRING ")"
expr         = pipe_expr
pipe_expr    = primary ("|>" stage_call)*
stage_call   = ("temporal")? IDENT "(" args? ")"
primary      = IDENT | tap_expr | literal
tap_expr     = "tap" "(" STRING ")"
stage_def    = annotation* "stage" IDENT "(" params ")" "->" type ":" block
type         = primitive_type | array_type | stream_type | record_type
annotation   = "@" IDENT ("(" args ")")?
```

### Test Criteria
- [ ] Parse all 3 blueprint examples (video, statistics, financial) into valid ASTs
- [ ] Reject syntactically invalid input with clear error messages and line numbers
- [ ] Round-trip test: parse → pretty-print → parse → verify identical AST
- [ ] Parse module definitions in both .flow and markdown format

### Checkpoint
Hand the parser to Claude Code with the grammar sketch and test criteria. Verify output by having it parse the blueprint examples and print the AST structure. Human review: does the AST look right for each example?

---

## 5. Milestone 2: Dataflow Graph Construction

### Goal
Convert the AST into a Dataflow Graph — the core data structure the rest of the compiler operates on.

### Data Structure

```rust
struct DataflowGraph {
    nodes: Vec<Node>,        // stages (computation)
    edges: Vec<Edge>,        // pipes (data flow)
    taps: Vec<Tap>,          // I/O boundaries
    emits: Vec<Emit>,        // output boundaries
}

struct Node {
    id: NodeId,
    kind: NodeKind,          // Stage, Accumulator, Branch, Merge
    operation: String,       // "decay", "crossover", "resize", etc.
    args: Vec<Argument>,     // stage arguments
    input_types: Vec<Type>,  // input edge types
    output_type: Type,       // output edge type
    temporal: bool,          // has temporal dependency?
    annotations: Vec<Annotation>,
    // Filled by static analyzer (Milestone 3):
    parallel_profile: Option<ParallelProfile>,
}

struct Edge {
    from: NodeId,
    to: NodeId,
    data_type: Type,
}

struct ParallelProfile {
    purity: Purity,                    // Pure | Impure
    parallelizable_dims: Vec<usize>,   // which dimensions can parallelize
    sequential_dims: Vec<usize>,       // which dimensions must be sequential
    arithmetic_intensity: Intensity,    // Low | Medium | High
    memory_access: AccessPattern,       // Coalesced | Strided | Scattered
    dependency_class: DepClass,         // Independent | Temporal | Reduction | Sequential
}
```

### Transformations
- Resolve pipe chains into graph edges
- Resolve operation names to vanilla/module definitions
- Validate type compatibility across all edges
- Detect and flag cycles (only valid if marked temporal)

### Test Criteria
- [ ] Video example produces a linear chain: tap → decode → resize → color_grade → composite → encode → emit
- [ ] Financial example produces: tap → decay (temporal) → crossover → position_size → emit
- [ ] Statistics example produces a DAG with branching (group_by fans out to multiple aggs)
- [ ] Type mismatch between connected stages produces a clear compile error
- [ ] Circular dependency without temporal marker produces an error

### Checkpoint
Verify graph structure by having the compiler dump the graph in DOT format (for Graphviz visualization). Human review: does the graph match the expected pipeline shape?

---

## 6. Milestone 3: Static Analyzer

### Goal
Annotate every node in the dataflow graph with its parallel profile — the information the cost model and partitioner need to make GPU/CPU decisions.

### Analysis Passes

**Pass 1: Purity Analysis**
- Vanilla operations have pre-declared purity (all vanilla is pure)
- Module operations carry purity in their manifest
- Custom stages: analyze function body for side effects (external calls, I/O, mutable globals)
- Result: each node tagged `Pure` or `Impure`

**Pass 2: Dimensional Analysis**
- From type signatures, determine which dimensions each stage operates on
- `scale(x: float) -> float` → operates per-element → all dimensions parallelizable
- `row_mean(row: float[N]) -> float` → operates per-row → parallelize across rows
- `global_sum(data: float[M,N]) -> float` → full reduction → limited parallelism
- Result: each node tagged with `parallelizable_dims` and `sequential_dims`

**Pass 3: Arithmetic Intensity Estimation**
- Count compute operations vs memory operations in the stage body
- For vanilla ops: pre-declared (e.g., `multiply` = 1 op per element = low intensity; `std` = ~10 ops per element = medium; `matmul` = O(N) ops per element = high)
- For composition modules: aggregate from constituent operations
- Result: each node tagged `Low | Medium | High` intensity

**Pass 4: Dependency Classification**
- Independent: no dependency between data elements (embarrassingly parallel)
- Temporal: iteration N depends on N-1 (marked with `temporal` keyword)
- Reduction: many-to-one or many-to-few (e.g., `sum`, `group_by`)
- Sequential: strict ordering required (e.g., impure stages with side effects)
- Result: each node tagged with dependency class

**Pass 5: Memory Estimation**
- For each node, estimate GPU memory required as a function of input size
- Vanilla ops declare their memory formulas
- For composition modules: aggregate from constituent stages + intermediate buffers
- Result: each node tagged with `memory_formula(input_size) -> bytes`

### Hard Classification (from static analysis alone)

After all passes, some nodes can be classified immediately:

| Profile | Classification | Reason |
|---------|---------------|--------|
| Pure + Independent + High intensity | **Hard-GPU** | Perfectly parallel, compute-heavy |
| Pure + Independent + Medium intensity + large data | **Hard-GPU** | Parallel, enough work to justify GPU |
| Impure (side effects, I/O) | **Hard-CPU** | Cannot parallelize, needs OS access |
| String operations | **Hard-CPU** | GPU hostile |
| Small fixed-size computation | **Hard-CPU** | Transfer overhead > compute savings |
| Everything else | **Conditional** | Depends on runtime data size |

### Test Criteria
- [ ] Video example: decode → CPU, resize → GPU, color_grade → GPU, composite → GPU, encode → CPU
- [ ] Financial example: decay → conditional (temporal + parallel across instruments), crossover → GPU, position_size → GPU
- [ ] Statistics example: group_by → CPU, agg(mean,std,skew,kurtosis) → GPU, filter → CPU
- [ ] Pure stages are correctly identified as pure
- [ ] Temporal stages are correctly identified
- [ ] Memory estimates are within 2x of actual (for known vanilla operations)

### Checkpoint
Print the annotated graph showing each node's parallel profile and classification. Human review: do the classifications match intuition?

---

## 7. Milestone 4: Cost Model

### Goal
For nodes classified as "conditional," generate a runtime decision function that determines GPU vs CPU based on actual data size.

### Hardware Parameters (auto-detected or configured)

```rust
struct HardwareProfile {
    // GPU
    gpu_available: bool,
    gpu_name: String,
    gpu_memory_bytes: u64,
    gpu_compute_tflops_f32: f64,    // theoretical peak TFLOPS
    gpu_memory_bandwidth_gbps: f64,  // HBM/GDDR bandwidth
    gpu_sm_count: u32,
    gpu_max_threads_per_sm: u32,
    
    // CPU
    cpu_cores: u32,
    cpu_freq_ghz: f64,
    cpu_memory_bandwidth_gbps: f64,
    
    // Interconnect
    pcie_bandwidth_gbps: f64,        // CPU↔GPU transfer bandwidth
    pcie_latency_us: f64,            // per-transfer fixed latency
}
```

### Cost Estimation Formulas

**For memory-bound operations** (element-wise, simple transforms):
```
gpu_time_ms = (data_bytes / gpu_memory_bandwidth_bps) × 1000
cpu_time_ms = (data_bytes / cpu_memory_bandwidth_bps) × 1000
transfer_ms = (data_bytes / pcie_bandwidth_bps) × 1000 + pcie_latency_ms
```

**For compute-bound operations** (matmul, complex math):
```
gpu_time_ms = (total_flops / gpu_compute_flops) × 1000
cpu_time_ms = (total_flops / cpu_compute_flops) × 1000
transfer_ms = ((input_bytes + output_bytes) / pcie_bandwidth_bps) × 1000 + pcie_latency_ms
```

**Decision:**
```
if gpu_time_ms + transfer_ms < cpu_time_ms:
    assign GPU
else:
    assign CPU
```

### Transfer Context Awareness

The cost model is **context-aware** — it knows whether neighboring stages are GPU or CPU:

- If the previous stage is already GPU → transfer cost for this stage's input is 0 (data already on GPU)
- If the next stage is already GPU → transfer cost for this stage's output is 0
- Only stages at GPU↔CPU boundaries incur transfer cost

This means the cost model and graph partitioner iterate together (Milestone 5 feeds back into Milestone 4).

### Generated Decision Functions

For each conditional node, the compiler generates a lightweight Rust function:

```rust
fn should_use_gpu_node_7(data_size: usize, hw: &HardwareProfile, context: &NeighborContext) -> bool {
    let data_bytes = data_size * 4; // float32
    let transfer_needed = context.prev_is_cpu || context.next_is_cpu;
    let transfer_cost = if transfer_needed { 
        data_bytes as f64 / hw.pcie_bandwidth_bps + hw.pcie_latency_us 
    } else { 0.0 };
    let gpu_time = data_bytes as f64 / hw.gpu_memory_bandwidth_bps;
    let cpu_time = data_bytes as f64 / hw.cpu_memory_bandwidth_bps;
    
    gpu_time + transfer_cost < cpu_time
}
```

### Hardware Detection

At first run (or when `flow detect-hardware` is invoked):

```rust
// Vulkan device enumeration
vkEnumeratePhysicalDevices()  → list GPUs
vkGetPhysicalDeviceProperties() → name, limits, memory
vkGetPhysicalDeviceMemoryProperties() → VRAM size

// CPU detection
sysconf / /proc/cpuinfo → cores, frequency

// Benchmark micro-kernels (one-time, ~2 seconds)
// Measure actual bandwidth, not theoretical peak
gpu_bandwidth_test()  → actual GPU memory bandwidth
cpu_bandwidth_test()  → actual CPU memory bandwidth  
pcie_bandwidth_test() → actual transfer bandwidth
```

Results cached in `~/.flowgpu/hardware_profile.json`.

### Test Criteria
- [ ] For 100-element float32 array: cost model says CPU for all operations
- [ ] For 1M-element float32 array: cost model says GPU for element-wise operations
- [ ] For matmul of 1000×1000: cost model says GPU
- [ ] For matmul of 10×10: cost model says CPU
- [ ] Context-aware: if both neighbors are GPU, a borderline operation tips to GPU
- [ ] Hardware detection produces sane values on test machine

### Checkpoint
Print cost model decisions for each blueprint example at various data sizes (100, 10K, 1M). Human review: do the crossover points make sense?

---

## 8. Milestone 5: Graph Partitioner

### Goal
Partition the annotated dataflow graph into GPU subgraphs and CPU subgraphs with minimal data transfer.

### Algorithm

**Step 1: Initial Assignment**
Assign each node to its classified device (hard-GPU, hard-CPU, or conditional evaluated by cost model).

**Step 2: Identify Boundaries**
Find all edges where source and destination are on different devices. These are transfer points.

**Step 3: GPU Promotion Analysis**
For each CPU node sandwiched between GPU nodes:
```
promotion_cost = cpu_execution_time_on_gpu  (slower, but no transfer)
no_promotion_cost = cpu_execution_time_on_cpu + 2 × transfer_time

if promotion_cost < no_promotion_cost:
    promote node to GPU
```

**Step 4: Subgraph Fusion**
Merge adjacent same-device nodes into contiguous subgraphs:
- Adjacent GPU nodes → single GPU dispatch
- Adjacent CPU nodes → single CPU function
- This reduces kernel launch overhead and eliminates unnecessary synchronization

**Step 5: Transfer Node Insertion**
At each remaining GPU↔CPU boundary, insert an explicit transfer node:
- GPU→CPU: `vkMapMemory` / buffer copy
- CPU→GPU: `vkFlushMappedMemoryRanges` / buffer copy
- Transfers are async where possible (overlap with computation)

### Output

```rust
struct PartitionedGraph {
    gpu_subgraphs: Vec<GpuSubgraph>,  // groups of fused GPU nodes
    cpu_subgraphs: Vec<CpuSubgraph>,  // groups of fused CPU nodes
    transfers: Vec<Transfer>,          // CPU↔GPU data movements
    execution_order: Vec<ExecutionStep>, // topological order respecting dependencies
}

enum ExecutionStep {
    GpuDispatch(GpuSubgraphId),
    CpuExecute(CpuSubgraphId),
    Transfer(TransferId),
    Barrier,  // synchronization point
}
```

### Test Criteria
- [ ] Video example: 1 CPU subgraph (decode), 1 GPU subgraph (resize+color_grade+composite fused), 1 CPU subgraph (encode), 2 transfers
- [ ] Three adjacent GPU stages fuse into one GPU subgraph
- [ ] GPU promotion: CPU node between two GPU nodes is promoted when profitable
- [ ] Execution order respects all data dependencies
- [ ] No transfer inserted between same-device nodes

### Checkpoint
Visualize the partitioned graph with GPU nodes colored green, CPU nodes colored blue, transfers colored red. Human review: does the partitioning make sense?

---

## 9. Milestone 6: SPIR-V Code Generator (GPU)

### Goal
Compile GPU subgraphs into valid SPIR-V compute shader bytecode.

### Why SPIR-V

SPIR-V is a binary format with a public specification. We emit bytes directly — no library required (though `spirv-val` can optionally validate output).

The format is structured as:
```
Magic number (0x07230203)
Version
Generator ID
Bound (highest ID + 1)
Schema (0)
Instructions...
```

Each instruction is: `word_count | opcode | operands...`

### SPIR-V Emitter

Build a custom SPIR-V emitter in Rust:

```rust
struct SpirVModule {
    instructions: Vec<Instruction>,
    next_id: u32,
}

impl SpirVModule {
    fn emit_capability(&mut self, cap: Capability);
    fn emit_memory_model(&mut self);
    fn emit_entry_point(&mut self, name: &str, id: u32);
    fn emit_type_int(&mut self, width: u32, signed: bool) -> u32;
    fn emit_type_float(&mut self, width: u32) -> u32;
    fn emit_type_vector(&mut self, component: u32, count: u32) -> u32;
    fn emit_type_array(&mut self, element: u32, length: u32) -> u32;
    fn emit_type_pointer(&mut self, storage_class: StorageClass, pointee: u32) -> u32;
    fn emit_variable(&mut self, type_id: u32, storage: StorageClass) -> u32;
    fn emit_load(&mut self, type_id: u32, pointer: u32) -> u32;
    fn emit_store(&mut self, pointer: u32, value: u32);
    fn emit_fadd(&mut self, type_id: u32, a: u32, b: u32) -> u32;
    fn emit_fmul(&mut self, type_id: u32, a: u32, b: u32) -> u32;
    // ... all needed SPIR-V ops
    
    fn to_bytes(&self) -> Vec<u8>;  // serialize to binary
}
```

### Compilation Patterns

**Pattern: Parallel Map (element-wise operation)**

The most common pattern. Each GPU thread processes one element.

```
OctoFlow: stream result = data |> scale(2.0)

SPIR-V compute shader:
  layout(local_size_x = 256) in;
  
  buffer InputBuffer  { float data[]; };
  buffer OutputBuffer { float result[]; };
  
  void main() {
      uint idx = gl_GlobalInvocationID.x;
      if (idx < data.length) {
          result[idx] = data[idx] * 2.0;
      }
  }
```

In SPIR-V binary:
- Declare compute capability
- Declare buffer storage class variables for input/output
- Get global invocation ID (built-in)
- Bounds check
- Load, compute, store
- Set workgroup size via execution mode

**Pattern: Parallel Reduction (sum, mean, etc.)**

Two-phase: parallel reduction within workgroups, then reduction across workgroups.

```
Phase 1: Each workgroup reduces its chunk using shared memory
Phase 2: A second dispatch reduces workgroup results

SPIR-V uses:
  - Workgroup shared memory (OpVariable with Workgroup storage class)
  - Barrier (OpControlBarrier)
  - Tree reduction within shared memory
```

**Pattern: Temporal Pipeline (EMA, cumulative ops)**

Sequential across time, parallel across data dimensions.

```
OctoFlow: stream ema = prices |> temporal decay(0.94)

SPIR-V: Each thread handles one instrument across all time steps
  thread_idx = gl_GlobalInvocationID.x;  // instrument index
  acc = 0.0;
  for t in 0..time_steps:
      acc = 0.94 * prices[t][thread_idx] + (1 - 0.94) * acc
      output[t][thread_idx] = acc
```

Parallel across instruments (thousands of threads), sequential across time within each thread.

**Pattern: Group-By Aggregation**

Pre-sorted or hash-based grouping on CPU → parallel aggregation per group on GPU.

The group_by itself runs on CPU (hash table construction). The per-group aggregation (mean, std, etc.) dispatches to GPU with one workgroup per group.

### Vulkan Compute Dispatch

The compiled SPIR-V is loaded and dispatched via Vulkan:

```rust
// Pseudocode for GPU dispatch
fn dispatch_gpu_subgraph(spirv_bytes: &[u8], input: &GpuBuffer, output: &GpuBuffer, data_size: usize) {
    let shader_module = device.create_shader_module(spirv_bytes);
    let pipeline = device.create_compute_pipeline(shader_module);
    let descriptor_set = bind_buffers(input, output);
    
    let workgroup_size = 256;
    let num_workgroups = (data_size + workgroup_size - 1) / workgroup_size;
    
    command_buffer.bind_pipeline(pipeline);
    command_buffer.bind_descriptor_set(descriptor_set);
    command_buffer.dispatch(num_workgroups, 1, 1);
    
    queue.submit(command_buffer);
}
```

### SPIR-V Validation

Optionally validate emitted SPIR-V using `spirv-val` (from SPIRV-Tools). This is a development/debug tool, not a runtime dependency.

```bash
spirv-val output.spv  # validate SPIR-V binary
spirv-dis output.spv  # disassemble for human inspection
```

### Test Criteria
- [ ] Emit valid SPIR-V for element-wise multiply (simplest case)
- [ ] `spirv-val` passes on all generated SPIR-V
- [ ] Element-wise shader produces correct results via Vulkan dispatch
- [ ] Parallel reduction (sum) produces correct results
- [ ] Temporal pipeline (EMA) produces correct results
- [ ] Performance within 5x of hand-written Vulkan compute shader (initial target — optimization comes later)

### Checkpoint
This is the critical milestone. Have Claude Code generate SPIR-V for the simplest case (element-wise multiply), validate with `spirv-val`, dispatch via Vulkan, verify results. Then progressively add patterns (reduction, temporal). Human review at each pattern: does the output match expected?

---

## 10. Milestone 7: CPU Code Generator

### Goal
Compile CPU subgraphs into native machine code.

### Backend: Cranelift

Cranelift is a code generator library written in Rust, used by Wasmtime (WebAssembly runtime). It is:

- Fast compilation (designed for JIT, much faster than LLVM)
- Reasonable optimization (not LLVM-level, but good enough)
- Stable Rust API
- ~200K lines of code (vs LLVM's ~30M)
- Actively maintained by the Bytecode Alliance

For CPU subgraphs, we:
1. Lower the dataflow graph nodes into Cranelift IR
2. Cranelift compiles to native x86-64 / ARM64
3. Output is a callable function pointer

### Alternative: Interpreted Fallback

For initial development, CPU subgraphs can be interpreted rather than compiled. This is slower but much simpler to implement:

```rust
fn interpret_cpu_subgraph(graph: &CpuSubgraph, input: &[f32]) -> Vec<f32> {
    let mut data = input.to_vec();
    for node in &graph.nodes {
        data = execute_vanilla_op(&node.operation, &data, &node.args);
    }
    data
}
```

**Recommended approach**: Start with interpreter (Milestone 7a), add Cranelift later (Milestone 7b). This unblocks end-to-end testing while deferring the harder code gen work.

### Test Criteria
- [ ] CPU interpreter correctly executes all vanilla operations
- [ ] String operations work correctly (CPU-only)
- [ ] File I/O taps work correctly
- [ ] Results match vanilla reference implementation

### Checkpoint
Run all blueprint examples with GPU stages on GPU and CPU stages interpreted. Verify end-to-end correctness.

---

## 11. Milestone 8: Transfer & Synchronization Generator

### Goal
Generate efficient async data transfer operations at GPU↔CPU boundaries.

### Transfer Implementation

Using Vulkan:

```rust
// CPU → GPU transfer
fn transfer_to_gpu(cpu_data: &[u8], gpu_buffer: &VkBuffer) {
    let staging = create_staging_buffer(cpu_data.len());  // host-visible
    copy_to_staging(staging, cpu_data);
    
    // Async copy from staging to device-local buffer
    cmd.copy_buffer(staging, gpu_buffer);
    cmd.pipeline_barrier(/* ensure transfer completes before compute */);
}

// GPU → CPU transfer
fn transfer_from_gpu(gpu_buffer: &VkBuffer, cpu_data: &mut [u8]) {
    let staging = create_staging_buffer(cpu_data.len());
    
    cmd.pipeline_barrier(/* ensure compute completes before transfer */);
    cmd.copy_buffer(gpu_buffer, staging);
    
    // After submission and fence wait:
    copy_from_staging(staging, cpu_data);
}
```

### Double Buffering

For streaming pipelines (continuous taps), use double buffering:
- While GPU processes buffer A, CPU fills buffer B
- Swap on each frame/tick
- Eliminates stall time waiting for transfers

### Synchronization

Vulkan provides fences (CPU↔GPU sync) and semaphores (GPU↔GPU sync):
- Fence after GPU dispatch: CPU waits for GPU completion before reading results
- Semaphore between dispatches: second GPU dispatch waits for first to finish
- Pipeline barriers within command buffers: ensure memory coherence

### Test Criteria
- [ ] CPU→GPU transfer of 1M floats completes correctly
- [ ] GPU→CPU transfer of 1M floats completes correctly
- [ ] Double buffering reduces stall time measurably
- [ ] Synchronization prevents data races (no corrupted results)

---

## 12. Milestone 9: Runtime

### Goal
A lightweight runtime that orchestrates execution of the partitioned graph.

### Components

```rust
struct FlowGpuRuntime {
    hardware: HardwareProfile,
    vulkan: VulkanContext,       // device, queues, command pool
    memory_pool: GpuMemoryPool, // pre-allocated, reusable GPU buffers
    scheduler: Scheduler,        // executes the partitioned graph
    monitor: MemoryMonitor,      // OOM prevention (never disabled)
}
```

### Scheduler

The scheduler walks the execution order from the partitioned graph:

```rust
fn execute(runtime: &mut Runtime, graph: &PartitionedGraph, inputs: HashMap<String, Data>) {
    // Bind tap inputs
    for tap in &graph.taps {
        runtime.bind_input(tap, &inputs[&tap.name]);
    }
    
    // Execute in topological order
    for step in &graph.execution_order {
        match step {
            ExecutionStep::GpuDispatch(id) => {
                let subgraph = &graph.gpu_subgraphs[*id];
                runtime.check_gpu_memory(subgraph.estimated_memory)?; // OOM prevention
                runtime.dispatch_gpu(subgraph);
            }
            ExecutionStep::CpuExecute(id) => {
                let subgraph = &graph.cpu_subgraphs[*id];
                runtime.execute_cpu(subgraph);
            }
            ExecutionStep::Transfer(id) => {
                let transfer = &graph.transfers[*id];
                runtime.execute_transfer(transfer);
            }
            ExecutionStep::Barrier => {
                runtime.gpu_fence_wait();
            }
        }
    }
    
    // Read emit outputs
    for emit in &graph.emits {
        runtime.read_output(emit);
    }
}
```

### GPU Memory Pool

Pre-allocate a pool of GPU buffers to avoid per-operation allocation overhead:

```rust
struct GpuMemoryPool {
    free_buffers: BTreeMap<usize, Vec<VkBuffer>>,  // size → available buffers
    allocated: usize,
    limit: usize,  // never exceed (OOM prevention)
}

impl GpuMemoryPool {
    fn allocate(&mut self, size: usize) -> Result<VkBuffer, OomError> {
        if self.allocated + size > self.limit {
            return Err(OomError::GpuMemoryExhausted {
                requested: size,
                available: self.limit - self.allocated,
            });
        }
        // Try to reuse an existing buffer of sufficient size
        // Otherwise allocate new
    }
    
    fn release(&mut self, buffer: VkBuffer, size: usize) {
        self.free_buffers.entry(size).or_default().push(buffer);
        self.allocated -= size;
    }
}
```

### Memory Monitor (Never Disabled)

```rust
fn check_gpu_memory(&self, required_bytes: usize) -> Result<(), FlowGpuError> {
    let available = self.memory_pool.limit - self.memory_pool.allocated;
    if required_bytes > available {
        // Option 1: Try to free unused buffers
        self.memory_pool.gc();
        
        // Option 2: If still insufficient, fall back to CPU
        if required_bytes > self.memory_pool.available_after_gc() {
            return Err(FlowGpuError::GpuMemoryInsufficient {
                required: required_bytes,
                available: available,
                suggestion: "Pipeline will execute this stage on CPU as fallback",
            });
        }
    }
    Ok(())
}
```

The runtime NEVER crashes with OOM. It either:
1. Falls back to CPU for the offending stage
2. Chunks the data into GPU-sized pieces
3. Reports a clear error with actionable suggestions

### Test Criteria
- [ ] Runtime discovers hardware correctly
- [ ] Memory pool allocates and reuses buffers
- [ ] OOM prevention triggers before actual OOM
- [ ] Full pipeline executes via scheduler in correct order
- [ ] Concurrent GPU and CPU stages execute in parallel where independent

---

## 13. Milestone 10: End-to-End Integration

### Goal
All milestones integrated into a single `flowgpu` binary that takes a .flow file and runs it.

### CLI

```bash
flowgpu run program.flow                    # compile and execute
flowgpu compile program.flow -o program.fgb # compile to binary
flowgpu run program.fgb                     # execute compiled binary
flowgpu check program.flow                  # pre-flight only (no execution)
flowgpu detect-hardware                     # detect and cache hardware profile
flowgpu graph program.flow                  # dump dataflow graph (DOT format)
flowgpu partition program.flow              # show GPU/CPU partitioning
```

### Test Criteria — The Three Blueprint Examples

**Test 1: Financial Signal Generation**
```flow
stream prices = tap("market_feed")
stream ema = prices |> temporal decay(0.94)
stream signals = ema |> crossover(fast=12, slow=26)
stream sized = signals |> position_size(risk=0.02)
emit(sized, "execution_engine")
```
- Input: 10,000 instruments × 1,000 time steps
- Expected: decay parallelizes across instruments, sequential across time
- Verify: output matches vanilla reference computation

**Test 2: Video Processing**
```flow
stream frames = tap("camera_feed")
stream resized = frames |> resize(1080, 1920)
stream filtered = resized |> color_grade(lut)
stream overlay = filtered |> composite(ui_layer)
emit(overlay, "output_file")
```
- Input: 100 frames of 1080p
- Expected: resize, color_grade, composite all on GPU, fused
- Verify: output images match vanilla reference

**Test 3: Statistical Analysis**
```flow
stream trades = tap("trade_log")
stream cleaned = trades |> drop_null() |> cast_types()
stream grouped = cleaned |> group_by("symbol")
stream stats = grouped |> agg(mean, std, skew, kurtosis)
stream ranked = stats |> rank("sharpe_ratio")
emit(ranked, "report")
```
- Input: 1M trade records, 500 symbols
- Expected: agg on GPU, group_by on CPU, appropriate transfers
- Verify: statistics match vanilla reference within float32 tolerance

### Performance Targets (v0.1)

| Metric | Target | Notes |
|--------|--------|-------|
| Correctness | 100% match vs vanilla reference | Non-negotiable |
| GPU utilization | > 50% for GPU-classified stages | Room to improve |
| vs hand-written Vulkan | Within 5x | Initial target, not final |
| Compilation time | < 2 seconds for blueprint examples | Fast iteration |
| Runtime overhead | < 10ms scheduler + dispatch | Lightweight |

### Checkpoint
This is the final integration checkpoint. Run all three blueprint examples, verify correctness, measure performance, document results. Human review: does it work? Is it correct? Is performance in the right ballpark?

---

## 14. Testing Strategy

### Test Pyramid

```
                    ┌───────────┐
                    │ End-to-End │  3 blueprint examples
                    │  (M10)     │  + performance benchmarks
                   ┌┴───────────┴┐
                   │  Integration │  Cross-milestone tests
                   │  (M8-M9)    │  transfer + runtime
                  ┌┴─────────────┴┐
                  │  Component     │  Per-milestone unit tests
                  │  (M1-M7)      │  parser, analyzer, SPIR-V gen
                 ┌┴───────────────┴┐
                 │  Vanilla Reference│  Correctness oracle
                 │  (always)        │  every operation has test vectors
                 └─────────────────┘
```

### Vanilla Reference Tests

Every vanilla operation has a set of reference test vectors:

```rust
#[test]
fn test_vanilla_ema() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let decay = 0.5;
    let expected = vec![1.0, 1.5, 2.25, 3.125, 4.0625];
    
    // Test CPU path
    assert_eq!(vanilla_ema_cpu(&input, decay), expected);
    
    // Test GPU path (SPIR-V)
    assert_eq!(vanilla_ema_gpu(&input, decay), expected);
    
    // Test auto-dispatch (should choose based on cost model)
    assert_eq!(vanilla_ema_auto(&input, decay), expected);
}
```

GPU and CPU paths must produce identical results (within float32 tolerance).

---

## 15. Future Optimization Passes

The v0.1 compiler prioritizes correctness over performance. Future versions add optimization passes without changing the architecture:

| Pass | Description | Expected Speedup |
|------|-------------|-----------------|
| Stage fusion | Merge adjacent element-wise stages into single kernel | 2-3x (reduces kernel launch overhead) |
| Memory coalescing | Reorder data layout for coalesced GPU memory access | 2-5x for memory-bound operations |
| Workgroup size tuning | Auto-tune workgroup dimensions per kernel | 10-50% |
| Shared memory tiling | Use shared memory for data reuse within tiles | 2-10x for stencil/convolution patterns |
| Async overlap | Overlap computation and transfer | 20-50% for streaming pipelines |
| Profiler-guided | Use actual runtime data to refine cost model | Better GPU/CPU decisions |
| SPIR-V optimization | Run spirv-opt on generated SPIR-V | 10-30% from dead code elimination, constant folding |

Each pass is independent and can be added incrementally. None require architectural changes.

---

## 16. Open Questions

### 16.1 SPIR-V Generation

- **Compute shader limits**: SPIR-V compute shaders have workgroup size limits (typically 1024 threads). How do we handle data sizes that require many workgroups?
- **Shared memory**: When should the compiler use workgroup shared memory vs global memory? What heuristics guide this?
- **Subgroup operations**: Modern GPUs support subgroup (warp/wavefront) operations for fast intra-group communication. When should the compiler use these?

### 16.2 Vulkan Compute

- **Multiple queues**: Should the runtime use multiple Vulkan queues for concurrent dispatch?
- **Descriptor set management**: How to efficiently manage Vulkan descriptor sets for dynamic pipelines?
- **Push constants vs uniform buffers**: When to use each for small parameters?

### 16.3 CPU Backend

- **Cranelift vs interpreter tradeoff**: At what pipeline complexity does JIT compilation outperform interpretation?
- **Multi-threading**: Should CPU subgraphs use multi-threading for data-parallel operations?
- **SIMD**: Should the CPU backend use SIMD instructions for vectorizable operations?

### 16.4 Cross-Platform

- **macOS/Metal**: Apple doesn't support Vulkan natively. Options: MoltenVK (Vulkan→Metal translation) or a separate Metal SPIR-V path?
- **WebGPU**: Could OctoFlow target WebGPU for browser-based execution?
- **Mobile**: Vulkan Compute works on Android. What about iOS (Metal)?

### 16.5 Incremental Compilation

- **Caching**: Should compiled SPIR-V be cached? How to invalidate when source changes?
- **Hot reload**: Can pipeline stages be recompiled without restarting the runtime?

---

*This annex is a living document. Implementation will reveal additional design decisions at each milestone.*
