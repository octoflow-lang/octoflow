# The Loom Engine

> Weaving threads of parallel computation.

```
        🐙
      ╱╱╲╲╲╲
     ╱╱  ╲╲╲╲        OctoFlow's Loom Engine
    ││ ◉◉ ││││       ─────────────────────
    ╲╲    ╱╱╱╱       GPU compute runtime
     ╲╲  ╱╱╱╱        for the rest of us
      ╰╱╲╱╲╯
     ╱│││││╲
    ╱ │╱╲│╲ ╲         Eight arms.
   ╱  ╱  ╲  ╲        Thousands of threads.
  ╱  ╱ ◇◇ ╲  ╲       One loom.
```

---

## What is Loom?

Loom is OctoFlow's GPU compute engine. It weaves thousands of parallel threads
into coordinated computation — sieving primes, multiplying matrices, reducing
datasets, training models — all from a high-level language, on any GPU.

No CUDA. No driver lock-in. Just write `.flow`, and the loom weaves.

```flow
use stdlib.loom.sieve
let count = gpu_prime_count(1000000000.0)
print("Primes below one billion: {count:.0}")
// → 50,847,534 (exact, verified)
```

Three lines. 50 million primes. Under 3 seconds on a mid-range GPU.

---

## How It Works

### The Weaving Metaphor

A textile loom takes many threads and weaves them into fabric following a pattern.
OctoFlow's Loom does the same:

| Textile Loom | OctoFlow Loom |
|---|---|
| Threads | GPU threads (thousands) |
| Pattern | Dispatch chain (pre-recorded kernel sequence) |
| Shuttle | Push constants (data passed to each kernel) |
| Fabric | Computed result |
| Loom frame | Compute unit (GPU memory + command buffer) |

The key insight: **the entire pattern is set up before weaving begins.** In Loom,
you record all your kernel dispatches into a chain, compile it once, then launch.
The GPU executes the entire chain with zero CPU interruption.

This is why Loom can execute 95,000+ kernel dispatches in a single submission —
the pattern is pre-woven, the GPU just runs it.

### Three Tiers

```
┌─────────────────────────────────────────────────────┐
│  Tier 1: Patterns                                   │
│  gpu_sum(data)  ·  gpu_sort(data)  ·  gpu_sieve(N)  │
│  One function call. The loom handles everything.     │
├─────────────────────────────────────────────────────┤
│  Tier 2: Expert                                     │
│  loom_boot → loom_dispatch → loom_build → loom_run  │
│  Custom dispatch chains. You control the pattern.    │
├─────────────────────────────────────────────────────┤
│  Tier 3: IR                                         │
│  ir_begin → ir_fmul → ir_barrier → ir_finalize      │
│  Write custom SPIR-V kernels in .flow code.          │
└─────────────────────────────────────────────────────┘
```

Most users stay at Tier 1. Power users drop to Tier 2 for custom pipelines.
Kernel authors use Tier 3 to emit new SPIR-V compute shaders — from OctoFlow
itself, not from C++ or GLSL.

---

## Quick Start

### Tier 1: One-Call Patterns

```flow
// Parallel reduction
use stdlib.loom.reduce
let total = gpu_sum(data)
let biggest = gpu_max(data)

// Parallel sieve
use stdlib.loom.sieve
let primes = gpu_prime_count(10000000000.0)   // π(10^10), exact

// Parallel map
use stdlib.loom.map
let doubled = gpu_map(data, "mul", 2.0)
let roots = gpu_map(data, "sqrt")

// Matrix multiply
use stdlib.loom.matmul
let C = gpu_matmul(A, B, rows_a, cols_b, cols_a)
```

### Tier 2: Custom Dispatch Chains

```flow
// Boot a compute unit
let unit = loom_boot(1.0, 8194, 4096)

// Upload data
loom_write(unit, 0.0, my_primes)

// Record kernel dispatches (the "pattern")
loom_dispatch(unit, "stdlib/loom/kernels/sieve_init.spv", [seg, words, n_lo, n_hi], 32.0)
loom_dispatch(unit, "stdlib/loom/kernels/sieve_mark.spv", [seg, words, 0, 53], 32.0)
loom_dispatch(unit, "stdlib/loom/kernels/sieve_count.spv", [words, count_off], 32.0)

// Compile and launch
let prog = loom_build(unit)
loom_launch(prog)           // async — returns immediately

// Poll for completion
while loom_poll(prog) < 0.5
end

// Read results
let result = loom_read(unit, 0.0, 0.0, 8194)
```

### Tier 3: Custom SPIR-V Kernels via IR

```flow
use stdlib.loom.ir

// Build a GPU kernel that doubles every element
let mut prog = ir_begin()
let entry = ir_entry(prog)
let body = ir_block(prog)

let gid = ir_global_id(entry)
let val = ir_buf_load(body, 0.0, gid)
let doubled = ir_fmul(body, val, ir_const(body, 2.0))
ir_buf_store(body, 0.0, gid, doubled)

let kernel = ir_finalize(prog)

// Dispatch the custom kernel
let unit = loom_boot(1.0, 1024, 0)
loom_dispatch_jit(unit, kernel, [], 4.0)
let p = loom_build(unit)
loom_run(p)
let result = loom_read(unit, 0.0, 0.0, 1024)
```

You just wrote a GPU compute shader in OctoFlow. No GLSL, no HLSL, no CUDA.
The IR builder emits valid SPIR-V binary that runs on any Vulkan GPU.

---

## API Reference

### Core Functions

| Function | Purpose |
|---|---|
| `loom_boot(bind, reg_size, globals)` | Create a compute unit with register and globals buffers |
| `loom_write(unit, offset, data)` | Upload an array to the unit's globals buffer |
| `loom_dispatch(unit, kernel, params, wg)` | Record a kernel dispatch into the unit's chain |
| `loom_dispatch_jit(unit, ir, params, wg)` | Record a JIT-compiled kernel dispatch |
| `loom_build(unit)` | Compile the dispatch chain into a Vulkan command buffer |
| `loom_run(prog)` | Execute synchronously (blocks until complete) |
| `loom_launch(prog)` | Execute asynchronously (returns immediately) |
| `loom_poll(prog)` | Check if async execution has completed (1.0 = done) |
| `loom_read(unit, bind, off, len)` | Read results back from GPU memory |

### Pattern Functions

| Function | Pattern | Description |
|---|---|---|
| `gpu_sum(data)` | Reduce | Sum all elements |
| `gpu_min(data)` | Reduce | Find minimum element |
| `gpu_max(data)` | Reduce | Find maximum element |
| `gpu_map(data, op, ...)` | Map | Apply operation to every element |
| `gpu_sort(data)` | Sort | Parallel radix sort |
| `gpu_scan(data)` | Scan | Prefix sum (inclusive) |
| `gpu_prime_count(N)` | Sieve | Count primes below N (exact) |
| `gpu_matmul(A, B, m, n, k)` | MatMul | Matrix multiply: A is m×k, B is k×n, result is m×n |

### IR Builder Functions (Tier 3)

| Function | SPIR-V | Purpose |
|---|---|---|
| `ir_begin()` | — | Start a new kernel program |
| `ir_entry(prog)` | OpFunction | Create entry point |
| `ir_block(prog)` | OpLabel | Create a basic block |
| `ir_global_id(block)` | BuiltIn GlobalInvocationId | Get thread ID |
| `ir_const(block, val)` | OpConstant | Float constant |
| `ir_const_u(block, val)` | OpConstant | Uint32 constant |
| `ir_buf_load(block, bind, idx)` | OpAccessChain + OpLoad | Load from buffer |
| `ir_buf_store(block, bind, idx, val)` | OpAccessChain + OpStore | Store to buffer |
| `ir_fadd`, `ir_fsub`, `ir_fmul`, `ir_fdiv` | OpFAdd/Sub/Mul/Div | Float arithmetic |
| `ir_iadd`, `ir_isub`, `ir_imul` | OpIAdd/Sub/Mul | Integer arithmetic |
| `ir_shl`, `ir_shr`, `ir_not` | OpShift/OpNot | Bitwise operations |
| `ir_bitcount(block, val)` | OpBitCount | Hardware popcount |
| `ir_buf_atomic_and(block, bind, idx, mask)` | OpAtomicAnd | Atomic bit-clear |
| `ir_barrier(block)` | OpControlBarrier | Workgroup sync |
| `ir_shared_load`, `ir_shared_store` | Workgroup memory | Shared memory ops |
| `ir_u32_to_u64`, `ir_u64_to_u32` | OpUConvert | 64-bit widening/narrowing |
| `ir_imul64`, `ir_iadd64`, `ir_udiv64` | 64-bit OpIMul/IAdd/UDiv | 64-bit arithmetic |
| `ir_finalize(prog)` | — | Emit SPIR-V binary |

---

## Architecture

### The Dispatch Chain Model

```
Record Phase                    Execute Phase
─────────────                   ─────────────
loom_boot()         ─┐
loom_write()         │
loom_dispatch() ×N   │── chain ──▶  loom_build() ──▶ loom_launch()
loom_dispatch() ×N   │                                    │
loom_dispatch() ×N  ─┘                                    ▼
                                                    GPU executes
                                                    entire chain
                                                    (zero CPU trips)
                                                          │
                                                          ▼
                                                    loom_read()
```

All kernel dispatches are **pre-recorded** into a chain. The chain is compiled
into a single Vulkan command buffer and submitted once. The GPU executes every
dispatch back-to-back without returning to the CPU.

At scale, this matters enormously. OctoFlow's prime sieve at 10^10 runs
95,370 dispatches in a single submission. CUDA frameworks need a CPU round-trip
for each kernel launch (~5-20us each). Loom's chains eliminate that overhead
entirely.

### Async VM Swarm

```
       ┌── Unit #0  ─── chain of 5,960 dispatches ──▶ GPU
       ├── Unit #1  ─── chain of 5,960 dispatches ──▶ GPU
       ├── Unit #2  ─── chain of 5,960 dispatches ──▶ GPU
CPU ───┤   ...                                        ╲
       ├── Unit #14 ─── chain of 5,960 dispatches ──▶  ▶ parallel
       └── Unit #15 ─── chain of 5,960 dispatches ──▶ GPU
```

Multiple compute units run simultaneously via `loom_launch()`. Each unit has
its own register buffer and dispatch chain. The CPU boots all units, records
all chains, then launches everything — no coordination overhead during execution.

### Memory Layout (per unit)

```
Registers (B0):
┌──────────────────────────────────────────────────┐
│ [0 .. NUM_WORDS-1]          Bitmap / work buffer │
│ [NUM_WORDS]                 Count scratch        │
│ [NUM_WORDS+1]               Accumulator          │
│ [NUM_WORDS+2 .. +NP]        Carry-forward state  │
└──────────────────────────────────────────────────┘

Globals (read-only, shared input):
┌──────────────────────────────────────────────────┐
│ [0 .. num_primes-1]         Input data (primes)  │
│ [num_primes .. pad]         Padding to 256       │
└──────────────────────────────────────────────────┘
```

Registers are read-write (GPU workspace). Globals are uploaded once and read
by all dispatches. Push constants (up to 5 floats) carry per-dispatch parameters
like segment index and array bounds.

---

## Proven at Scale

### Prime Sieve: Seven Generations

| Scale | Result | GPU Time | Total | Status |
|---|---|---|---|---|
| pi(10^7) | 664,579 | 10ms | 234ms | EXACT |
| pi(10^8) | 5,761,455 | 96ms | 507ms | EXACT |
| pi(10^9) | 50,847,536 | 792ms | 2,523ms | EXACT |
| pi(10^10) | 455,052,512 | 7,843ms | 22,400ms | EXACT |

Hardware: GTX 1660 SUPER (mid-range, 6GB). All results verified against
known prime counting function values.

### The Journey: v1 to v7

| Version | Key Innovation | pi(10^9) GPU | Speedup |
|---|---|---|---|
| v1 | f32 per-element, trial division | 65,000ms | baseline |
| v2 | Bit-packed uint32, hardware popcount | 1,350ms | 48x |
| v3 | L1-sized segments, shared memory, bucket sieve | 765ms | 1.8x |
| v4 | Carry-forward offsets, selective JIT | 764ms | ~1x |
| v5 | Runtime SPIR-V synthesis | 760ms | ~1x |
| v6 | uint64 addressing (breaks 4B wall) | 775ms | — |
| v7 | Sentinel carry-forward + uint64 | 792ms | (at 10^10+) |

From 65 seconds to under 1 second. 1000x less VRAM. Exact at 10 billion.

### GPU Patterns Used

| Pattern | Sieve Usage | Transferable To |
|---|---|---|
| Bit-packing (32 bools/word) | Prime bitmap | Bloom filters, image masks, graph adjacency |
| L1-sized segmentation (32KB) | Cache-hot sieve | Any streaming computation |
| Shared memory prime cache | Small prime marking | Cooperative loading for any shared data |
| Tree reduction (shared mem) | Parallel popcount | Sum, min, max, histogram |
| Atomic AND (bit-clear) | Composite marking | Lock-free set operations |
| Carry-forward state | Resume across segments | Iterative solvers, streaming aggregation |
| uint64 arithmetic | Address beyond 4B | Large-scale indexing, cryptographic ops |
| Async swarm (16 units) | Parallel segments | Any embarrassingly parallel workload |

---

## Standard Library

### Domain-Organized Modules

```
stdlib/loom/
├── core/                     Engine runtime
│   ├── boot.flow             Unit lifecycle
│   ├── dispatch.flow         Kernel dispatch + chain recording
│   └── monitor.flow          Profiling + diagnostics
│
├── ir.flow                   Kernel authoring (Tier 3) — re-exports compiler IR
│
├── patterns/                 One-call GPU compute (Tier 1)
│   ├── reduce.flow           gpu_sum, gpu_min, gpu_max
│   ├── map.flow              gpu_map (element-wise transforms)
│   ├── scan.flow             gpu_scan (prefix sum)
│   ├── sort.flow             gpu_sort (radix sort)
│   ├── sieve.flow            gpu_prime_count (parallel sieve)
│   └── matmul.flow           gpu_matmul (tiled matrix multiply)
│
├── math/                     Numerical computation
│   ├── linalg.flow           Linear algebra (dot, cross, normalize)
│   ├── stats.flow            Statistical operations on GPU
│   ├── signal.flow           Signal processing (FFT, convolution)
│   └── advanced.flow         Special functions (gamma, bessel)
│
├── nn/                       Neural network primitives
│   ├── attention.flow        Multi-head attention
│   ├── ffn.flow              Feed-forward layers
│   ├── rmsnorm.flow          RMS normalization
│   ├── rope.flow             Rotary position embedding
│   ├── silu.flow             SiLU activation
│   ├── softmax.flow          Softmax
│   ├── matmul_tiled.flow     Tiled GEMM
│   └── dequant.flow          Quantization (Q4_K, Q6_K)
│
├── data/                     Data-parallel operations
│   ├── array_ops.flow        GPU array operations
│   ├── aggregate.flow        Group-by, histogram
│   ├── composite.flow        Multi-step data pipelines
│   └── dlb_scan.flow         Load-balanced parallel scan
│
├── kernels/                  Pre-compiled SPIR-V binaries
│   ├── math/                 abs, add, sqrt, sin, cos, ...
│   ├── reduce/               reduce_sum, reduce_min, reduce_max
│   ├── sieve/                sieve_init, sieve_mark, sieve_count, ...
│   ├── nn/                   matvec, rmsnorm, rope, silu, softmax, ...
│   └── vm/                   vm_add, vm_scale, vm_relu, ...
│
├── emit/                     Kernel emitters (.flow → .spv)
│   ├── sieve/                Sieve kernel emitters (v1-v7)
│   ├── nn/                   Neural net kernel emitters
│   └── ops/                  Math operation kernel emitters
│
└── tests/                    Test suite
    ├── test_bitwise_ir.flow
    ├── test_uint64_ir.flow
    ├── test_sieve.flow
    ├── test_reduce.flow
    ├── test_nn_kernels.flow
    └── ...
```

### Design Principles

**Serve the dish, not the recipe.**

- `use stdlib.loom.sieve` gives you `gpu_prime_count(N)` — one call, exact result
- `use stdlib.loom.reduce` gives you `gpu_sum(data)` — not `boot + write + dispatch + build + launch + poll + read`
- `use stdlib.loom.nn.attention` gives you `gpu_attention(Q, K, V)` — not a 200-line dispatch chain

The patterns hide the machinery. The expert API exposes it when you need it.

**LLM-first naming.**

Every function name is guessable from its description:
- "sum this data on GPU" → `gpu_sum(data)`
- "count primes below N" → `gpu_prime_count(N)`
- "multiply matrices A and B" → `gpu_matmul(A, B, m, n, k)`

An LLM generating OctoFlow code should never need to read documentation to
find the right function name. If it can describe what it wants, it can guess
the function.

**Domain grouping, not implementation grouping.**

Old: `stdlib/gpu/emit_sieve_mark_v3_large.flow` (organized by what it IS)
New: `stdlib/loom/emit/sieve/mark_v3_large.flow` (organized by what it DOES)

Old: `stdlib/gpu/emit_vm_rmsnorm.flow` (mixed with sieve code)
New: `stdlib/loom/emit/nn/rmsnorm.flow` (with other neural net emitters)

Users find things by domain (sieve, neural nets, reduction), not by
implementation detail (emit, vm, v3).

---

## Under the Hood

### SPIR-V: The Fabric

Every Loom kernel compiles to SPIR-V — the standard intermediate language for
Vulkan compute shaders. SPIR-V is:
- **Vendor-neutral**: Runs on NVIDIA, AMD, Intel, Qualcomm, ARM Mali
- **Binary**: No runtime compilation step (unlike GLSL)
- **Validatable**: `spirv-val` checks correctness before GPU touches it

OctoFlow emits SPIR-V directly — no GLSL, no HLSL, no intermediate language.
The IR builder (`stdlib/loom/ir/ir.flow`) is itself written in OctoFlow: the
language writes its own GPU kernels.

### Vulkan: The Frame

Loom sits on Vulkan Compute via the `ash` crate (thin Rust bindings):
- **Command buffers**: Pre-recorded dispatch chains compiled to GPU-native commands
- **Descriptor sets**: Buffer bindings (registers, globals) set once per unit
- **Push constants**: Small per-dispatch parameters (up to 5 × 32-bit values)
- **Memory barriers**: Automatic between dispatches (no manual sync)

### f32 Precision Engineering

OctoFlow uses `Value::Float(f32)` for all values. f32 has a 24-bit mantissa —
exact integers only to 2^24 = 16,777,216. Loom's proven solutions:

| Challenge | Solution |
|---|---|
| Large uint32 constants | Compute on GPU: `ir_not(c0)` = 0xFFFFFFFF |
| Push constants > 2^24 | Pass small inputs, GPU computes full value |
| N > 2^24 (addressing) | Split: `N_hi × 2^24 + N_lo`, reconstruct in uint64 |
| Accumulation overflow | GPU accumulates in uint32, readback via `float_to_bits()` |
| NaN bit patterns | Never interpret raw uint32 as f32; use indirect computation |
| Boundary precision | Sentinel design: GPU-side is authoritative, CPU-side is advisory |

These patterns are hard-won through seven generations of GPU sieve development
and apply to any Loom program working with large integers or addresses.

---

## Icon: The Octopus Weaving

```
        ╭───────────────────────────╮
        │                           │
        │      🐙                   │
        │    Eight arms             │
        │    working the loom       │
        │                           │
        │    ═══╪═══╪═══╪═══       │
        │    ───┼───┼───┼───       │
        │    ═══╪═══╪═══╪═══       │
        │    ───┼───┼───┼───       │
        │    ═══╪═══╪═══╪═══       │
        │                           │
        │    Threads woven          │
        │    into fabric            │
        │                           │
        ╰───────────────────────────╯
```

**Visual concept**: An octopus sitting at a loom, its eight arms each pulling
a different thread through the weave. The warp threads (vertical) are data
streams. The weft threads (horizontal) are GPU operations. The fabric that
emerges is the computed result.

**Icon elements:**
- Octopus (brand identity) in profile view, working a loom
- Loom frame with visible thread grid (suggests parallel structure)
- Gradient from raw threads (left) to woven fabric (right) — input to output
- Eight arms visible, each engaged with different part of the weave

**Color palette:**
- Deep ocean blue (#1a3a5c) — background
- Warm amber (#f0a030) — thread/compute highlights
- Silver (#c0c0cc) — loom frame
- White (#ffffff) — fabric/output

**Tagline options:**
- "Weaving parallel computation"
- "Eight arms. Thousands of threads. One loom."
- "The GPU runtime for the rest of us"

---

## Getting Started

### Prerequisites

- OctoFlow compiler (latest)
- Vulkan-capable GPU with driver
- Vulkan SDK (for `spirv-val` validation, optional)

### Your First Loom Program

```flow
// hello_loom.flow — double every element on GPU

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

// Boot a compute unit (1 buffer, 8 registers, no globals)
let unit = loom_boot(1.0, 8, 0)

// Dispatch the "double" kernel
loom_dispatch(unit, "stdlib/loom/kernels/math/double.spv", [], 1.0)

// Build, run, read
let prog = loom_build(unit)
loom_run(prog)
let result = loom_read(unit, 0.0, 0.0, 8)

print("Input:  {data}")
print("Output: {result}")
// → [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
```

### Running It

```bash
octoflow run hello_loom.flow --allow-read
```

---

## LoomDB — GPU-Resident Data Layer

LoomDB captures GPU pipeline results and makes them searchable — without
touching disk. It runs in its own Loom, completely isolated from your main
compute pipeline.

### The Two-Loom Pattern

```
┌──────────────────────────────────┐
│  MAIN LOOM (zero I/O)            │
│                                  │
│  Compute: loom_dispatch chains   │
│  Capture: loomdb_capture()       │  GPU memory only
│  Search:  loomdb_search()        │  No file syscalls
│           loomdb_gpu_search()    │  No network calls
└────────────────┬─────────────────┘
                 │ shared VRAM
┌────────────────┴─────────────────┐
│  LOOMDB LOOM (owns all I/O)      │
│                                  │
│  Persist: loomdb_flush()         │  Writes .ldb + .vectors + .meta
│  Restore: loomdb_restore_*()     │  Loads from disk at startup
│  Never blocks the Main Loom.     │
└──────────────────────────────────┘
```

The Main Loom physically cannot do I/O — none of the capture/search functions
contain file operations. This isn't a convention; it's architectural enforcement.

### Quick Start

```flow
use "db/loomdb"

// Create: 384-dimensional embeddings, auto-flush at 10,000
let ldb = loomdb_create(384.0, 10000.0)
let vectors = loomdb_create_vectors()

// Capture results (GPU memory, zero I/O)
let _c1 = loomdb_capture(ldb, vectors, "emb_001", embedding_a, "batch=42")
let _c2 = loomdb_capture(ldb, vectors, "emb_002", embedding_b, "batch=42")
let _c3 = loomdb_capture(ldb, vectors, "emb_003", embedding_c, "batch=43")

// Search (GPU memory, zero I/O)
let _s = loomdb_search(ldb, vectors, query_vector, 5.0, "cosine")

// Read results
let count = loomdb_result_count(ldb)
for i in range(0, count)
  let id = loomdb_result_id(ldb, i)
  let score = loomdb_result_score(ldb, i)
  print("{id}: {score}")
end

// GPU-accelerated search (fast for 1000+ vectors)
let _g = loomdb_gpu_search(ldb, vectors, query_vector, 10.0)
```

### Persistence

```flow
// Check capacity
if loomdb_needs_flush(ldb) == 1.0
  let _f = loomdb_flush(ldb, vectors, "cache/embeddings")
end

// Restore on next startup
let ldb = loomdb_restore_meta("cache/embeddings")
let vectors = loomdb_restore_vectors("cache/embeddings")
// GPU-resident again, ready to search
```

### LoomDB API

| Function | I/O? | Description |
|----------|------|-------------|
| `loomdb_create(dims, cap)` | No | Create instance (dims = embedding size, cap = flush threshold) |
| `loomdb_create_vectors()` | No | Create empty vectors array |
| `loomdb_capture(ldb, vecs, id, emb, meta)` | No | Capture a vector to GPU memory |
| `loomdb_search(ldb, vecs, q, k, metric)` | No | CPU similarity search (cosine/dot/euclidean) |
| `loomdb_gpu_search(ldb, vecs, q, k)` | No | GPU-accelerated search via gpu_matmul |
| `loomdb_needs_flush(ldb)` | No | Check if capacity threshold reached |
| `loomdb_normalize(ldb, vecs)` | No | Pre-normalize for faster cosine search |
| `loomdb_result_count(ldb)` | No | Number of search results |
| `loomdb_result_id(ldb, i)` | No | Result ID at position i |
| `loomdb_result_score(ldb, i)` | No | Result score at position i |
| `loomdb_result_meta(ldb, i)` | No | Result metadata at position i |
| `loomdb_flush(ldb, vecs, path)` | **Yes** | Persist to .ldb + .vectors + .meta |
| `loomdb_restore_meta(path)` | **Yes** | Load metadata from disk |
| `loomdb_restore_vectors(path)` | **Yes** | Load vectors from disk |

---

## OctoDB — Structured Data Storage

OctoDB is OctoFlow's embedded database for structured data — tables, rows,
CRUD operations, and `.odb` file persistence. It also serves as LoomDB's
cold storage tier.

### Quick Start

```flow
use "db/core"
use "db/engine"

// Create a table
let db = db_create()
let users = db_table(db, "users", ["name", "age", "email"])

// Insert
let mut row = map()
row["name"] = "Alice"
row["age"] = 30.0
row["email"] = "alice@example.com"
let _i = db_insert(users, row)

// Query
let indices = db_where(users, "age", ">", 25.0)
let avg_age = db_aggregate(users, "age", "avg")

// Multi-condition
let results = db_select(users, ["age", "name"], [">", "contains"], [25.0, "Ali"])

// Persist
let _s = db_save(users, "data/users.odb")
let restored = db_load("data/users.odb")
```

### OctoDB API

| Function | Description |
|----------|-------------|
| `db_create()` | Create database |
| `db_table(db, name, columns)` | Create table |
| `db_insert(table, row)` | Insert row (returns index) |
| `db_select_row(table, idx)` | Get row by index |
| `db_select_column(table, col, n)` | Get first n values of a column |
| `db_where(table, col, op, val)` | Filter rows (==, !=, >, <, >=, <=, contains) |
| `db_select(table, cols, ops, vals)` | Multi-condition AND filter |
| `db_update(table, idx, row)` | Update row fields |
| `db_delete(table, idx)` | Soft delete |
| `db_count(table)` | Row count |
| `db_distinct(table, col)` | Unique values |
| `db_aggregate(table, col, op)` | sum, avg, min, max, count |
| `db_save(table, path)` / `db_load(path)` | Single-table persistence (.odb) |
| `db_import_csv(table, path)` | Import CSV into table |

For multi-table persistence: `db_save_all_start`, `db_save_all_add`, `db_load_all`
(see `stdlib/db/persist.flow`).

---

## The Two-Tier Pattern

OctoDB and LoomDB work together:

```
GPU Memory (LoomDB)          Disk (OctoDB)
┌───────────────────┐        ┌──────────────┐
│ loomdb_capture()  │        │ .odb files   │
│ loomdb_search()   │ flush  │ .ldb files   │
│ loomdb_gpu_search │ ────>  │ .vectors     │
│                   │        │ .meta        │
│ Source of truth   │ <────  │ Cold storage │
│ during runtime    │restore │ between runs │
└───────────────────┘        └──────────────┘
```

**Use OctoDB** for structured data: user tables, config, logs, CSV imports.
**Use LoomDB** for GPU pipeline results: embeddings, features, similarity search.
**Use both** when you need GPU-speed search with disk persistence between sessions.

---

## Roadmap

| Phase | What | Status |
|---|---|---|
| Loom Core | boot, dispatch, build, run, launch, poll, read | **Done** |
| IR Builder | 60+ SPIR-V ops including uint64, atomics, shared memory | **Done** |
| Prime Sieve | v1-v7, exact to 10^10, 95K dispatches | **Done** |
| Neural Net Kernels | attention, ffn, rmsnorm, rope, silu, softmax, matvec | **Done** |
| API Rename | vm_* → loom_* function aliases | **Done** |
| LoomDB | GPU-resident data layer with I/O isolation | **Done** |
| OctoDB | Structured CRUD with .odb persistence | **Done** |
| Two-Tier DB | LoomDB + OctoDB integration | **Done** |
| Pattern Library | gpu_sum, gpu_sort one-call wrappers | Partial |
| Console Monitor | loom_profile_start/end, timing, VRAM stats | Planned |
| Multi-GPU Swarm | Network dispatch across machines | Future |
| Compiled Chains | Eliminate interpreter bottleneck for dispatch recording | Future |

---

## Why "Loom"?

- A **loom** weaves many threads into fabric — we weave GPU threads into results
- **Threads** are the fundamental unit of both textiles and GPU compute
- The **dispatch chain** is the pattern — pre-recorded, then woven in one pass
- The **octopus** works the loom with eight arms — our brand, our architecture
- Four characters. Zero ecosystem collision. Immediately evocative.

The loom weaves. The octopus works the loom. The fabric is your result.
