# Annex W: The Loom Engine — OctoFlow's GPU Compute Runtime

> Weaving threads of parallel computation.

**Date:** 2026-02-25
**Status:** Design document — naming, API, pattern library, migration plan

---

## Why "Loom"

The GPU compute runtime was called "GPU VM" — generic, invites comparison to
JVM/CLR/WASM virtual machines, and doesn't convey what it actually does: weave
thousands of parallel threads into coordinated computation.

**Loom** captures the essence:
- A loom weaves many threads into fabric — our engine weaves GPU threads into results
- Threads are the fundamental unit of both textiles and GPU compute
- The dispatch chain is the pattern — pre-recorded, then woven in one pass
- The octopus metaphor: eight arms working the loom simultaneously

**Brand fit:** OctoFlow's Loom Engine. Four characters, distinctive, zero ecosystem
collision, immediately evocative of parallel threading.

---

## API Rename

### Core Functions

| Current (GPU VM) | New (Loom) | Purpose |
|---|---|---|
| `vm_boot(bind, reg_size, globals_size)` | `loom_boot(bind, reg_size, globals_size)` | Create a compute unit |
| `vm_write_globals(unit, offset, data)` | `loom_write(unit, offset, data)` | Upload data to unit |
| `vm_dispatch(unit, kernel, params, wg)` | `loom_dispatch(unit, kernel, params, wg)` | Record a kernel dispatch |
| `vm_dispatch_mem(unit, ir_prog, params, wg)` | `loom_dispatch_jit(unit, ir_prog, params, wg)` | Record a JIT kernel dispatch |
| `vm_build(unit)` | `loom_build(unit)` | Compile dispatch chain |
| `vm_execute(prog)` | `loom_run(prog)` | Execute synchronously |
| `vm_execute_async(prog)` | `loom_launch(prog)` | Execute asynchronously |
| `vm_poll(prog)` | `loom_poll(prog)` | Check async completion |
| `vm_read_register(unit, bind, off, len)` | `loom_read(unit, bind, off, len)` | Read results back |

### Naming Principles

1. **Short verbs**: `boot`, `write`, `dispatch`, `build`, `run`, `launch`, `poll`, `read`
2. **No redundancy**: `loom_read` not `loom_read_register` — the unit handle implies context
3. **Industry terms**: `launch` (async start) and `dispatch` (kernel invoke) are standard GPU terminology
4. **JIT explicit**: `loom_dispatch_jit` instead of `loom_dispatch_mem` — makes it clear this is runtime-compiled

### How It Reads

```flow
// Boot 16 compute units
let mut units = []
let mut i = 0.0
while i < 16.0
    let unit = loom_boot(1.0, 8194, 4096)
    loom_write(unit, 0.0, primes)
    push(units, unit)
    i = i + 1.0
end

// Record dispatch chains
loom_dispatch(unit, "stdlib/loom/kernels/sieve_init.spv", params, 32.0)
loom_dispatch(unit, "stdlib/loom/kernels/sieve_mark.spv", params, 32.0)
loom_dispatch(unit, "stdlib/loom/kernels/sieve_count.spv", params, 32.0)

// Build and launch
let prog = loom_build(unit)
loom_launch(prog)

// Poll and read
while loom_poll(prog) < 0.5
end
let result = loom_read(unit, 0.0, 0.0, 8194)
```

An LLM reading this code immediately understands: boot compute units, record work,
build, launch, poll, read. No "VM" abstraction to explain.

---

## Directory Structure

### Current → New

```
stdlib/gpu/                          → stdlib/loom/
stdlib/gpu/kernels/                  → stdlib/loom/kernels/
stdlib/gpu/tests/                    → stdlib/loom/tests/
stdlib/gpu/emit_sieve_*.flow         → stdlib/loom/emit_sieve_*.flow
stdlib/compiler/ir.flow              → stdlib/loom/ir.flow
```

### New Layout

```
stdlib/loom/
├── ir.flow                          # IR builder (SPIR-V instruction emitter)
├── patterns/                        # Pre-built compute patterns (library)
│   ├── sieve.flow                   # GPU parallel sieve (one-call API)
│   ├── reduce.flow                  # Parallel reduction (sum, min, max)
│   ├── scan.flow                    # Prefix sum (Hillis-Steele)
│   ├── sort.flow                    # Parallel radix sort
│   ├── matmul.flow                  # Matrix multiplication
│   └── map.flow                     # Parallel map (element-wise transform)
├── kernels/                         # Pre-compiled SPIR-V binaries
│   ├── sieve_init_v6.spv
│   ├── sieve_mark_v6_small.spv
│   ├── sieve_mark_v7_large.spv
│   ├── sieve_init_offsets_v7.spv
│   ├── sieve_count_v3.spv
│   ├── sieve_accum_v2.spv
│   ├── reduce_sum.spv
│   ├── reduce_min.spv
│   ├── reduce_max.spv
│   └── ...
├── emit/                            # Kernel emitters (.flow → .spv)
│   ├── emit_sieve_init_v6.flow
│   ├── emit_sieve_mark_v6_small.flow
│   ├── emit_sieve_mark_v7_large.flow
│   ├── emit_sieve_init_offsets_v7.flow
│   ├── emit_sieve_count_v3.flow
│   ├── emit_sieve_accum_v2.flow
│   └── ...
└── tests/
    ├── test_bitwise_ir.flow
    ├── test_uint64_ir.flow
    ├── test_sieve.flow
    └── test_reduce.flow
```

---

## Pattern Library: One-Call GPU Compute

The Loom pattern library compiles common GPU operations into single-function-call
modules. Each pattern handles booting, dispatch chain recording, execution, and
result readback internally.

### Design Philosophy

```
Expert API:     loom_boot → loom_write → loom_dispatch × N → loom_build → loom_launch → loom_read
Pattern API:    let result = gpu_sieve(N)
```

The expert API stays for custom kernels. The pattern API wraps it for common operations.
LLMs generate pattern-level code by default, dropping to expert API only when needed.

### Pattern: Parallel Sieve

```flow
use stdlib.loom.patterns.sieve

// One call — handles everything internally
let count = gpu_prime_count(1000000000.0)       // π(10^9) = 50,847,534
print("Primes below 10^9: {count:.0}")

// With options
let count = gpu_prime_count(10000000000.0)      // π(10^10)
```

Internally: boots 16 units, CPU-sieves to √N, uploads primes, records v7 dispatch
chains (init → mark_small → init_offsets → mark_large → count → accum), launches
async, polls, tallies. The user sees one function call.

### Pattern: Parallel Reduction

```flow
use stdlib.loom.patterns.reduce

let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
let total = gpu_sum(data)          // 31.0
let biggest = gpu_max(data)        // 9.0
let smallest = gpu_min(data)       // 1.0
```

### Pattern: Parallel Map

```flow
use stdlib.loom.patterns.map

let data = range_array(1.0, 1000000.0)
let squares = gpu_map(data, "sqrt")           // built-in op
let scaled = gpu_map(data, "mul", 2.5)        // element × 2.5
let clamped = gpu_map(data, "clamp", 0.0, 1.0)
```

### Pattern: Parallel Sort

```flow
use stdlib.loom.patterns.sort

let data = [3.0, 1.0, 4.0, 1.0, 5.0]
let sorted = gpu_sort(data)        // [1.0, 1.0, 3.0, 4.0, 5.0]
```

### Pattern: Matrix Multiply

```flow
use stdlib.loom.patterns.matmul

// A: 1024×512, B: 512×768 → C: 1024×768
let C = gpu_matmul(A, B, 1024, 768, 512)
```

---

## LLM-Friendly Design

### Why This Matters

OctoFlow's Layer 3 is an LLM frontend — natural language → .flow code. The Loom
API must be easy for LLMs to generate correctly.

### Design Rules for LLM Generation

1. **Consistent prefix**: Every Loom function starts with `loom_` (expert) or `gpu_` (pattern).
   LLMs learn prefixes fast — one pattern covers all functions.

2. **Minimal parameters**: `gpu_sum(data)` not `gpu_reduce(data, "sum", 0.0, 256, 16)`.
   Sensible defaults for everything. Expert overrides via optional params.

3. **No mode flags**: `gpu_sum` / `gpu_max` / `gpu_min` — separate functions, not
   `gpu_reduce(data, mode="sum")`. LLMs generate wrong mode strings; separate
   functions have zero ambiguity.

4. **Return values, not output params**: `let result = gpu_sort(data)` not
   `gpu_sort(data, output_buffer)`. Functional style is what LLMs default to.

5. **Errors via try()**: Pattern functions use OctoFlow's standard error handling.
   ```flow
   let r = try(gpu_prime_count(N))
   if r.ok > 0.5
       print("Count: {r.value:.0}")
   else
       print("GPU error: {r.error}")
   end
   ```

6. **Discoverable names**: `gpu_prime_count`, `gpu_sum`, `gpu_sort`, `gpu_matmul` —
   an LLM can guess the function name from the description. No abbreviations,
   no internal jargon.

### LLM Prompt Example

User: "Count the primes below one billion using the GPU"

LLM generates:
```flow
use stdlib.loom.patterns.sieve
let count = gpu_prime_count(1000000000.0)
print("Primes below 10^9: {count:.0}")
```

Three lines. No boot/dispatch/build/launch/poll/read ceremony. The LLM doesn't
need to understand SPIR-V, Vulkan, dispatch chains, or bitmap encoding.

---

## Loom Monitor (UI)

### Console Profiler

```flow
use stdlib.loom.monitor

loom_profile_start()

// ... GPU work ...
let count = gpu_prime_count(1000000000.0)

let stats = loom_profile_end()
print("GPU time:    {stats.gpu_ms:.1}ms")
print("Dispatches:  {stats.dispatches:.0}")
print("VRAM used:   {stats.vram_kb:.0}KB")
print("Units used:  {stats.units:.0}")
print("Throughput:  {stats.ops_per_sec:.0} ops/s")
```

### Live Dashboard (future, requires ext.ui)

```
┌─ Loom Engine ──────────────────────────────────────┐
│ Units: 16/16 active    VRAM: 1.7MB / 6144MB       │
│ Dispatches: 95,370     Chain: 5,960/unit           │
│ GPU Time: 7,843ms      Status: RUNNING ████████░░  │
│                                                     │
│ Unit  Segs   Dispatches  Status                    │
│ #0    597    2,985       DONE ✓                    │
│ #1    597    2,985       DONE ✓                    │
│ #2    597    2,985       RUNNING ██████░░          │
│ ...                                                │
│ #15   594    2,970       QUEUED                    │
└────────────────────────────────────────────────────┘
```

---

## IR Builder in Loom Context

The IR builder (`ir.flow`) moves from `stdlib/compiler/` to `stdlib/loom/` because
it's the kernel authoring tool — you use it to write custom SPIR-V kernels.

### Expert: Custom Kernel via IR

```flow
use stdlib.loom.ir

// Build a custom kernel
let mut prog = ir_begin()
let entry = ir_entry(prog)
let body = ir_block(prog)

let gid = ir_global_id(entry)
let val = ir_buf_load(body, 0.0, gid)
let result = ir_fmul(body, val, ir_const(body, 2.0))
ir_buf_store(body, 0.0, gid, result)

let kernel = ir_finalize(prog)

// Use with Loom
let unit = loom_boot(1.0, 1024, 0)
loom_dispatch_jit(unit, kernel, [], 4.0)
let p = loom_build(unit)
loom_run(p)
```

### Tiered Complexity

```
Level 1 (Pattern):   gpu_sum(data)                          ← LLM default
Level 2 (Expert):    loom_boot → loom_dispatch → loom_run   ← custom pipelines
Level 3 (IR):        ir_begin → ir_fmul → ir_finalize       ← custom SPIR-V kernels
```

Most users (and LLMs) stay at Level 1. Power users drop to Level 2 for custom
dispatch chains. Kernel authors use Level 3 to emit new SPIR-V patterns.

---

## Migration Plan

### Phase 1: Rename Runtime Functions (in Rust evaluator)

Add `loom_*` functions as aliases for `vm_*` in `flowgpu-cli/src/compiler.rs`
(or wherever the evaluator dispatches built-in functions). Keep `vm_*` working
as deprecated aliases for backward compatibility.

| Old | New | Evaluator change |
|-----|-----|-----------------|
| `vm_boot` | `loom_boot` | Add match arm |
| `vm_dispatch` | `loom_dispatch` | Add match arm |
| `vm_dispatch_mem` | `loom_dispatch_jit` | Add match arm + rename |
| `vm_build` | `loom_build` | Add match arm |
| `vm_execute` | `loom_run` | Add match arm |
| `vm_execute_async` | `loom_launch` | Add match arm |
| `vm_poll` | `loom_poll` | Add match arm |
| `vm_read_register` | `loom_read` | Add match arm |
| `vm_write_globals` | `loom_write` | Add match arm |

### Phase 2: Move stdlib Files

```bash
git mv stdlib/gpu/ stdlib/loom/
mkdir stdlib/loom/patterns/
mkdir stdlib/loom/emit/
git mv stdlib/loom/emit_*.flow stdlib/loom/emit/
git mv stdlib/compiler/ir.flow stdlib/loom/ir.flow
```

Update all kernel path references in example .flow files.

### Phase 3: Build Pattern Library

Implement one-call wrappers for existing patterns:
1. `gpu_prime_count(N)` — wraps sieve v7
2. `gpu_sum(data)` / `gpu_min` / `gpu_max` — wraps existing reduce patterns
3. `gpu_map(data, op)` — wraps existing map patterns

### Phase 4: Update Documentation

- Rename "GPU VM" → "Loom Engine" across all docs
- Update CLAUDE.md operations reference
- Update gpu-sieve.md, gpu-vm-learnings.md
- Update CODING-GUIDE.md

### Phase 5: Deprecate vm_* (future)

After all examples and stdlib are migrated, emit deprecation warnings for `vm_*`
functions. Remove in a later phase.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Loom** | OctoFlow's GPU compute runtime — weaves parallel threads into results |
| **Unit** | A single compute unit (was "VM") — owns a register buffer and globals |
| **Dispatch** | A single kernel invocation recorded into a unit's chain |
| **Chain** | A sequence of dispatches pre-recorded for batch execution |
| **Pattern** | A pre-built compute operation (sieve, reduce, sort) wrapped as a single function |
| **Kernel** | A SPIR-V compute shader — the code that runs on GPU threads |
| **IR** | Intermediate representation — the instruction builder for custom kernels |
| **Weave** | (informal) The act of recording and executing a dispatch chain |

---

## Summary

The Loom Engine gives OctoFlow's GPU runtime a proper identity:

- **Name**: "Loom" — weaving parallel threads, four characters, zero collision
- **API**: `loom_boot` / `loom_dispatch` / `loom_launch` / `loom_read` — clean, short, standard
- **Patterns**: `gpu_prime_count(N)`, `gpu_sum(data)` — one-call GPU compute for LLMs
- **Three tiers**: Pattern (LLM default) → Expert (custom chains) → IR (custom kernels)
- **Migration**: Alias-first, backward compatible, phased rollout

The loom weaves. The octopus works the loom with eight arms.
