# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

OctoFlow — The first LLM-native, GPU-native programming language.

**Status: v1.3.** 210+ builtins, 246 stdlib modules, 1,340+ Rust tests, self-hosted compiler
(69% .flow). Loom Engine with 102 GPU kernels. LLM inference on GPU. Algorithm Space Exploration
Engine (ASE) for evolving algorithms on GPU. `octoflow chat`, `octoflow new`, OctoView, OctoUI.

## Current Phase: 4 — Adaptive Computing Fabric

**Pipeline:** 4A → 4A-H → F2 → ASE-F → F3 → ASE Demos → F4 → Swarm Sort → 4B OctoPress GPU Fractal DONE → Tier 1 Data Science DONE → **OctoSearch NEXT**

**Seven Primitives (all shipped):**

| # | Primitive | Status |
|---|-----------|--------|
| 1 | JIT Adaptive Kernels (IR builder, 80+ ops) | SHIPPED |
| 2 | On-Demand Main Looms (park/unpark, auto_spawn) | SHIPPED |
| 3 | Mailbox (ring buffer IPC) | SHIPPED |
| 4 | OctoPress (analysis + raw/delta/fractal encode/decode + .ocp format + GPU fractal) | SHIPPED |
| 5 | Multi-Stack Topologies | DOCUMENTED |
| 6 | Varying Bit Compute | DOCUMENTED |
| 7 | CPU Thread Pool (loom_threads, async_read, await) | SHIPPED |

**Active work:**
- Dev 1: Idle — Support Loom Threading DONE (7/7 tasks)
- Dev 2: OctoSearch Performance Fix — cached index, SPIR-V cache, park VM, faster top-K

**Recently shipped:**
- Support Loom Threading: SPIR-V file cache, non-blocking fence, queue mutex (18 sites), batch pacing, async present, batched upload, `loom_prefetch` builtin
- OctoSearch Infra: `walk_dir`/`is_dir`, `octoflow search` CLI, `ln`/`read` aliases, `file_mtime`, cross-module fn fix
- 4B-02: `octopress_gpu_encode(data, block_size)` — GPU fractal compression with CPU fallback
- 4B-01: GPU fractal domain search kernel (SPIR-V emitted at runtime)
- Tier 1: Data science batches 4-7 fixed (70+ `end`, 53 `append→push`, `&&`/`||` rewrites)
- GPU ML: `stdlib/ml/gpu_ml.flow` — 17 GPU-accelerated ML functions
- F4-04: `extend(dest, src)`, `array_copy(dest, d_off, src, s_off, count)`, `array_extract(src, off, count)`
- F4-02: `loom_pool_warm(count, instances, reg_size, globals_size)` — pre-boot VMs for swarm demos
- F4-03: OctoPress fractal method 2 (CPU) — IFS attractor fitting encode/decode
- SS-01: Swarm Sort Discovery — 721 lines, 8-gene genome, multi-VM cooperative sorting

---

## Key Documents

1. `docs/roadmap.md` — What's shipped, what's next
2. `docs/language-guide.md` — Complete language reference
3. `docs/loom-engine.md` — Loom Engine public-facing guide
4. `docs/loom-engine-phase4.md` — Phase 4 architecture (Loom Computer Model)
5. `docs/algorithm-space-engine.md` — ASE design + 68 application domains
6. `docs/builtins.md` — All 210+ built-in functions
7. `docs/llm/contract-v1.md` — LLM system prompt (~2,800 tokens)

---

## Repository Structure

```
OctoFlow/
├── Cargo.toml                    # Workspace root
├── run_test.ps1                  # MSVC/Vulkan SDK build environment (REQUIRED)
├── compiler/                     # octoflow-cli — main compiler + runtime
│   ├── src/
│   │   ├── lib.rs                # Value enum (Float, Int, Str, Map, None), CliError
│   │   ├── main.rs               # octoflow binary (run/check/repl/chat/new/build/update)
│   │   ├── runtime/mod.rs        # AST interpreter, 210+ builtins, GPU dispatch (~13K lines)
│   │   ├── runtime/tests.rs      # 1,340+ Rust tests
│   │   ├── chat/mod.rs           # octoflow chat AI assistant
│   │   ├── analysis/preflight.rs # Pre-flight validation (SCALAR_FNS, KNOWN_BUILTINS)
│   │   └── ...
│   ├── parser/                   # octoflow-parser — recursive descent
│   │   └── src/
│   │       ├── lib.rs            # parse() entry point
│   │       ├── ast.rs            # 28 Statement + 13 Expression types
│   │       └── lexer.rs          # Tokenizer
│   └── vulkan/                   # octoflow-vulkan — GPU compute runtime
│       └── src/
│           ├── device.rs         # VulkanCompute (instance, device, queue, Mutex)
│           ├── vm.rs             # Loom VM (create, dispatch, read, write)
│           ├── dispatch.rs       # GPU dispatch, 102 embedded kernels
│           └── vk_sys.rs         # Raw Vulkan C bindings (no ash)
├── stdlib/                       # 246 .flow modules across 18 domains
│   └── loom/
│       ├── ase/                  # Algorithm Space Exploration Engine
│       │   ├── genome.flow       # Genome create/randomize/read (65 lines)
│       │   ├── evolve.flow       # Evolution: selection + crossover + mutation (101 lines)
│       │   ├── fitness.flow      # Fitness: correctness + speed + combined (47 lines)
│       │   └── test_ase.flow     # 10 tests (219 lines)
│       ├── octopress.flow         # OctoPress .flow wrappers
│       └── ...
│   └── compiler/
│       └── ir.flow               # IR builder: 80+ ops, full SPIR-V emission
├── examples/                     # Demo programs
│   ├── ase_sort_discovery.flow   # GPU bitonic sort evolution (single VM)
│   ├── ase_bitwise_discovery.flow # Boolean circuit evolution (CPU)
│   ├── ase_hash_discovery.flow   # Hash function parameter evolution (CPU)
│   ├── ase_swarm_sort.flow       # Multi-VM cooperative sort (IN PROGRESS)
│   └── ...
├── octoui/                       # OctoUI widget toolkit (Windows)
├── OctoBrain/                    # GPU-native ML brain
└── docs/                         # Design documents
```

---

## Build and Test Commands

**CRITICAL:** On this Windows/MSYS2 system, builds MUST use the PowerShell wrapper for MSVC + Vulkan SDK.

```bash
# Build
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\\OctoFlow\\run_test.ps1" build

# Test all
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\\OctoFlow\\run_test.ps1" test --workspace

# Run OctoFlow
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\\OctoFlow\\run_test.ps1" run --bin octoflow -- run examples/hello.flow
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\\OctoFlow\\run_test.ps1" run --bin octoflow -- repl
```

---

## Value Types and Arrays

```rust
pub enum Value {
    Float(f32),    // Default numeric type
    Int(i64),      // Integer (42 vs 42.0)
    Str(String),   // Strings with interpolation: "val: {x}"
    Map(HashMap<String, Value>),  // Maps
    None,          // none keyword, is_none() builtin
}
```

**Arrays are stored separately** in `HashMap<String, Vec<Value>>` — NOT in the Value enum.
This means:
- **`push(array, scalar)` works** — appends a scalar element
- **`push(array, array_variable)` FAILS** — second arg resolved as scalar via eval_scalar()
- **Workaround:** Use flat arrays with index math: `pop[p * GENE_COUNT + g]`
- **Fix in progress (F4-04):** `extend(dest, src)`, `array_copy()`, `array_extract()` builtins

---

## Loom Engine Architecture

- **Main Loom** = GPU-only compute. Receives dispatches. Never initiates I/O.
- **Support Loom** = CPU↔GPU I/O bridge. Owns boot, state, double buffer, presentation.
- N Main Looms → 1 Support Loom (many-to-one). No cross-Main reads.
- Homeostasis: auto-paces dispatches via timing (EMA baseline, ±50/-10μs).

**Key builtins:**
```
loom_boot(instances, reg_size, globals_size) → vm_id
loom_write(vm_id, slot, data)
loom_dispatch(vm_id, spv_path, push_constants, workgroups)
loom_build(vm_id) → prog_id
loom_run(prog_id)
loom_read(vm_id, instance, reg_idx, count) → array
loom_shutdown(vm_id)
loom_park(vm_id) / loom_unpark(vm_id)
loom_auto_spawn(instances, reg_size, globals_size) → vm_id  (reuses parked VMs)
loom_mailbox(vm_id, slot_size, slot_count)
loom_mail_send(from, to, data) / loom_mail_recv(vm_id) → array
```

**IR Builder** (`stdlib/compiler/ir.flow`):
```
ir_new() → ir_block("entry") → ir_load_gid/lid → ir_* ops → ir_emit_spirv(path)
```
80+ operations: arithmetic, comparison, bitwise (iand/ior/ixor/ishl/ishr), shared memory, barriers, select, min/max. Used by 20+ production emitters.

**`emit_word(val)`** treats float as integer value (`val & 255`), NOT IEEE-754 bit pattern. This is correct and intentional — same function used for all SPIR-V constants.

---

## Algorithm Space Exploration Engine (ASE)

**Core insight:** Explore the space of ALGORITHMS, not just run them.

**Pattern:** Genome (flat float array) → IR builder emits SPIR-V → Boot VMs → Dispatch → Read results → Fitness evaluation → Evolution (selection + crossover + mutation)

**Libraries:**
- `stdlib/loom/ase/genome.flow` — `ase_genome_create(n)`, `ase_genome_randomize(g, schema)`
- `stdlib/loom/ase/evolve.flow` — `ase_next_generation(pop, scores, schema, keep, mut_rate)`
- `stdlib/loom/ase/fitness.flow` — `ase_fitness_correctness()`, `ase_fitness_speed()`, `ase_fitness_combined()`

**IMPORTANT:** ASE libraries use nested arrays internally (population = array of genomes). Due to the `push(array, array)` limitation, demos must use flat-array double-buffer pattern instead of calling `ase_next_generation()` directly. See `examples/ase_bitwise_discovery.flow` for the workaround.

**68 application domains** researched across 8 fields (Geometry, Physics, Cross-domain, Chemistry, Biology, Economics, Biochemistry). 12 genuinely impossible without the Loom Engine. See `docs/algorithm-space-engine.md`.

---

## OctoPress Compression

**Builtins (all shipped):**
```
octopress_init(block_size)           → 0.0 (must be power-of-2)
octopress_analyze(data)              → map {mean, variance, self_similarity, delta_ratio, recommended_method}
octopress_encode(data, method)       → compressed array (0=raw, 1=delta, 2=fractal WIP)
octopress_decode(compressed)         → original array
octopress_save(compressed, path)     → 0.0 (writes .ocp file: OCP1 magic + u32 LE header + f32 body)
octopress_load(path)                 → compressed array
```

**.flow wrappers:** `stdlib/loom/octopress.flow` — `octopress_compress()`, `octopress_decompress()`, `octopress_auto()`, `octopress_ratio()`

---

## Known Gotchas

| Gotcha | Workaround |
|--------|-----------|
| `push(array, array_var)` fails | Flat arrays with index math. F4-04 adds `extend`/`array_copy`/`array_extract`. |
| `map()` can't be inline arg | `let m = map()` then pass `m` |
| `map_keys()` on empty map → `""` | Guard: `if len(keys) == 0.0` |
| Mutable scalars don't cross function boundaries | Use array elements instead |
| `print()` uses interpolation | `print("val: {x}")` not `print("val: " + str(x))` |
| Shared-memory bitonic sort can't cross workgroups | Use multi-VM swarm + merge |
| `RETURNED_ARRAY` side-channel | Functions returning arrays use thread-local hack |
| No `and`/`or` keywords | Use `&&`/`||` for boolean logic. `!` for negation. Short-circuit eval. |
| `let x = x + 1` in loops can reset | Use `let mut x = 0.0` before loop, bare `x = x + 1` inside (no `let`). |
| Cross-module `rt_*` calls fail | User-defined `rt_*` functions don't resolve across `use`. Use builtins directly. |

---

## Standing Principles

1. **LLM-native first.** Every feature: "Can an LLM generate correct code for this on the first try?"
2. **Zero dependencies.** No new external crates. One binary.
3. **Self-hosting direction.** New functionality in .flow, not Rust, unless it's an OS boundary.
4. **Ship the dish.** Users get working solutions. Recipes (stdlib) for tinkering.
5. **GPU is invisible.** Users never think about kernels, VRAM, or dispatch chains.

---

## Cross-Team Coordination

| Workspace | Role | Location |
|-----------|------|----------|
| **Development** | Build features, write code | `C:\OctoFlow` |
| **Enhancement** | Specs, patterns, benchmarks | `G:\OctoFlow-Enhancement` |
| **Auditor** | Track progress, validate, orchestrate | `G:\OctoFlow-Lang Auditor` |

**Live Tracker (single source of truth):** `G:\OctoFlow-Lang Auditor\LIVE_TRACKER2.md`

**Dev briefs (current):**
- Dev 1: `G:\OctoFlow-Lang Auditor\prompts\DEV1_LOOM_THREADING.md` (DONE)
- Dev 2: `G:\OctoFlow-Lang Auditor\prompts\DEV2_OCTOSEARCH_FIX.md`

---

## Hardware

- GPU: NVIDIA GeForce GTX 1660 SUPER (1408 cores, 6GB GDDR6, TU116)
- CPU: AMD Ryzen 5 4600G (6c/12t)
- Vulkan 1.1, shaderInt64 enabled, f16 supported

## The North Star Test

> Every decision: "Does this make it easier for a 1.5B model to generate
> correct OctoFlow from a one-sentence description?"
