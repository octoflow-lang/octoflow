# Loom Engine: stdlib Reorganization Plan

**Date:** 2026-02-25
**Status:** Approved design — phased execution

---

## Principle: Serve the Dish, Not the Recipe

Current stdlib is organized by implementation concern (compiler, gpu, tests).
The new layout organizes by **what users do** — by domain.

```
Before: "Where is the file that emits the sieve mark kernel?"
         stdlib/gpu/emit_sieve_mark_v3_large.flow        ← hunting

After:  "I need sieve stuff"
         stdlib/loom/emit/sieve/mark_v3_large.flow        ← obvious
```

---

## Directory Mapping

### stdlib/loom/ — The Loom Engine

Everything GPU compute moves under `stdlib/loom/`, sorted by domain:

```
stdlib/loom/
│
├── ir.flow                           ← symlink/import from stdlib/compiler/ir.flow (compiler owns it)
│
├── patterns/                         ← NEW: one-call GPU compute (Tier 1)
│   ├── reduce.flow                    gpu_sum, gpu_min, gpu_max
│   ├── map.flow                       gpu_map (element-wise)
│   ├── scan.flow                      gpu_scan (prefix sum)
│   ├── sort.flow                      gpu_sort (radix sort)
│   ├── sieve.flow                     gpu_prime_count
│   └── matmul.flow                    gpu_matmul
│
├── math/                             ← from stdlib/gpu/ (scattered)
│   ├── linalg.flow                    ← stdlib/gpu/linalg.flow
│   ├── stats.flow                     ← stdlib/gpu/stats.flow
│   ├── signal.flow                    ← stdlib/gpu/signal.flow
│   └── advanced.flow                  ← stdlib/gpu/math_advanced.flow
│
├── nn/                               ← from stdlib/gpu/ (transformer files)
│   ├── attention.flow                 ← stdlib/gpu/attention.flow
│   ├── ffn.flow                       ← stdlib/gpu/ffn.flow
│   ├── rmsnorm.flow                   ← stdlib/gpu/rmsnorm.flow
│   ├── rope.flow                      ← stdlib/gpu/rope.flow
│   ├── silu.flow                      ← stdlib/gpu/silu.flow
│   ├── softmax.flow                   ← stdlib/gpu/softmax.flow
│   ├── matmul_tiled.flow              ← stdlib/gpu/matmul_tiled.flow
│   └── dequant.flow                   ← stdlib/gpu/dequant.flow
│
├── data/                             ← from stdlib/gpu/ (data processing)
│   ├── array_ops.flow                 ← stdlib/gpu/array_ops.flow
│   ├── aggregate.flow                 ← stdlib/gpu/aggregate.flow
│   ├── composite.flow                 ← stdlib/gpu/composite.flow
│   └── dlb_scan.flow                  ← stdlib/gpu/gpu_dlb_scan.flow
│
├── ops/                              ← from stdlib/gpu/ (runtime infrastructure)
│   ├── runtime.flow                   ← stdlib/gpu/runtime.flow
│   ├── ops.flow                       ← stdlib/gpu/ops.flow
│   ├── patterns.flow                  ← stdlib/gpu/patterns.flow
│   ├── vk.flow                        ← stdlib/gpu/vk.flow
│   ├── debug.flow                     ← stdlib/gpu/debug.flow
│   └── homeostasis.flow               ← stdlib/gpu/homeostasis.flow
│
├── kernels/                          ← from stdlib/gpu/kernels/ (reorganized)
│   ├── math/                          Basic math kernels
│   │   ├── abs.spv
│   │   ├── add.spv, add_pc.spv
│   │   ├── ceil.spv, cos.spv, div.spv, exp.spv, floor.spv
│   │   ├── log.spv, mul.spv, negate.spv, round.spv
│   │   ├── sin.spv, sqrt.spv, sub.spv
│   │   ├── clamp_pc.spv, max_pc.spv, min_pc.spv, mod_pc.spv
│   │   ├── multiply_pc.spv, divide_pc.spv, subtract_pc.spv
│   │   ├── pow_pc.spv, scale_shift_pc.spv, normalize_pc.spv
│   │   └── double.spv, where.spv
│   │
│   ├── reduce/                        Reduction kernels
│   │   ├── reduce_sum.spv
│   │   ├── reduce_min.spv
│   │   ├── reduce_max.spv
│   │   └── reduce_mul.spv
│   │
│   ├── sieve/                         Sieve kernels (all versions)
│   │   ├── init.spv, init_v2.spv, init_v6.spv
│   │   ├── init_offsets.spv, init_offsets_v7.spv
│   │   ├── mark.spv, mark_v2.spv
│   │   ├── mark_v3_small.spv, mark_v3_large.spv
│   │   ├── mark_v4.spv
│   │   ├── mark_v6_small.spv, mark_v6_large.spv
│   │   ├── mark_v7_large.spv
│   │   ├── count.spv, count_v2.spv, count_v3.spv
│   │   ├── accum.spv, accum_v2.spv
│   │   └── metrics.spv
│   │
│   ├── nn/                            Neural net kernels
│   │   ├── matvec.spv, matmul_tiled.spv
│   │   ├── rmsnorm.spv, rope.spv
│   │   ├── silu.spv, softmax.spv
│   │   ├── dlb_scan.spv
│   │   └── attention_score.spv
│   │
│   ├── ops/                           VM operation kernels
│   │   ├── vm_add.spv, vm_affine.spv, vm_copy.spv, vm_scale.spv
│   │   ├── vm_relu.spv, vm_silu_mul.spv
│   │   ├── vm_rmsnorm.spv, vm_rmsnorm_apply.spv, vm_rmsnorm_apply_b1.spv
│   │   ├── vm_sum_sq.spv, vm_sum_sq_b1.spv
│   │   ├── vm_matvec.spv
│   │   ├── vm_delta_encode.spv, vm_delta_decode.spv
│   │   ├── vm_dequant_q4k.spv
│   │   ├── vm_where_gt.spv, vm_where_mul.spv
│   │   ├── vm_reduce_sum.spv
│   │   ├── vm_dict_lookup.spv
│   │   ├── vm_regulator.spv, vm_maxnorm.spv, vm_scheduler.spv
│   │   └── gol_step.spv
│   │
│   └── test/                          Test kernels
│       ├── test_bitwise.spv
│       ├── test_uint64.spv
│       └── test_rt.spv
│
├── emit/                             ← from stdlib/gpu/emit_*.flow (reorganized)
│   ├── sieve/                         Sieve kernel emitters
│   │   ├── init.flow                  ← emit_sieve_init.flow
│   │   ├── init_v2.flow               ← emit_sieve_init_v2.flow
│   │   ├── init_v6.flow               ← emit_sieve_init_v6.flow
│   │   ├── init_offsets.flow          ← emit_sieve_init_offsets.flow
│   │   ├── init_offsets_v7.flow       ← emit_sieve_init_offsets_v7.flow
│   │   ├── mark.flow                  ← emit_sieve_mark.flow
│   │   ├── mark_v2.flow               ← emit_sieve_mark_v2.flow
│   │   ├── mark_v3_large.flow         ← emit_sieve_mark_v3_large.flow
│   │   ├── mark_v3_small.flow         ← emit_sieve_mark_v3_small.flow
│   │   ├── mark_v4.flow               ← emit_sieve_mark_v4.flow
│   │   ├── mark_v6_large.flow         ← emit_sieve_mark_v6_large.flow
│   │   ├── mark_v6_small.flow         ← emit_sieve_mark_v6_small.flow
│   │   ├── mark_v7_large.flow         ← emit_sieve_mark_v7_large.flow
│   │   ├── count.flow                 ← emit_sieve_count.flow
│   │   ├── count_v2.flow              ← emit_sieve_count_v2.flow
│   │   ├── count_v3.flow              ← emit_sieve_count_v3.flow
│   │   ├── accum.flow                 ← emit_sieve_accum.flow
│   │   ├── accum_v2.flow              ← emit_sieve_accum_v2.flow
│   │   ├── metrics.flow               ← emit_sieve_metrics.flow
│   │   └── planner.flow               ← emit_sieve_planner.flow
│   │
│   ├── nn/                            Neural net kernel emitters
│   │   ├── matvec.flow                ← emit_matvec.flow / emit_vm_matvec.flow
│   │   ├── matmul_tiled.flow          ← emit_matmul_tiled.flow
│   │   ├── rmsnorm.flow               ← emit_rmsnorm.flow / emit_vm_rmsnorm.flow
│   │   ├── rmsnorm_apply.flow         ← emit_vm_rmsnorm_apply.flow
│   │   ├── rmsnorm_apply_b1.flow      ← emit_vm_rmsnorm_apply_b1.flow
│   │   ├── rope.flow                  ← emit_rope.flow
│   │   ├── silu.flow                  ← emit_silu.flow / emit_vm_silu_mul.flow
│   │   ├── softmax.flow               ← emit_softmax.flow
│   │   ├── sum_sq.flow                ← emit_vm_sum_sq.flow
│   │   ├── sum_sq_b1.flow             ← emit_vm_sum_sq_b1.flow
│   │   └── dequant_q4k.flow           ← emit_vm_dequant_q4k.flow
│   │
│   ├── ops/                           VM operation kernel emitters
│   │   ├── add.flow                   ← emit_vm_add.flow
│   │   ├── affine.flow                ← emit_vm_affine.flow
│   │   ├── copy.flow                  ← emit_vm_copy.flow
│   │   ├── scale.flow                 ← emit_vm_scale.flow
│   │   ├── relu.flow                  ← emit_vm_relu.flow
│   │   ├── delta_encode.flow          ← emit_vm_delta_encode.flow
│   │   ├── delta_decode.flow          ← emit_vm_delta_decode.flow
│   │   ├── where_gt.flow              ← emit_vm_where_gt.flow
│   │   ├── where_mul.flow             ← emit_vm_where_mul.flow
│   │   ├── reduce_sum.flow            ← emit_vm_reduce_sum.flow
│   │   ├── dict_lookup.flow           ← emit_vm_dict_lookup.flow
│   │   ├── regulator.flow             ← emit_vm_regulator.flow
│   │   ├── maxnorm.flow               ← emit_vm_maxnorm.flow
│   │   ├── scheduler.flow             ← emit_vm_scheduler.flow
│   │   └── pc_kernels.flow            ← emit_pc_kernels.flow
│   │
│   └── misc/                          Standalone emitters
│       ├── gol_step.flow              ← emit_gol_step.flow
│       ├── raytrace.flow              ← raytrace_emit.flow
│       ├── b64.flow                   ← b64_emit.flow
│       └── dlb_scan.flow              ← dlb_scan_emit.flow
│
└── tests/                            ← from stdlib/gpu/tests/ (reorganized)
    ├── core/                          Core engine tests
    │   ├── test_dispatch.flow         ← test_vm_dispatch.flow
    │   ├── test_poll.flow             ← test_vm_poll.flow
    │   ├── test_registers.flow        ← test_vm_registers.flow
    │   ├── test_chain.flow            ← test_vm_chain.flow
    │   ├── test_pipeline.flow         ← test_vm_pipeline.flow
    │   ├── test_multi.flow            ← test_vm_multi.flow
    │   ├── test_stream.flow           ← test_vm_stream.flow
    │   └── test_indirect.flow         ← test_vm_indirect.flow
    │
    ├── ir/                            IR builder tests
    │   ├── test_bitwise.flow          ← test_bitwise_ir.flow
    │   ├── test_uint64.flow           ← test_uint64_ir.flow
    │   └── test_runtime_spirv.flow    ← test_runtime_spirv.flow
    │
    ├── nn/                            Neural net kernel tests
    │   ├── test_kernels.flow          ← test_transformer_kernels.flow
    │   ├── test_layer.flow            ← test_transformer_layer.flow
    │   ├── test_compose.flow          ← test_transformer_compose.flow
    │   ├── test_matmul.flow           ← test_matmul_tiled.flow
    │   ├── test_matvec.flow           ← test_matvec.flow
    │   ├── test_dequant.flow          ← test_dequant_q4k.flow
    │   ├── test_ffn_chain.flow        ← test_vm_ffn_chain.flow
    │   ├── test_layer_vm.flow         ← test_vm_layer.flow
    │   └── test_gguf.flow             ← test_vm_gguf.flow
    │
    ├── patterns/                      Pattern tests
    │   ├── test_gpu_ops.flow          ← test_gpu_ops.flow
    │   ├── test_gpu_advanced.flow     ← test_gpu_advanced.flow
    │   ├── test_gpu_recipes.flow      ← test_gpu_recipes.flow
    │   ├── test_gpu_edge.flow         ← test_gpu_edge_cases.flow
    │   ├── test_dlb_scan.flow         ← test_gpu_dlb_scan.flow
    │   └── test_pc_kernels.flow       ← test_pc_kernels.flow
    │
    ├── stress/                        Stress tests and benchmarks
    │   ├── test_stress.flow           ← test_gpu_stress.flow
    │   ├── test_compress.flow         ← test_vm_compress.flow
    │   ├── bench_performance.flow     ← bench_gpu_performance.flow
    │   ├── bench_fs.flow              ← bench_gpu_fs.flow
    │   └── bench_resident.flow        ← bench_resident.flow
    │
    └── misc/
        ├── test_homeostasis.flow      ← test_homeostasis.flow
        ├── test_debug.flow            ← test_debug.flow
        ├── test_deferred.flow         ← test_deferred_chain.flow
        ├── test_vk_dispatch.flow      ← test_vk_dispatch.flow
        └── test_gpu_fs.flow           ← test_gpu_fs.flow
```

---

## Rust Code Changes

### include_bytes! Path Updates

**dispatch.rs** (54 references):
```rust
// Before
include_bytes!("../../../stdlib/gpu/kernels/add.spv")
// After
include_bytes!("../../../stdlib/loom/kernels/math/add.spv")
```

**compiler.rs** (6 references):
```rust
// Before
include_bytes!("../../../stdlib/gpu/kernels/matvec.spv")
// After
include_bytes!("../../../stdlib/loom/kernels/nn/matvec.spv")
```

**gpu_run.rs** (2 references):
```rust
// Before
include_bytes!("../../../stdlib/gpu/kernels/double.spv")
// After
include_bytes!("../../../stdlib/loom/kernels/math/double.spv")
```

**loader.rs** (path construction):
```rust
// Before
"stdlib/compiler/eval.flow"
// After — compiler stays in stdlib/compiler/ (not part of Loom)
// No change needed
```

### Function Aliases (evaluator)

In the evaluator (wherever `vm_boot` etc. are matched):
```rust
// Add loom_* as primary, keep vm_* as deprecated alias
"loom_boot" | "vm_boot" => { /* ... */ }
"loom_dispatch" | "vm_dispatch" => { /* ... */ }
"loom_dispatch_jit" | "vm_dispatch_mem" => { /* ... */ }
"loom_build" | "vm_build" => { /* ... */ }
"loom_run" | "vm_execute" => { /* ... */ }
"loom_launch" | "vm_execute_async" => { /* ... */ }
"loom_poll" | "vm_poll" => { /* ... */ }
"loom_read" | "vm_read_register" => { /* ... */ }
"loom_write" | "vm_write_globals" => { /* ... */ }
```

---

## Execution Phases

### Phase 1: Function Aliases (safe, backward compatible)

Add `loom_*` as aliases in the Rust evaluator. All existing code keeps working.
New code can use either naming convention.

**Risk: None.** Old names still work.

### Phase 2: Directory Structure

```bash
# Create new structure
mkdir -p stdlib/loom/{patterns,math,nn,data,ops,emit/{sieve,nn,ops,misc},kernels/{math,reduce,sieve,nn,ops,test},tests/{core,ir,nn,patterns,stress,misc}}

# Move files (git mv preserves history)
git mv stdlib/gpu/linalg.flow stdlib/loom/math/linalg.flow
git mv stdlib/gpu/attention.flow stdlib/loom/nn/attention.flow
# ... (full list in migration script)
# NOTE: stdlib/compiler/ is NOT touched — ir.flow stays in compiler
```

**Risk: Medium.** Breaks .flow path references in `stdlib/gpu/`. Must update simultaneously.

### Phase 3: Path Updates in .flow Files

Batch find-replace across ~196 files:
```
"stdlib/gpu/kernels/" → "stdlib/loom/kernels/"
```

Plus domain-specific renames:
```
"stdlib/gpu/kernels/sieve_" → "stdlib/loom/kernels/sieve/"
"stdlib/gpu/kernels/reduce_" → "stdlib/loom/kernels/reduce/"
"stdlib/gpu/kernels/vm_" → "stdlib/loom/kernels/ops/vm_"
```

### Phase 4: Rust include_bytes! Updates

Update 62 include_bytes! paths in 3 Rust files.

### Phase 5: Documentation Updates

Update all docs that reference `stdlib/gpu/` or `vm_*` functions.

### Phase 6: Pattern Library (new code)

Implement one-call wrappers:
- `gpu_prime_count(N)` → wraps sieve v7
- `gpu_sum(data)` → wraps reduce
- `gpu_map(data, op)` → wraps map patterns

---

## What Does NOT Move

| Directory | Reason |
|---|---|
| `stdlib/compiler/` | **Self-hosting compiler — DO NOT TOUCH** |
| `stdlib/compiler/ir.flow` | Stays in compiler. Loom imports it, doesn't own it |
| `stdlib/ai/` | AI/ML modules, uses Loom but isn't Loom |
| `stdlib/llm/` | LLM inference, uses Loom but organized by LLM domain |
| `stdlib/collections/` | Pure CPU data structures |
| `stdlib/data/`, `stdlib/web/`, etc. | CPU-side libraries |

**Branding is surface-level.** We rename `stdlib/gpu/` → `stdlib/loom/` and add
`loom_*` function aliases. We do NOT touch compiler internals, Rust evaluator
logic, or SPIR-V emission code. The compiler stays exactly as-is.

The line: **if it's in `stdlib/gpu/`, it moves to `stdlib/loom/`.** Everything
else stays put. The Rust code changes are limited to:
1. Adding `loom_*` match arms alongside existing `vm_*` (aliases, not replacements)
2. Updating `include_bytes!` paths (mechanical find-replace)

Exception: `stdlib/llm/kernels/` has 11 .spv files that duplicate `stdlib/loom/kernels/nn/`.
These should be consolidated — one copy in `stdlib/loom/kernels/nn/`, with
`stdlib/llm/` referencing the Loom versions.

---

## File Count Summary

| Section | Files | Notes |
|---|---|---|
| **loom/ir.flow** | 1 | Re-export of stdlib/compiler/ir.flow (compiler owns it) |
| **loom/patterns/** | 6 | New one-call wrappers |
| **loom/math/** | 4 | Numerical computation |
| **loom/nn/** | 8 | Neural net primitives |
| **loom/data/** | 4 | Data-parallel operations |
| **loom/ops/** | 6 | Runtime infrastructure |
| **loom/kernels/** | ~90 | Pre-compiled .spv (sorted into 5 subdirs) |
| **loom/emit/** | ~40 | Kernel emitters (sorted into 4 subdirs) |
| **loom/tests/** | ~42 | Tests (sorted into 6 subdirs) |
| **Total** | ~201 | Down from ~170 in gpu/ + new patterns |

---

## Success Criteria

After reorganization:

1. **"I need sieve stuff"** → look in `stdlib/loom/emit/sieve/` and `stdlib/loom/kernels/sieve/`
2. **"I need neural net kernels"** → look in `stdlib/loom/nn/` and `stdlib/loom/kernels/nn/`
3. **"I want to sum an array on GPU"** → `use stdlib.loom.patterns.reduce` → `gpu_sum(data)`
4. **All existing tests pass** with new paths
5. **`spirv-val`** passes on all kernel .spv files in new locations
6. **No vm_ references** in new code (loom_ only; vm_ kept as deprecated aliases)
