# Annex W2: Loom Engine Formalization — Internal Architecture Spec

**Date:** March 3, 2026
**Status:** Implemented (v1.5.0)
**Classification:** INTERNAL — Trade secret. Never publish.
**Predecessor:** Annex W (Loom Engine rename + API design)
**Author context:** Documents the formalized two-tier architecture shipped in `stdlib/loom/`

---

## The Octopoid Mapping

Every Loom Engine is an octopus. This is not a metaphor — it is the architecture.

```
                        ┌─────────────────────────────────┐
                        │         SWARM (Colony)           │
                        │   Multiple engines, independent  │
                        │   Broadcast / Scatter / Gather   │
                        │   Cross-engine channels          │
                        └──────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼─────────┐ ┌───────▼───────┐   ┌────────▼────────┐
    │  ENGINE (Octopus)  │ │   ENGINE #2   │   │   ENGINE #N     │
    │                    │ └───────────────┘   └─────────────────┘
    │  ┌──────────────┐  │
    │  │ SUPPORT LOOM │  │  ← Brain: lifecycle, I/O, topology,
    │  │   (Brain)     │  │     adaptation, resource budgets
    │  └──────┬───────┘  │     Implemented as .flow orchestration
    │         │ polls     │     Never runs GPU compute directly
    │    ┌────┴────┐      │
    │    │         │      │
    │  ┌─▼──┐  ┌──▼─┐    │
    │  │ARM │  │ARM │... │  ← Main Looms: raw GPU compute
    │  │ 0  │  │ 1  │    │     Self-regulating (micro-homeostasis)
    │  └────┘  └────┘    │     One vkQueueSubmit, one fence
    │                    │     Never message the brain
    │  Topology:         │
    │  • Parallel        │  ← Same kernel, different data chunks
    │  • Sequential      │  ← Pipeline, output chains forward
    │  • Hierarchical    │  ← Tree reduction
    └────────────────────┘
```

**Mapping table:**

| Biological | Loom | File | Role |
|-----------|------|------|------|
| Brain | Support Loom | `engine.flow` | Lifecycle, I/O, topology, adaptation |
| Arm | Main Loom | `main_loom.flow` | GPU compute, self-regulation |
| Nervous system | Micro-homeostasis | `main_loom.flow` | Regulator kernel, maxnorm, pacing |
| Organism | Engine | `engine.flow` | One brain + N arms |
| Colony | Swarm | `swarm.flow` | Multiple independent engines |
| Spatial awareness | Topology | `topology.flow` | Parallel / Sequential / Hierarchical |

---

## Design Principles

Ten principles govern the formalized architecture. All are implemented.

1. **Polling, not messaging.** The brain never receives messages from arms. It polls their metrics SSBO via `loom_read_metrics`. Arms are autonomous — the brain watches.

2. **Bitwise heterogeneity.** Different arms can run at different precision levels simultaneously. Attention arms at Q6_K, feedforward arms at Q4_K, embedding arms at F32. The engine tracks `precisions[i]` per arm.

3. **OctoPress per-Loom.** Each arm has independent hot/warm/cold weight tiers. Hot = VRAM SSBO (full precision). Warm = VRAM Heap (OctoPress-compressed, fractal method=2.0). Cold = disk (.octopress files). The brain manages promotion/eviction across all arms.

4. **Micro-homeostasis.** Each arm self-corrects via GPU-autonomous kernels — `vm_regulator.spv` adjusts scale based on `vm_maxnorm.spv` output. Zero CPU roundtrip during compute. The brain only adjusts thresholds.

5. **JIT-adaptive.** New precision tiers (Q5_K, 1-bit) dispatch via `loom_dispatch_jit` — SPIR-V bytes emitted at runtime from `ir.flow`. No pre-compiled `.spv` files needed for new quantization formats.

6. **Topology is runtime-selectable.** The brain sets topology per engine: parallel for data-parallel work, sequential for pipeline stages, hierarchical for tree reduction. `topo_auto_select` picks based on workload characteristics.

7. **Mailbox for sibling communication.** Arms can't talk to the brain, but they CAN mailbox each other via `loom_mail_send`/`loom_mail_recv` (ring-buffer GPU IPC). Cross-engine communication uses shared mailbox bridges.

8. **Pool-backed lifecycle.** `engine_spawn` uses `loom_auto_spawn` (pooled VM allocation), not `loom_boot` (fresh). `engine_release_last` returns VMs to pool via `loom_auto_release`. This keeps hot VMs ready.

9. **Adaptation loop.** `engine_adapt` is the brain's decision cycle — reads stability from all arms, identifies idle/unstable arms, merges or releases as needed. Thresholds: idle = maxnorm < 0.001, unstable = stability_flag < 0.5.

10. **Swarm independence.** Each engine in a swarm has its own Support Loom. No shared state except explicit channels. A swarm is coordination, not hierarchy.

---

## Main Loom Deep Dive

**File:** `stdlib/loom/main_loom.flow`
**Functions:** 30
**Role:** The "arm" — a GPU VM with self-regulation

### Core Dispatch

```
fn main_dispatch(vm_id, kernel, pc, wg)       // queue kernel dispatch
fn main_build(vm_id)                           // compile dispatch chain → program
fn main_run_prog(prog)                         // execute compiled program
fn main_free_prog(prog)                        // release program resources
fn main_exec(vm_id)                            // build + run + free (convenience)
```

The dispatch chain model: queue multiple dispatches, build once, submit once. One `vkQueueSubmit`, one fence. This is why Loom is fast — no per-kernel submission overhead.

### Data I/O

```
fn main_write(vm_id, offset, data)             // CPU → GPU Globals
fn main_read(vm_id, offset, count)             // GPU Globals → CPU
fn main_send(vm_id, mailbox, offset, count)    // GPU → mailbox (sibling IPC)
fn main_recv(vm_id, mailbox, offset)           // mailbox → GPU
fn main_poll_mail(mailbox)                     // check mailbox depth (non-blocking)
```

### OctoPress Three-Tier Weight Management

Three memory tiers per arm:

| Tier | Location | State Array | Precision | Access Speed |
|------|----------|-------------|-----------|-------------|
| Hot | GPU SSBO (Globals) | `ocp_hot[i]` | Full (F32) | Immediate — kernel reads directly |
| Warm | GPU Heap | `ocp_warm[i]` | Compressed (fractal, method=2.0) | Promote → dequant → Globals |
| Cold | Disk (.octopress) | `ocp_cold[i]` | Compressed file | Stream → decompress → Globals |

```
fn main_load_hot(vm_id, offset, data)          // load directly into Globals (hot tier)
fn main_load_warm(vm_id, data)                 // compress + store in Heap (warm tier)
fn main_promote_warm(vm_id, dequant_kernel, pc, wg)  // decompress warm → hot (GPU-side)
fn main_evict_to_cold(data, path)              // compress + save to disk (cold tier)
fn main_stream_from_cold(vm_id, path, offset)  // stream cold → hot (block by block)
fn main_octopress_info(path)                   // inspect .octopress file metadata
```

**Compression:** All warm/cold storage uses OctoPress fractal encoding (method=2.0, block size 256 floats). This keeps evicted weights in VRAM at reduced footprint — faster than disk reload.

**The 24GB-on-6GB claim:** Only the active layer's weights need hot tier (full VRAM). Previous layers compress to warm tier. Distant layers evict to cold. A 24GB model needs ~1 layer hot + N layers warm + rest cold.

### Bitwise Precision — Sieve-Pattern Kernel Selection

```
fn main_select_dequant(precision)              // precision → dequant kernel path
fn main_needs_dequant(precision)               // does this precision need dequant step?
fn main_dequant_workgroups(precision, n_weights)  // workgroup count for dequant
fn main_dispatch_with_dequant(vm_id, precision, compute_kernel, compute_pc, compute_wg, dequant_pc, dequant_wg)
```

**Precision → Kernel mapping:**

| Precision | Dequant Kernel | Notes |
|-----------|---------------|-------|
| `"Q4_K"` | `stdlib/loom/kernels/ops/vm_dequant_q4k.spv` | Pre-compiled, wg = ceil(n/256) |
| `"Q6_K"` | `stdlib/llm/kernels/dequant_q6k.spv` | Pre-compiled, wg = ceil(n/256) |
| `"Q5_K"` | JIT via `loom_dispatch_jit` | CPU dequant exists (`gguf_dequant_q5k_block`) |
| `"1bit"` | JIT via `loom_dispatch_jit` | Binary weights |
| `"F32"` | None (`"none"`) | No dequant needed |

This generalizes the prime sieve v7's dual-kernel architecture: a "small" kernel selects/decompresses, then a "large" kernel computes. Every precision-adaptive dispatch follows this sieve pattern.

### Micro-Homeostasis (GPU-Autonomous)

```
fn main_attach_regulator(vm_id, target_norm, lo_bound, hi_bound)
fn main_update_regulator(vm_id, target_norm, lo_bound, hi_bound)
fn main_dispatch_regulated(vm_id, kernel, pc, wg, metrics_offset, n_elements)
fn main_dispatch_full(vm_id, precision, compute_kernel, compute_pc, compute_wg, dequant_pc, dequant_wg, metrics_offset, n_elements)
fn main_read_stability(vm_id, metrics_offset)  // returns [maxnorm, scale, stability_flag, pace_us]
fn main_set_pace(vm_id, pace_us)
fn main_is_stable(vm_id, metrics_offset)       // returns 1.0 (stable) or 0.0 (correcting)
```

**Kernel paths:**
- `stdlib/loom/kernels/ops/vm_maxnorm.spv` — computes max absolute value
- `stdlib/loom/kernels/ops/vm_regulator.spv` — adjusts scale based on maxnorm vs bounds

**Default thresholds** (set by `engine_create`): `target_norm=10.0, lo_bound=5.0, hi_bound=20.0`

**Stability readout:** `main_read_stability` reads the metrics SSBO and returns `[maxnorm, scale, stability_flag, pace_us]`. A `stability_flag` of `1.0` means the arm is within bounds; `0.0` means it's actively correcting.

**Regulated dispatch chain:** `main_dispatch_regulated` queues: compute kernel → maxnorm → regulator, all in one dispatch chain. Zero CPU roundtrip — the GPU self-corrects.

**Full dispatch chain:** `main_dispatch_full` queues: dequant → compute → maxnorm → regulator. This is the complete precision-aware, self-regulating dispatch — the universal pattern for inference.

### JIT Kernel Emission

```
fn main_dispatch_jit(vm_id, spirv_bytes, pc, wg)
```

For precision tiers without pre-compiled `.spv` files (Q5_K, 1-bit), the runtime emits SPIR-V bytes via `ir.flow` and dispatches them through `loom_dispatch_jit`. This means new quantization formats can be added purely in `.flow` — no Rust changes needed.

### Pipeline Reuse

```
fn main_pipe_create(vm_id)    // create reusable dispatch chain
fn main_pipe_exec(pipe_id)    // execute pipeline (build + run)
```

**Note:** `loom_pipe_add(pipe_id, kernel, pc, wg)` must be called directly — the preflight checker does not allow 4-arg calls through a wrapper function. This is a known preflight limitation.

---

## Engine Deep Dive

**File:** `stdlib/loom/engine.flow`
**Functions:** 28
**Role:** The "brain" — Support Loom managing all Main Looms
**Imports:** `use "main_loom"`

### Lifecycle

```
fn engine_create(n_main, globals_size, default_precision, main_ids, precisions)
fn engine_create_mixed(n_main, globals_size, precision_list, main_ids, precisions)
fn engine_shutdown(engine, main_ids)
```

**`engine_create`:** Boots `n_main` VMs via `loom_boot`, attaches regulator with default thresholds (10.0/5.0/20.0), sets up thread pool (`loom_threads(4.0)`). Returns engine map with fields: `id`, `globals_size`, `topology`, `n_main`, `mailbox_id`.

**`engine_create_mixed`:** Same but accepts a `precision_list` array — each arm gets a different precision tier. This enables bitwise heterogeneity from creation.

**Global state:** `engine_next_id` auto-increments for unique engine IDs.

### Loom Management

```
fn engine_spawn(engine, main_ids, precisions, precision)
fn engine_release_last(engine, main_ids, precisions)
fn engine_reprecision(precisions, index, new_precision)
```

**`engine_spawn`:** Uses `loom_auto_spawn` (pooled, fast) not `loom_boot` (fresh, slow). Attaches regulator immediately. Increments `engine.n_main`.

**`engine_release_last`:** Returns the last arm to pool via `loom_auto_release`. Will not release below 1 arm. Decrements `engine.n_main`.

**`engine_reprecision`:** Changes a single arm's precision tier. Just updates the tracking array — the next dispatch will use the new dequant kernel automatically.

### Topology

```
fn engine_set_topology(engine, mode)       // "parallel" | "sequential" | "hierarchical"
fn engine_get_topology(engine)
fn engine_dispatch_all(engine, main_ids, kernel, pc, wg)
fn engine_dispatch_chain(engine, main_ids, kernels, pcs, wgs)
fn engine_build_all(engine, main_ids)
fn engine_run_all(progs)
fn engine_free_all(progs)
fn engine_exec_all(engine, main_ids)
```

Topology is set per-engine and affects how `engine_dispatch_all` routes work. The engine delegates to `topology.flow` functions based on the current mode.

### Mailbox

```
fn engine_create_mailbox(engine, slot_size, slot_count)
fn engine_get_mailbox(engine)
```

Creates a ring-buffer GPU IPC channel. Stored in `engine.mailbox_id`. Arms access it via `main_send`/`main_recv`.

### Polling & Health

```
fn engine_poll_all(engine, main_ids)       // read stability from all arms
fn engine_all_stable(engine, main_ids)     // returns 1.0 if ALL arms stable
fn engine_health(engine, main_ids)         // returns {n_main, n_stable, total_pace, vram_used}
```

**Stability threshold:** `stability_flag < 0.5` = unstable. This treats the flag as boolean at the 0.5 boundary.

### Adapt Loop (The Brain's Decision Cycle)

```
fn engine_adapt(engine, main_ids, precisions)
```

The brain's core decision function. Called periodically. Logic:

1. Poll all arms for `[maxnorm, scale, stability_flag, pace_us]`
2. Count idle arms (`maxnorm < 0.001` AND `stability_flag > 0.5`)
3. Count unstable arms (`stability_flag < 0.5`)
4. **Release trigger:** `n_unstable > n/2` AND `n_idle > 0` → release last arm
5. **Merge trigger:** `n_idle > 1` → release last arm (consolidate)
6. Minimum: will not release below 1 arm

This is meso-homeostasis — engine-wide adaptation based on aggregate arm state.

### Data Distribution

```
fn engine_write_all(engine, main_ids, offset, data)     // broadcast same data to all arms
fn engine_write_chunks(engine, main_ids, data, chunk_size) // scatter data chunks across arms
fn engine_read_all(engine, main_ids, offset, count)     // gather results from all arms
```

**`engine_read_all`** returns an array-of-arrays (one per arm). Compare with `topo_parallel_gather` which flattens into a single array.

### Precision-Aware Dispatch

```
fn engine_dispatch_with_dequant(engine, main_ids, precisions, compute_kernel, compute_pc, compute_wg, dequant_pc, dequant_wg)
fn engine_dispatch_regulated(engine, main_ids, kernel, pc, wg, n_elements)
fn engine_dispatch_full(engine, main_ids, precisions, compute_kernel, compute_pc, compute_wg, dequant_pc, dequant_wg, n_elements)
```

**`engine_dispatch_full`** is the complete pipeline: for each arm, dispatches dequant (per precision) → compute → maxnorm → regulator. This is the engine-level universal dispatch pattern.

### OctoPress Management

```
fn engine_warm_all(engine, main_ids, data)
fn engine_promote_warm_all(engine, main_ids, precisions, dequant_wg)
```

Bulk warm/promote across all arms. Used for layer preloading during inference.

---

## Topology Deep Dive

**File:** `stdlib/loom/topology.flow`
**Functions:** 8
**Role:** Dispatch strategy implementations
**Imports:** `use "main_loom"`

### Parallel Topology

```
fn topo_parallel_dispatch(main_ids, n_looms, kernel, base_pc, chunk_size, wg)
fn topo_parallel_exec(main_ids, n_looms)
fn topo_parallel_write(main_ids, n_looms, data, chunk_size)
fn topo_parallel_gather(main_ids, n_looms, offset, count)
```

Same kernel runs on all arms with different data chunks. Push constant `pc[0]` holds the chunk offset per arm. Build-then-submit-all pattern: builds all programs first, then submits all in a second pass (maximizes GPU parallelism).

**Gather:** `topo_parallel_gather` uses `extend()` to flatten results into a single contiguous array. Different semantic from `engine_read_all` which returns array-of-arrays.

### Sequential Topology

```
fn topo_sequential_dispatch(main_ids, n_looms, kernels, pcs, wgs, globals_size)
fn topo_sequential_read_last(main_ids, n_looms, n_stages, offset, count)
```

Pipeline: output of Loom N copies to Loom N+1 via `loom_copy(src, dst, 0.0, 0.0, globals_size)`. GPU-to-GPU transfer — no CPU roundtrip. Uses `main_exec` per stage (build + run + free inline).

### Hierarchical Topology

```
fn topo_hierarchical_reduce(main_ids, n_looms, kernel, pc, wg, globals_size)
```

Tree reduction. Stride doubles each round: `1 → 2 → 4 → ...`. Each round copies right-neighbor's output into destination's secondary region at `half = globals_size / 2.0`. Reduce kernel push constants: `[0.0, half, half]` — offset A, offset B, count. Final result always lands on `Loom[0]`.

### Auto-Select Heuristic

```
fn topo_auto_select(n_looms, workload_size, n_kernels)
```

| Condition | Result |
|-----------|--------|
| 1 kernel AND `workload_size > n_looms * 1024` | `"parallel"` |
| 1 kernel AND small workload | `"hierarchical"` |
| Multiple kernels | `"sequential"` |
| Default | `"parallel"` |

Threshold: `n_looms * 1024.0` elements.

---

## Three-Level Homeostasis

Homeostasis operates at three levels. This is not a future plan — it is implemented.

```
┌──────────────────────────────────────────────────────────┐
│ MACRO — App-Level                                        │
│ • loom_vram_budget / loom_vram_used (VRAM constraints)   │
│ • Application decides GPU budget, max VMs                │
│ • Thermal queries (loom_status per VM)                   │
└──────────────────────┬───────────────────────────────────┘
                       │ sets constraints
┌──────────────────────▼───────────────────────────────────┐
│ MESO — Engine-Wide (Support Loom)                        │
│ • engine_adapt: split/merge arms based on aggregate      │
│ • engine_reprecision: change arm precision tiers          │
│ • engine_warm_all / promote_warm_all: OctoPress balance   │
│ • Thermal budget: read total_pace, adjust arm count       │
└──────────────────────┬───────────────────────────────────┘
                       │ adjusts thresholds
┌──────────────────────▼───────────────────────────────────┐
│ MICRO — Per-VM, GPU-Autonomous (Main Loom)               │
│ • vm_maxnorm.spv: compute max absolute value             │
│ • vm_regulator.spv: adjust scale if outside bounds       │
│ • loom_pace: timing control per dispatch                 │
│ • Zero CPU roundtrip — runs inside dispatch chain        │
└──────────────────────────────────────────────────────────┘
```

**Shipped builtins per level:**

| Level | Builtins / Functions |
|-------|---------------------|
| Macro | `loom_vram_budget`, `loom_vram_used`, `loom_vm_count`, `loom_vm_info`, `loom_max_vms` |
| Meso | `engine_adapt`, `engine_health`, `engine_all_stable`, `engine_poll_all`, `engine_reprecision`, `engine_spawn`, `engine_release_last`, `engine_warm_all`, `engine_promote_warm_all` |
| Micro | `main_attach_regulator`, `main_update_regulator`, `main_dispatch_regulated`, `main_dispatch_full`, `main_read_stability`, `main_set_pace`, `main_is_stable` |

---

## Swarm Architecture

**File:** `stdlib/loom/swarm.flow`
**Functions:** 10
**Role:** Multi-engine colony
**Imports:** `use "engine"`

### Lifecycle

```
fn swarm_create(n_engines, main_per_engine, globals_size, default_precision, engines, all_main_ids, all_precisions)
fn swarm_shutdown(swarm, engines, all_main_ids)
```

Creates `n_engines` independent engines, each with `main_per_engine` arms. Returns swarm map with `n_engines` and `globals_size`. State is tracked via parallel arrays: `engines[i]`, `all_main_ids[i]`, `all_precisions[i]`.

### Broadcast & Scatter/Gather

```
fn swarm_broadcast_dispatch(swarm, engines, all_main_ids, kernel, pc, wg)
fn swarm_exec_all(swarm, engines, all_main_ids)
fn swarm_write_all(swarm, engines, all_main_ids, offset, data)
fn swarm_scatter(swarm, engines, all_main_ids, data, chunk_per_engine)
fn swarm_gather(swarm, engines, all_main_ids, offset, count)
```

**Broadcast:** Same kernel dispatched to all engines (which then dispatch to all their arms).
**Scatter:** Data chunked across engines — each engine gets `chunk_per_engine` elements.
**Gather:** Reads from `mids[0]` (Loom index 0) of each engine as the primary result holder.

### Cross-Engine Channels

```
fn swarm_create_channel(engine_a, engine_b, slot_size, slot_count)
```

Both engines share the exact same `mailbox_id` — the mailbox is created once via `loom_mailbox` and assigned to both engines. This enables cross-engine, cross-arm communication.

### Health

```
fn swarm_health(swarm, engines, all_main_ids)   // {n_engines, total_main, total_stable, total_pace, vram_used}
fn swarm_all_stable(swarm, engines, all_main_ids)
```

Aggregates health across all engines. `swarm_all_stable` returns `1.0` only if every arm in every engine reports stable.

---

## Builtin → Role Mapping

Complete mapping of all ~50 loom builtins to their tier in the formalized architecture.

### Support Loom (Brain) — Lifecycle & Management

| Builtin | Purpose |
|---------|---------|
| `loom_boot` | Create new VM |
| `loom_shutdown` | Destroy VM |
| `loom_set_heap` | Configure VM heap size |
| `loom_threads` | Set CPU thread pool size |
| `loom_cpu_count` | Query CPU cores |
| `loom_auto_spawn` | Allocate VM from pool |
| `loom_auto_release` | Return VM to pool |
| `loom_park` | Park idle VM |
| `loom_unpark` | Wake parked VM |
| `loom_pool_size` | Query pool size |
| `loom_pool_warm` | Pre-warm pool |
| `loom_pool_info` | Pool diagnostics |
| `loom_max_vms` | Set VM limit |
| `loom_vm_count` | Count active VMs |
| `loom_vm_info` | VM diagnostics |
| `loom_vram_budget` | Set VRAM budget |
| `loom_vram_used` | Query VRAM usage |
| `loom_mailbox` | Create mailbox |
| `loom_copy` | GPU-to-GPU data transfer |
| `loom_async_read` | Async file read |
| `loom_await` | Wait for async result |
| `loom_prefetch` | Prefetch file data |

### Main Loom (Arm) — GPU Compute

| Builtin | Purpose |
|---------|---------|
| `loom_dispatch` | Queue kernel dispatch |
| `loom_dispatch_jit` | Queue JIT-emitted kernel |
| `loom_build` | Compile dispatch chain |
| `loom_run` | Execute program |
| `loom_free` | Free program |
| `loom_launch` | Async execute |
| `loom_poll` | Check async status |
| `loom_wait` | Wait for async completion |
| `loom_write` | CPU → GPU data |
| `loom_read_globals` | GPU → CPU data (3-arg) |
| `loom_read` | GPU → CPU data (4-arg, register) |
| `loom_write_control` | Write control SSBO |
| `loom_read_control` | Read control SSBO |
| `loom_write_metrics` | Write metrics SSBO |
| `loom_read_metrics` | Read metrics SSBO |
| `loom_mail_send` | Send to mailbox |
| `loom_mail_recv` | Receive from mailbox |
| `loom_mail_poll` | Check mailbox depth |
| `loom_mail_depth` | Query mailbox depth |
| `loom_status` | VM status query |
| `loom_pace` | Set dispatch pacing |
| `loom_elapsed_us` | Last dispatch timing |
| `loom_dispatch_time` | Dispatch timing |
| `loom_present` | Display output |

### Express Loom (One-Call) — Convenience

| Builtin | Purpose |
|---------|---------|
| `loom_compute` | One-call GPU compute (auto VM management) |
| `loom_workgroups` | Compute workgroup count: `ceil(n/256)` |

### Pipeline Reuse

| Builtin | Purpose |
|---------|---------|
| `loom_pipe` | Create reusable dispatch chain |
| `loom_pipe_add` | Add dispatch to pipeline |
| `loom_pipe_exec` | Execute pipeline |

### OctoPress (Weight Compression)

| Builtin | Purpose |
|---------|---------|
| `octopress_init` | Initialize compressor |
| `octopress_analyze` | Analyze data statistics |
| `octopress_encode` | Compress array |
| `octopress_decode` | Decompress array |
| `octopress_save` | Save to .octopress file |
| `octopress_load` | Load from .octopress file |
| `octopress_info` | Inspect file metadata |
| `octopress_gpu_encode` | GPU-accelerated compression |
| `octopress_stream_open` | Open streaming decompression |
| `octopress_stream_next` | Read next block |
| `octopress_stream_info` | Query stream state |
| `octopress_stream_reset` | Rewind stream |
| `octopress_stream_close` | Close stream |

---

## Sieve Pattern Generalization

The prime sieve v7 (24 kernels in `stdlib/loom/kernels/sieve/`) established the fundamental dual-kernel architecture:

1. **Small kernel** — selects, marks, or decompresses (runs on subset)
2. **Large kernel** — computes on full data (runs on everything)

This pattern generalizes to every precision-adaptive dispatch in the Loom Engine:

| Domain | Small Kernel | Large Kernel |
|--------|-------------|-------------|
| Prime sieve | `sieve_mark_v7_large` | `sieve_accum_v2` |
| Q4_K inference | `vm_dequant_q4k` | `vm_matvec` |
| Q6_K inference | `dequant_q6k` | `vm_matvec` |
| Regulated compute | `vm_maxnorm` + `vm_regulator` | (any compute kernel) |

The `main_dispatch_full` function is the universal form: dequant (small) → compute (large) → maxnorm (small) → regulator (small). Four-kernel chain, one submission, zero CPU roundtrip.

---

## File Reference

| File | Functions | Lines | Role |
|------|-----------|-------|------|
| `stdlib/loom/main_loom.flow` | 30 | — | GPU arm: dispatch, OctoPress, micro-homeostasis, JIT |
| `stdlib/loom/engine.flow` | 28 | — | Brain: lifecycle, adapt, topology, data distribution |
| `stdlib/loom/topology.flow` | 8 | — | Parallel / Sequential / Hierarchical strategies |
| `stdlib/loom/swarm.flow` | 10 | — | Multi-engine colony |
| `stdlib/loom/octopress.flow` | 6 | — | High-level compress/decompress wrappers |
| `stdlib/loom/octopress_stream.flow` | 13 | — | Block-by-block streaming decompression |
| **Total** | **95** | | |

### Kernel Files Referenced

| Kernel | Path | Used By |
|--------|------|---------|
| `vm_maxnorm.spv` | `stdlib/loom/kernels/ops/` | Micro-homeostasis (all regulated dispatches) |
| `vm_regulator.spv` | `stdlib/loom/kernels/ops/` | Micro-homeostasis (scale adjustment) |
| `vm_dequant_q4k.spv` | `stdlib/loom/kernels/ops/` | Q4_K precision dequantization |
| `dequant_q6k.spv` | `stdlib/llm/kernels/` | Q6_K precision dequantization |

---

## Implementation Notes

1. **homeostasis.flow does not exist as a separate file.** Micro-homeostasis is in `main_loom.flow` (7 functions). Meso-homeostasis is in `engine.flow` (`engine_adapt`). Macro-homeostasis uses raw builtins. This is a cross-cutting concern, not a module.

2. **OctoPress builtins are fully shipped.** The `octopress.flow` wrapper file notes "ready for when Dev ships those builtins" but the builtins (`octopress_encode`, `octopress_decode`, etc.) are all implemented in the runtime at lines 8111-8697.

3. **`loom_pipe_add` preflight limitation.** The 4-arg call cannot go through a wrapper function due to a preflight checker constraint. Must be called directly. Documented in `main_loom.flow`.

4. **Sequential topology uses `loom_copy`.** This is GPU-to-GPU transfer — no CPU roundtrip between pipeline stages. The `loom_copy` builtin does a GPU-side buffer copy.

5. **Swarm channels share mailbox IDs.** `swarm_create_channel` assigns the same `mailbox_id` to both engines. This is the mechanism for cross-engine communication.
