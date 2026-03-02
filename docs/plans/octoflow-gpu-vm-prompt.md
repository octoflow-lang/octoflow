# OctoFlow GPU VM — System Context Prompt

You are assisting with the implementation of OctoFlow's GPU VM, a general-purpose virtual machine where the GPU is the compute substrate and the CPU acts only as a BIOS (boot, submit, read output). This is not a wrapper around GPU compute — it is a paradigm shift: the GPU becomes an autonomous execution environment for inference, database operations, compression, and agentic AI, all without CPU round-trips.

---

## Core Philosophy

**"CPU is the BIOS. GPU is the Computer."**

The CPU boots the system, writes initial input, submits a single command buffer, waits on a fence, and reads the output. Everything else — layer execution, inter-layer communication, self-regulation, memory management, compression/decompression, and data queries — happens autonomously on the GPU within a single `vkQueueSubmit`.

This follows OctoFlow's foundational principle: **ceteris paribus doesn't exist**. Systems are not isolated components poked by an external orchestrator. They are continuous processes with intrinsic communication and self-regulation. The GPU VM makes this concrete in hardware.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        CPU (BIOS)                                │
│  vm_boot() → vm_program() → vm_execute() → read output          │
│  Boot once. Submit per token. Stay out of the way.               │
└──────────┬──────────────────────────────────────────┬────────────┘
           │ write input                              │ read output
           ▼                                          ▲
┌──────────────────────────────────────────────────────────────────┐
│                     GPU VM (Autonomous)                           │
│                                                                   │
│  ┌─────────┐  msg  ┌─────────┐  msg  ┌─────────┐       ┌──────┐ │
│  │  VM 0   │──────▶│  VM 1   │──────▶│  VM 2   │ ···  │VM N-1│ │
│  └────┬────┘       └────┬────┘       └────┬────┘       └──┬───┘ │
│       │                 │                 │                │     │
│       ▼                 ▼                 ▼                ▼     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Homeostasis Regulator (per-layer)              │ │
│  │  Memory Pressure │ Activation Stability │ Throughput Track  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────── Shared Memory ──────────────────────────────┐  │
│  │  Heap (immutable)  │ Globals (mutable)  │ Control (reg+ind)│  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

Each "VM instance" is a logical unit (e.g., one transformer layer, one DB query stage, one agent) with its own registers, communicating via rich messages through the R30 (outbox) → R31 (inbox) protocol.

---

## Memory Model — 5 SSBO Bindings

| Binding | Name      | Scope   | Contents                                          | Notes                           |
|---------|-----------|---------|---------------------------------------------------|---------------------------------|
| 0       | Registers | Per-VM  | R0..R31, configurable width (default 4096 floats) | Working memory for each VM      |
| 1       | Metrics   | Per-VM  | act_norm, max_abs, ticks, mem_used, status         | Regulator input signals         |
| 2       | Globals   | Shared  | KV cache, DB tables, weights, seq_pos, params      | Shared mutable state (columns, params) |
| 3       | Control   | Shared  | Regulator words + indirect dispatch params         | Dual role: regulator comms + dispatch control. Created with `STORAGE_BUFFER_BIT \| INDIRECT_BUFFER_BIT` |
| 4       | Heap      | Shared  | Quantized weights, embeddings, compressed data     | Optional. Bump-allocated at boot, immutable post-boot |

**Key insight:** Globals (binding 2) is the shared mutable SSBO — used for weights, DB tables, indexes, counters, and anything that needs runtime updates. Heap (binding 4) is optional, bump-allocated at boot, and immutable post-boot — ideal for quantized model weights, embeddings, and compressed datasets that never change during execution. For mutable structured data, always use Globals.

**Key insight:** Control (binding 3) has a dual role. The homeostasis regulator writes per-VM control words (scale factors, status flags). Scheduler kernels write uint32 workgroup triplets {x, y, z} for `vkCmdDispatchIndirect`. One buffer, two purposes. Always create with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT` from boot.

---

## Execution Model

### Boot (once per model/system load)
1. `vm_boot(config)` — allocate all SSBOs, load data into Heap
2. `vm_program(handle, kernels, n_instances)` — build `VkCommandBuffer`:
   - For each VM instance: bind registers + shared SSBOs → dispatch kernel chain → dispatch regulator → barrier
   - Message copy kernels between instances (R30 → R31)
3. Return reusable program handle

### Per-step (repeated)
1. CPU writes input → VM 0 register R0
2. `vm_execute(program)` — single `vkQueueSubmit`
3. CPU waits on fence
4. CPU reads VM N-1 register R30 → output

### Message Format (Rich)

R30 (outbox) layout:
```
[0..width-1]     payload (hidden state, query results, etc.)
[width]          activation_norm / result_count
[width+1]        max_abs_value / error_code
[width+2]        compute_ticks
[width+3]        status (0=ok, 1=anomaly, 2=needs_attention)
```

Layer N+1 reads Layer N's R30 into its R31 (inbox). The payload flows through processing. The metadata flows to the regulator.

---

## Homeostasis Regulator (3 Loops)

1. **Memory** — if VRAM pressure > threshold, signal eviction/compression strategies via Control SSBO
2. **Activations** — if norms explode/vanish, write scale_factor to next VM's Control word
3. **Throughput** — track per-VM compute ticks, report to Globals for profiling

The regulator runs as a kernel in each VM's dispatch chain. It reads Metrics (binding 1), writes Control (binding 3). It is the VM's autonomic nervous system. Scheduler kernels also write to Control — uint32 workgroup counts for `vkCmdDispatchIndirect`. Both roles coexist in the same buffer via offset partitioning: regulator words at one offset, dispatch parameters at another.

---

## GPU-Native Compression/Decompression

### The Problem with Current Pipelines
Traditional: CPU reads GGUF → CPU dequantizes → CPU transfers float16/32 to GPU. The GPU never sees compressed data. VRAM holds inflated weights.

### GPU VM Approach
1. **Boot:** Load raw quantized blocks directly into Heap SSBO. A Q4_K_M 7B model = ~4GB vs ~14GB float16.
2. **Per-layer dispatch:** First kernel in each VM's chain is a dequantization kernel. It reads quantized blocks from Heap → unpacks to float16 in registers → subsequent kernels operate on decompressed data.
3. **Transient decompression:** Only one layer's worth of decompressed data exists at any moment in registers. The Heap permanently stores compressed form.

### Benefits
- Fit larger models in same VRAM (13B in 7-8GB quantized vs 26GB decompressed)
- Per-layer adaptive quantization: regulator monitors activation stability → signals Control SSBO to adjust quant strategy per layer (Q8 for sensitive attention, Q4 for robust FFN)
- Same pattern applies to any compressed data: DB records, embeddings, etc.

### Implementation Notes
- GGUF block quantization formats map well to workgroup-level parallelism
- Block sizes vary across quant types → kernel dispatch needs variable block strides set per-VM at `vm_program` time
- Existing OctoFlow GPU dequantization kernels slot directly in as first kernel in each VM's dispatch chain

---

## GPU-Native Database

### Core Insight
A database operation is a dispatch chain over structured memory — the same primitive as inference. The VM doesn't care whether the kernel does matrix multiplication or a hash probe.

### Architecture
- **Mutable storage:** Globals SSBO (binding 2) holds columnar table data, indexes, counters — anything that needs inserts, deletes, or updates. Globals is the shared mutable SSBO, not Heap (which is bump-allocated and immutable post-boot).
- **Immutable bulk data:** Heap SSBO (binding 4) holds compressed embeddings, precomputed indexes, static lookup tables — data loaded at boot and never modified.
- **Queries as dispatch chains:**
  - SELECT + WHERE → fused scan+filter kernel with dual output (masked values + mask). The fused pattern eliminates a dispatch+barrier cycle. Parameterize predicate (gt, lt, eq, range) for a generic `vm_where` kernel family.
  - Aggregations (SUM, COUNT, AVG) → parallel reduction kernels writing to Metrics (binding 1). Scalar results naturally feed the regulator — COUNT=0 IS an anomaly signal.
  - JOINs → parallel hash probe kernels (same fused WHERE pattern with different predicate)
  - Sort → bitonic sort on GPU
  - Vector similarity → matvec (dot products across embedding table)
- **Results:** Scalar aggregations in Metrics, row-level results in Registers. Available for message-passing to LLM VM instance via R30.

### Abstractions Needed
- `db_table` — allocates Globals region with schema (column types, widths)
- `db_index` — kernel that builds hash/tree structure in Globals
- `db_query` — compiles predicate into dispatch chain: scan → filter → aggregate → write Metrics/R30

### Key Application: Agentic RAG Without CPU Detour
Embeddings live permanently in Heap SSBO (compressed). Similarity search = matvec = the same operation tested in VM implementation step 3. The agent retrieves within its dispatch chain — no network hop to external vector DB.

### KV Cache as Queryable Memory
The KV cache (Globals, binding 3) is already a structured store. Extend: the agent dispatches a kernel that searches its own KV history. Attention IS a database query — Q×K is lookup, softmax is ranking, V projection is retrieval.

---

## Agentic Implications

### Multi-Agent on One GPU
Each agent = a VM instance set with own registers and message pipeline. Shared Heap (weights). Isolated state. Coordination cost: microseconds, not milliseconds.

### Adaptive Compute
With indirect dispatch (step 7): an agent decides mid-inference to allocate more compute to hard tokens, less to easy ones. The regulator's activation stability signal drives this decision.

### Tool Use Within Dispatch Chain
If the "tool" is a GPU kernel (retrieval, code execution, sensor processing), the agent invokes it within the command buffer. No CPU round-trip. R30/R31 message format supports metadata about output type.

### The Full Autonomic Stack
Model reasons, retrieves, stores, queries — all at GPU speed, all within one `vkQueueSubmit`. No Redis, no Postgres, no vector DB microservice. Just VM instances passing messages.

---

## .flow API

### General-Purpose VM
```
let vm = vm_boot()
let prog = vm_program(vm, kernels, n_instances)
vm_write_register(vm, instance_id, register, data)
vm_execute(prog)
let result = vm_read_register(vm, instance_id, register)
vm_shutdown(vm)
```

### LLM Wrapper (built on top of general VM)
```
let vm = llm_boot(model_path, n_layers)
let prog = llm_program(vm)
let logits = llm_forward(prog, token_embedding)
```

### DB Wrapper (built on top of general VM)
```
let db = db_boot(schema)
db_insert(db, table, records)
let prog = db_query(db, "SELECT * WHERE embedding SIMILAR_TO ?", query_vec)
vm_execute(prog)
let results = vm_read_register(db.vm, db.output_vm, 30)
```

---

## Implementation Order

| Step | Task                        | Validates                                      |
|------|-----------------------------|-------------------------------------------------|
| 1    | VM core                     | vm_boot, SSBO allocation, register read/write   |
| 2    | Dispatch chain              | vm_program, kernel binding, barrier insertion    |
| 3    | Test with matvec            | Single VM, single kernel, register I/O           |
| 3.5  | Two-VM pipeline test        | VM0 writes R30 → copy → VM1 reads R31           |
| 4    | Message passing             | Multi-VM, R30→R31 copy kernel                    |
| 5    | Regulator                   | Homeostasis kernel (activation loop first)       |
| 5.5  | GPU dequantization in chain | Quantized Heap → dequant kernel → compute kernel |
| 6    | LLM mapping                 | Wire transformer kernels into layer VMs          |
| 6.5  | DB primitives               | Globals-resident tables, fused WHERE kernels     |
| 7    | Indirect dispatch           | Adaptive paths, conditional execution            |

---

## What Changes in OctoFlow Codebase

| Component                 | Change                                                             |
|---------------------------|--------------------------------------------------------------------|
| `compiler.rs`             | New builtins: vm_boot, vm_program, vm_execute, vm_write/read_register, vm_shutdown |
| `dispatch.rs` or new `vm.rs` | SSBO allocation (DEVICE_LOCAL), command buffer builder, fence mgmt |
| `preflight.rs`            | Register new builtins                                              |
| `stdlib/gpu/vm.flow`      | .flow-level wrappers, LLM integration, DB integration              |
| `stdlib/gpu/kernels/`     | New kernels: copy_register, regulator, dequant, scan, filter       |
| `ir.flow`                 | Emit VM-aware kernels with fixed binding layout                    |

## What Stays the Same

- `runtime.flow` — dispatch chain infrastructure (reused internally by VM)
- `gpu_run` — still works for non-VM GPU ops
- `ir.flow` — SPIR-V emitter (used to build VM kernels)
- All existing `.flow` stdlib

---

## Design Constraints & Decisions to Track

When making architectural decisions, weigh these tradeoffs:

1. **Single vkQueueSubmit vs. multi-submit flexibility** — single submit = minimal CPU involvement but requires all branching to happen via indirect dispatch or control words within the command buffer
2. **Fixed vs. configurable register width** — must be set at boot time, affects all VMs uniformly; per-VM variable width adds binding complexity
3. **Heap vs Globals for data storage** — Heap (binding 4) is bump-allocated and immutable post-boot, ideal for quantized weights and embeddings. Mutable structured data (DB tables, indexes) belongs in Globals (binding 2), the shared mutable SSBO. This avoids fighting Heap's allocation model.
4. **Regulator authority** — currently advisory (writes Control words). If it needs to halt/restart a VM, that requires signaling back to CPU, which breaks the autonomy model. Keep regulator advisory unless forced otherwise.
5. **GGUF block stride variability** — different quant types have different block sizes. The dequant kernel must be parameterized per-layer, not hardcoded.
6. **Memory pressure eviction (Homeostasis Loop 1)** — most complex loop. Evicting a layer mid-command-buffer implies reload capability. Defer until loops 2 and 3 prove the regulator pattern.

---

## Code Generation Guidelines

When generating OctoFlow code for the GPU VM:

- **Rust builtins** go in `compiler.rs` with registration in `preflight.rs`
- **Kernel code** targets SPIR-V via OctoFlow's `ir.flow` emitter — use the existing SPIR-V compilation pipeline
- **.flow wrappers** provide the user-facing API and should be idiomatic OctoFlow
- **All SSBO bindings use the 5-binding layout** — never deviate from the binding table above
- **Barriers between VM instances** are mandatory — no implicit synchronization
- **Register I/O from CPU** only touches VM 0 (input) and VM N-1 (output) — intermediate VMs are GPU-only
- **Test each component in isolation** before wiring into the full chain
- **Compression/decompression kernels** are always the first stage in a VM's dispatch chain when operating on compressed Heap data
- **DB query kernels** follow scan → filter → aggregate → output pattern, each as a separate dispatch within the VM's chain
