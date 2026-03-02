# Loom Engine Phase 4: Adaptive Computing Fabric

**Status:** Architecture Document — Pre-Implementation
**Date:** March 1, 2026
**Prerequisite:** Phase 3P (Polish) complete

---

## Vision

> Turn a $200 consumer GPU into a self-optimizing compute fabric
> that adapts kernels, precision, parallelism, and memory on the fly.

Phase 3 built the Loom Engine: Main Looms for GPU compute, Support Loom for I/O,
homeostasis for pacing, threading for throughput. Phase 4 makes it **adaptive**.

Seven primitives, each building on what already ships:

| # | Primitive | What It Does | Builds On |
|---|-----------|-------------|-----------|
| 1 | **JIT Adaptive Kernels** | Generate precision-optimal kernels at runtime | IR Builder (85 ops, 60 emitters) |
| 2 | **On-Demand Main Looms** | Support Loom spawns/destroys compute units dynamically | Multi-VM (16 proven in sieve) |
| 3 | **Mailbox** | Cross-loom message passing, signal buffers | loom_copy (GPU-to-GPU DMA) |
| 4 | **OctoZip** | GPU-native fractal compression/decompression engine | Delta encode/decode kernels, fractal emitter |
| 5 | **Multi-Stack Topologies** | Stacks of Loom stacks: hierarchical, sequential, parallel | Homeostasis, queue mutex |
| 6 | **Varying Bit Compute** | Per-loom precision: FP32, FP16, INT8, INT4, ternary, 1-bit | JIT kernels + on-demand looms |
| 7 | **CPU Thread Pool** | Multi-threaded runtime services for Support Loom (BIOS layer) | Phase 3O threading (async present, prefetch) |

Together these form the **Loom Computer Model**: CPU = BIOS + runtime services,
Support Loom = GPU Operating System, Main Looms = processes, GPU = the computer.

---

## Foundation Inventory (Already Shipped)

Everything Phase 4 builds on exists today:

### Kernel Infrastructure
- **IR Builder:** 85 SPIR-V ops via `ir_new()` → `ir_write_spv()` pipeline
- **Kernel Emitters:** 60 .flow emitters in `stdlib/loom/emit/`
- **Pre-compiled Kernels:** 23 .spv binaries in `stdlib/loom/kernels/ops/`
- **Shared Memory:** `ir_shared_load()`, `ir_shared_store()`, `ir_barrier()` — configurable size
- **Atomics:** `ir_buf_atomic_load/store/iadd/and` — device + workgroup scope
- **Fractal Emitter:** `emit_fractal.flow` — Mandelbrot, Julia, Burning Ship via IR builder

### Loom Infrastructure
- **Multi-VM:** 16 VMs concurrent (sieve_gpu_v3.flow) — boot, dispatch, async launch, poll, collect
- **loom_boot:** `(n_instances, reg_size, globals_size) → vm_id` — allocates registers + globals + metrics + control + heap
- **loom_copy:** `(src_vm, src_offset, dst_vm, dst_offset, count)` — GPU-to-GPU DMA via `vkCmdCopyBuffer`, no CPU roundtrip
- **Homeostasis:** Per-VM adaptive pacing — EMA baseline, pace debt, settlement at `vm_present`
- **Queue Mutex:** `std::sync::Mutex<()>` wrapping `vkQueueSubmit` — thread-safe submission
- **SPIR-V Cache:** Thread-local `HashMap<String, Vec<u8>>` — eliminates per-dispatch disk I/O
- **Prefetch Thread:** Background file read via `JoinHandle` — overlaps I/O with compute

### Compression Infrastructure
- **Delta Encode/Decode:** `emit_vm_delta_encode.flow` + `emit_vm_delta_decode.flow` — GPU parallel delta coding
- **Dictionary Lookup:** `vm_dict_lookup.spv` — gather from heap at arbitrary indices
- **Dequantization:** `emit_dequant_q4k.flow` — Q4_K block dequant on GPU
- **.octo Column Format:** `ENCODING_RAW` + `ENCODING_DELTA` with per-column codec

### Inference Pipeline
- **Layer Streaming:** `gguf_prefetch_layer()` → `gguf_infer_layer()` → `gguf_evict_layer()`
- **Buffer Caches:** `GPU_BUFFER_CACHE` + `TENSOR_CACHE` + `FILE_CACHE` + `LAYER_RESIDENCY`
- **VRAM Tracking:** `GPU_CACHE_BYTES` — total VRAM usage counter

---

## Primitive 1: JIT Adaptive Kernels

### What Exists
The IR builder generates SPIR-V at runtime from .flow code. 60 kernel emitters
produce specialized compute shaders. This is already JIT compilation — we just
haven't used it adaptively.

### What's New
The Support Loom analyzes data characteristics and JIT-compiles **precision-optimal
kernels** instead of using fixed pre-compiled .spv files.

### Design

```
SUPPORT LOOM (runtime analysis)
│
├── Profile data: measure value ranges, sparsity, distribution
│   └── One-pass GPU kernel: min/max/mean/variance per block
│
├── Select precision per operation:
│   ├── Values in [-1, 1] with low variance → FP16 sufficient
│   ├── Integer indices only → UINT32 (no float overhead)
│   ├── Sparse data (>80% zeros) → compressed sparse format
│   └── High dynamic range → FP32 required
│
├── JIT compile with ir_new():
│   ├── ir_new("matmul_fp16", 1)    // FP16 kernel
│   ├── ir_new("gather_u32", 1)     // integer-only kernel
│   └── ir_write_spv("adaptive.spv")
│
└── Cache compiled kernel:
    └── SPIRV_FILE_CACHE already handles this — zero cost on reuse
```

### Kernel Template Pattern

Instead of one emitter per operation, parameterized emitters accept precision:

```
fn emit_matmul(name, M, N, K, precision)
  ir_new(name, 1)
  // precision = 0: FP32, 1: FP16, 2: INT8
  if precision == 0.0
    // standard float multiply-accumulate
    let prod = ir_fmul(block, a_val, b_val)
    let acc = ir_fadd(block, sum, prod)
  end
  if precision == 1.0
    // pack two FP16 values per 32-bit word
    // use float_to_bits / bits_to_float for bit manipulation
    let packed = ir_fmul(block, a_half, b_half)  // GPU FP16 via relaxed precision
    let acc = ir_fadd(block, sum, packed)
  end
  if precision == 2.0
    // INT8: values pre-quantized to [0,255] range
    // integer multiply via float with floor/round
    let prod = ir_fmul(block, a_byte, b_byte)
    let acc = ir_fadd(block, sum, ir_floor(block, prod))
  end
  ir_write_spv(name + ".spv")
end
```

### Speculative JIT
To avoid first-use latency, the Support Loom can pre-compile likely kernels:

```
// During model load (before first inference)
emit_matmul("attn_fp16", hidden, hidden, hidden, 1.0)   // attention usually FP16
emit_matmul("mlp_int8", hidden, ffn, hidden, 2.0)       // MLP tolerates INT8
emit_matmul("norm_fp32", hidden, 1, 1, 0.0)             // norms need FP32

// All cached in SPIRV_FILE_CACHE — zero cost at inference time
```

### Missing IR Ops Needed

The IR builder currently lacks integer bitwise operations as builtins. For
precision packing and OctoZip compression, we need:

| Op | SPIR-V | Purpose |
|----|--------|---------|
| `ir_iand(block, a, b)` | OpBitwiseAnd | Bit masking, hash functions |
| `ir_ior(block, a, b)` | OpBitwiseOr | Flag sets, bit packing |
| `ir_ixor(block, a, b)` | OpBitwiseXor | Hash mixing, checksums |
| `ir_ishl(block, a, bits)` | OpShiftLeftLogical | Bit packing, addressing |
| `ir_ishr(block, a, bits)` | OpShiftRightLogical | Bit unpacking, extraction |
| `ir_popcount(block, a)` | OpBitCount | Compression analysis, sieve counting |

**Note:** `ir_buf_atomic_and` (op 92) exists for atomic bitwise AND.
`float_to_bits()` / `bits_to_float()` exist for IEEE 754 reinterpretation.
The missing ops are the non-atomic integer bitwise operations.

### Edge Cases

| Edge Case | Risk | Mitigation |
|-----------|------|------------|
| JIT kernel explosion — too many unique kernels | Cache bloat, compile overhead | Template reuse: cache by (op, shape, precision) key. Limit to ~32 live kernels. |
| Precision autodetect wrong — picks INT8 where FP16 needed | Quality degradation | Validation pass: compare JIT output vs FP32 reference on sample batch, escalate if error > threshold |
| Speculative JIT compiles unused kernels | Wasted startup time | Lazy: only pre-compile top 3 most likely configs; JIT remainder on first use |
| Relaxed precision not supported on hardware | Wrong results | Query `gpu_info()` for `shader_float16_support`, fall back to FP32 emulation |

### Test Plan
- `test_jit_precision_select` — verify FP16 kernel produces same output as FP32 within tolerance
- `test_jit_kernel_cache` — verify second dispatch of same (shape, precision) reuses cached .spv
- `test_jit_speculative` — verify pre-compiled kernels available without JIT delay at inference time

---

## Primitive 2: On-Demand Main Looms

### What Exists
`loom_boot()` creates a VM. `loom_shutdown()` destroys it. The prime sieve boots
16 VMs in a loop. This is manual — the programmer decides how many VMs.

### What's New
The Support Loom becomes an **orchestrator** that spawns and destroys Main Looms
based on workload, within a resource budget.

### Design

```
SUPPORT LOOM
│
├── Resource Budget
│   ├── VRAM budget: gpu_info().total_memory * 0.85 (leave 15% headroom)
│   ├── VM budget: loom_max_vms(n) — hard cap, default 16
│   └── Per-VM cost: reg_size * 4 * 32 * instances + globals_size * 4 bytes
│
├── Spawn Policy
│   ├── Workload > threshold → spawn new Main Loom
│   ├── Batch items > 1 → spawn parallel inference looms
│   ├── Pipeline stage idle > N frames → don't respawn
│   └── VRAM < budget floor → refuse spawn, log warning
│
├── Reuse Pool (avoid thrashing)
│   ├── loom_park(vm) — idle VM, keep allocated, skip dispatch
│   ├── loom_unpark(vm) — resume dispatching to parked VM
│   └── Parked VMs reclaimed after N seconds idle
│
└── Lifecycle Events
    ├── on_spawn(vm_id) → log, update metrics
    ├── on_park(vm_id) → stop dispatching, keep buffers
    ├── on_unpark(vm_id) → resume dispatching
    └── on_shutdown(vm_id) → free all resources, remove from pool
```

### API

```
// Budget control
loom_max_vms(n)              // set hard cap on total VMs
loom_vram_budget(bytes)      // set VRAM budget for dynamic allocation

// Pool management
loom_park(vm)                // idle a VM without destroying it
loom_unpark(vm)              // resume a parked VM
loom_pool_size()             // count active + parked VMs

// Adaptive spawn (Support Loom decides)
loom_auto_spawn(reg_size, globals_size)  // spawn if budget allows, else reuse parked
loom_auto_release(vm)                    // park or shutdown based on pool policy
```

### Spawn-Destroy Hysteresis

To prevent thrashing (rapid spawn/destroy cycles), use a cooldown:

```
SPAWN:    only if workload sustained > threshold for 3 consecutive frames
PARK:     only if VM idle for > 10 frames (not immediate)
SHUTDOWN: only if parked for > 5 seconds AND VRAM pressure exists
```

### Edge Cases

| Edge Case | Risk | Mitigation |
|-----------|------|------------|
| Loom spawn storm — workload spike spawns too many VMs | VRAM exhaustion | Hard cap via `loom_max_vms()`, VRAM watermark check before `loom_boot` |
| Spawn/destroy thrashing — rapid create/kill cycles | Overhead exceeds benefit | Hysteresis: park before shutdown, reuse pool, cooldown timers |
| Orphaned VMs — code path skips shutdown | Memory leak | Support Loom tracks all VMs, `loom_shutdown_all()` at program exit |
| Parked VM holds VRAM but never used again | Wasted memory | Timeout: reclaim parked VMs after 5s idle if VRAM pressure |
| VM budget exceeded silently | Unexpected OOM | `loom_auto_spawn` returns error/0 if budget exceeded, never silent |

### Test Plan
- `test_auto_spawn_budget` — verify spawn fails gracefully when VRAM budget exceeded
- `test_park_unpark_reuse` — verify parked VM resumes correctly with existing buffer data
- `test_hysteresis` — verify rapid spawn requests within cooldown are coalesced
- `test_shutdown_all` — verify all VMs cleaned up at exit, zero VRAM leak

---

## Primitive 3: Mailbox — Cross-Loom Communication

### What Exists
`loom_copy(src_vm, src_offset, dst_vm, dst_offset, count)` copies GPU memory
between VMs. This is raw DMA — no signaling, no ordering, no protocol.

### What's New
A structured message-passing primitive built on `loom_copy` with signaling,
ordering guarantees, and backpressure.

### Design

```
MAILBOX BUFFER LAYOUT (GPU memory)
┌──────────────────────────────────────────────┐
│ Header (4 floats):                            │
│   [0] seq_write   — sender's sequence number  │
│   [1] seq_read    — receiver's sequence number │
│   [2] payload_len — floats in current message  │
│   [3] flags       — 0:empty, 1:ready, 2:overflow │
├──────────────────────────────────────────────┤
│ Ring Buffer (capacity floats):                │
│   [4..4+capacity] — circular payload storage  │
│   write_pos = (seq_write * slot_size) % capacity │
│   read_pos  = (seq_read * slot_size) % capacity  │
└──────────────────────────────────────────────┘
```

### Protocol

The Support Loom mediates all mailbox operations — Main Looms never directly
access each other's memory. This eliminates race conditions by design.

```
SEND (Support Loom orchestrates):
1. Main Loom A dispatches kernel → writes payload to its own globals
2. Support Loom reads signal word from Loom A: loom_read(vm_a, signal_offset, 1)
3. If signal == READY:
   a. loom_copy(vm_a, payload_offset, mailbox_vm, write_pos, payload_len)
   b. Increment seq_write in mailbox header
   c. Clear Loom A's signal word
4. If mailbox full (seq_write - seq_read >= capacity/slot_size):
   → Set overflow flag, skip copy (backpressure)

RECV (Support Loom orchestrates):
1. Support Loom checks mailbox: loom_read(mailbox_vm, 0, 4) → header
2. If seq_write > seq_read (message available):
   a. loom_copy(mailbox_vm, read_pos, vm_b, dest_offset, payload_len)
   b. Increment seq_read in mailbox header
3. If empty: return 0 (no message)
4. Main Loom B dispatches kernel → reads payload from its own globals
```

### API

```
// Lifecycle
let mb = loom_mailbox(capacity)           // create mailbox VM with ring buffer
loom_mailbox_destroy(mb)                  // free mailbox VM

// Sending
loom_mail_send(mb, src_vm, src_offset, count)  // copy from src to mailbox ring
loom_mail_try_send(mb, src_vm, src_offset, count) // non-blocking, returns 0 if full

// Receiving
loom_mail_recv(mb, dst_vm, dst_offset)    // copy next message to dst, returns count
loom_mail_try_recv(mb, dst_vm, dst_offset) // non-blocking, returns 0 if empty

// Status
loom_mail_poll(mb)          // 1.0 if message available, 0.0 if empty
loom_mail_depth(mb)         // count of pending messages in ring
loom_mail_seq(mb)           // current sequence number (total messages sent)
```

### Why No Race Conditions
1. Main Looms write to **their own** globals — never to another VM's memory
2. Support Loom is the **only** entity that calls `loom_copy` between VMs
3. `loom_copy` goes through queue mutex — serialized submission
4. Support Loom dispatch ordering is sequential — send completes before recv starts
5. Ring buffer indices are managed by Support Loom CPU code, not GPU atomics

The mailbox VM is a lightweight VM (small globals, no registers) that exists
solely as a shared buffer. It never runs kernels itself.

### Edge Cases

| Edge Case | Risk | Mitigation |
|-----------|------|------------|
| Message ordering — two sends, recv gets wrong one | Logic error | Sequence numbers in header; recv always reads oldest first (FIFO) |
| Mailbox overflow — sender faster than receiver | Lost messages | Ring buffer: oldest messages overwritten OR backpressure (try_send returns 0) |
| Deadlock — A waits for B's mailbox, B waits for A's | Hang | Timeout: `loom_mail_recv` has optional timeout_ms parameter; Support Loom detects circular wait |
| Orphaned mailbox — creator crashes | Memory leak | Support Loom tracks all mailboxes; `loom_shutdown_all()` cleans up |
| Large payload — mailbox used for bulk transfer | Perf regression | Size limit on mailbox capacity; recommend `loom_copy` for bulk, mailbox for signals + small data |
| Multiple senders to one mailbox | Interleaved messages | Per-sender slot reservation OR single-sender-per-mailbox policy |

### Test Plan
- `test_mailbox_send_recv` — basic send/recv between 2 VMs, verify data integrity
- `test_mailbox_fifo_order` — send 10 messages, verify recv in sequence order
- `test_mailbox_overflow` — fill ring, verify backpressure (try_send returns 0)
- `test_mailbox_multi_vm` — 4 VMs sending to 1 mailbox, verify no corruption
- `test_mailbox_empty_recv` — recv on empty mailbox returns 0 immediately

---

## Primitive 4: OctoZip — GPU-Native Fractal Compression

### What Exists
- Delta encode/decode GPU kernels (`emit_vm_delta_encode.flow`, `emit_vm_delta_decode.flow`)
- Fractal emitter (`emit_fractal.flow`) — Mandelbrot/Julia/Burning Ship iteration on GPU
- Dictionary lookup kernel (`vm_dict_lookup.spv`) — gather at arbitrary indices
- Q4_K dequantization kernel (`emit_dequant_q4k.flow`) — block dequant on GPU
- `.octo` column format with per-column codec selection (RAW, DELTA)

### What's New
A universal GPU-native compression engine that detects self-similarity and applies
fractal, delta, or direct-store encoding per block. Persistent `.octozip` cache
on disk. Both compression and decompression run as GPU kernels.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      OctoZip Engine                       │
│                                                           │
│  INPUT: raw data buffer (GPU memory, any type/size)      │
│                                                           │
│  STEP 1: Block Partition                                  │
│    Split input into fixed-size blocks (default 4096 floats)│
│    Each block analyzed independently (parallel)           │
│                                                           │
│  STEP 2: Similarity Analysis (GPU kernel)                │
│    Per block, compute:                                    │
│    ├── self_similarity: autocorrelation at multiple lags  │
│    ├── entropy: Shannon entropy of value distribution     │
│    ├── delta_ratio: compressibility via delta encoding    │
│    └── score → FRACTAL | DELTA | HOLDER decision         │
│                                                           │
│  STEP 3: Block Encoding (GPU kernels, per-type)          │
│    ├── FRACTAL [F]: IFS attractor fitting                │
│    │   Store: transform coefficients + iteration count    │
│    │   Ratio: 5-20x (self-similar data)                  │
│    │                                                      │
│    ├── DELTA [D]: Delta encoding (existing kernel)       │
│    │   Store: first value + deltas (variable-length)      │
│    │   Ratio: 2-5x (smooth/sequential data)              │
│    │                                                      │
│    └── HOLDER [H]: Direct store (passthrough)            │
│        Store: raw bytes, compacted alignment              │
│        Ratio: 1:1 (incompressible data)                  │
│                                                           │
│  STEP 4: Pack + Write                                     │
│    Concatenate block headers + encoded blocks             │
│    Write to .octozip file (persistent cache)              │
└──────────────────────────────────────────────────────────┘
```

### .octozip File Format

```
OCTOZIP FILE LAYOUT
┌─────────────────────────────────────┐
│ Magic: "OZIP" (4 bytes)            │
│ Version: u32                        │
│ Source hash: u64 (integrity check)  │
│ Total blocks: u32                   │
│ Uncompressed size: u64              │
│ Compressed size: u64                │
├─────────────────────────────────────┤
│ Block Directory (per block):        │
│   type: u8    (F=0, D=1, H=2)      │
│   offset: u32 (byte offset in file) │
│   raw_size: u32 (decompressed)      │
│   enc_size: u32 (encoded size)      │
│   checksum: u32 (CRC of raw block)  │
├─────────────────────────────────────┤
│ Block 0: [F] coefficients           │
│ Block 1: [D] deltas                 │
│ Block 2: [H] raw bytes              │
│ ...                                 │
└─────────────────────────────────────┘
```

### Fractal Block Encoding — IFS Attractor

For blocks with high self-similarity (e.g., neural network weight matrices
where columns follow similar distributions):

```
COMPRESSION (offline, GPU-accelerated):
1. Partition block into N sub-ranges (domain blocks)
2. For each sub-range, find best-matching larger range (range block)
   → This is the self-similar mapping: small ≈ transform(large)
3. Encode: affine transform coefficients (scale, offset) per mapping
4. Store: N × (range_idx: u16, scale: f16, offset: f16) = N × 6 bytes
   vs. original N × block_size × 4 bytes

DECOMPRESSION (GPU kernel, parallel):
1. Load coefficients from .octozip block
2. For each sub-range in parallel:
   a. Read range block (larger region)
   b. Apply affine transform: value = range[i] * scale + offset
   c. Iterate K times (fixed-point convergence)
3. Output: reconstructed block in GPU memory

K (iteration count) controls quality vs. speed:
  K=1: fast, approximate (good enough for INT8 weight blocks)
  K=3: balanced (good for FP16 weights)
  K=8: high fidelity (use for FP32 precision-sensitive data)
```

### Why Fractals Work on Neural Network Weights

Transformer models have structural self-similarity:
1. **Cross-layer:** All 80 layers have identical architecture (Q,K,V,O + gate,up,down)
2. **Cross-head:** Attention heads within a layer share statistical distribution
3. **Cross-position:** Weight columns for adjacent token positions are correlated
4. **Post-quantization:** Q4_K has only 16 values per block — extreme regularity

OctoZip detects these patterns automatically via the analysis pass. Blocks that
aren't self-similar get DELTA or HOLDER encoding — never forced into fractal.

### Persistent Cache — Amortized-Free Compression

```
FIRST RUN:
  model.gguf (40GB) → OctoZip analyze + compress → model.octozip (4-8GB)
  Time: minutes (one-time cost, GPU-accelerated analysis)

EVERY SUBSEQUENT RUN:
  model.octozip → decompress per-layer on GPU → full weights in VRAM
  Time: milliseconds per layer (parallel GPU iteration)

CACHE INVALIDATION:
  Source hash stored in header. If model.gguf changes → recompress.
  User can force: octozip_invalidate("model.octozip")
```

### Integration with Inference Pipeline

```
CURRENT (Phase 3):
  Disk → CPU dequant → GPU upload → compute
  Layer working set: ~500MB in VRAM per layer
  PCIe bottleneck: ~42ms per layer transfer (70B model)

WITH OCTOZIP (Phase 4):
  Disk (.octozip) → GPU decompress (Main Loom 1) → compute (Main Loom 2)
  Layer working set: ~50MB compressed + ~500MB decompressed = 550MB
  PCIe transfer: ~4ms (compressed), GPU decompress: ~2ms (parallel)
  Overlap: decompress layer N+1 while computing layer N
  Effective: ~5ms per layer (compute-bound, not memory-bound)

70B MODEL ON 6GB GPU:
  Active VRAM at any time:
    1 decompressed layer (attention + MLP):  ~800MB
    OctoZip stream buffer:                   ~200MB
    KV cache:                               ~1500MB
    Activations + mailboxes:                 ~500MB
    ─────────────────────────────────────────────
    Total:                                   ~3.0GB ← fits in 6GB
```

### API

```
// Compression (offline or first-run)
let handle = octozip_compress(data_array, block_size)     // GPU analysis + encode
octozip_save(handle, "output.octozip")                    // persist to disk

// Loading
let handle = octozip_load("model.octozip")                // load directory + block map
let valid = octozip_verify(handle)                        // check source hash

// Decompression (GPU, per-block)
let block = octozip_decompress(handle, block_index)       // decompress single block to GPU
let buf = octozip_decompress_to(handle, block_index, vm, offset) // decompress into VM buffer

// Streaming (pipeline mode)
let stream = octozip_stream(handle)                       // create streaming decompressor
let ready = octozip_next(stream)                          // decompress next block, non-blocking
let done = octozip_stream_done(stream)                    // 1.0 if all blocks decompressed

// Introspection
let ratio = octozip_ratio(handle)                         // compression ratio
let blocks = octozip_block_count(handle)                  // total blocks
let btype = octozip_block_type(handle, index)             // F=0, D=1, H=2
```

### OctoZip Kernel Emitters Needed

| Emitter | Purpose | GPU Pattern |
|---------|---------|-------------|
| `emit_octozip_analyze.flow` | Per-block self-similarity + entropy analysis | Parallel reduction: autocorrelation, histogram |
| `emit_octozip_fractal_enc.flow` | IFS attractor fitting (domain→range matching) | Parallel search: minimize MSE per sub-range |
| `emit_octozip_fractal_dec.flow` | IFS iteration (attractor → data reconstruction) | Embarrassingly parallel: iterate per element |
| `emit_octozip_delta_enc.flow` | Delta encoding (reuse existing) | Already exists: `emit_vm_delta_encode.flow` |
| `emit_octozip_delta_dec.flow` | Delta decoding (reuse existing) | Already exists: `emit_vm_delta_decode.flow` |
| `emit_octozip_pack.flow` | Pack encoded blocks + headers into output buffer | Sequential (single workgroup) |

### Edge Cases

| Edge Case | Risk | Mitigation |
|-----------|------|------------|
| Decompression divergence — fractal iteration produces NaN/Inf | Data corruption | Clamp after each iteration; checksum per block; abort + fallback to holder |
| Zero self-similarity — pure random data | Wasted analysis time | Early-exit: sample 1% of block; if entropy > 0.95 → holder immediately, skip full analysis |
| Cache staleness — .octozip doesn't match source | Wrong results | Source hash in header; verify on load; version field for format changes |
| Block size too small — overhead dominates | Poor ratio, slow | Minimum block size = 256 floats; auto-size: larger blocks for larger data |
| Block size too large — poor locality, high latency | Slow decompression | Maximum block size = 65536 floats; adaptive sizing based on VRAM budget |
| GPU OOM during decompress | Crash | Pre-check: `decompressed_size <= vram_available`; budget-aware decompressor |
| Precision loss stacking — fractal loss + quantization loss | Quality degradation | Per-block quality metric; if accumulated error > threshold → use HOLDER for that block |
| Concurrent decompress — multiple streams competing for GPU | Throughput collapse | Stream priority levels; inference stream always wins over background decompression |

### Test Plan
- `test_octozip_roundtrip` — compress → save → load → decompress → compare, verify bit-exact for holder blocks
- `test_octozip_ratio` — compress known self-similar data (repeated pattern), verify ratio > 5x
- `test_octozip_holder_fallback` — compress random data, verify all blocks are holders, ratio ~1.0
- `test_octozip_delta_reuse` — verify delta blocks use existing emit_vm_delta_encode kernel
- `test_octozip_cache_invalidation` — modify source, verify load detects hash mismatch
- `test_octozip_streaming` — stream decompress 100 blocks, verify in-order delivery
- `test_octozip_inference_integration` — compress GGUF layer weights, decompress on GPU, verify inference quality within tolerance

---

## Primitive 5: Multi-Stack Topologies

### What Exists
One Loom Engine stack: 1 Support Loom + N Main Looms. All VMs share one GPU device.

### What's New
Stacks of stacks. Three topologies for composing Loom Engine instances:

### Topology A: Sequential (Pipeline)

Output of one stack feeds input of the next. Like Unix pipes for GPU compute.

```
STACK A              STACK B              STACK C
┌────────────┐      ┌────────────┐      ┌────────────┐
│ Support    │      │ Support    │      │ Support    │
│ Main 1     │─────►│ Main 1     │─────►│ Main 1     │
│ (tokenize) │ mail │ (infer)    │ mail │ (sample)   │
└────────────┘      └────────────┘      └────────────┘
                     mailbox connects output → input
```

**Use case:** Streaming inference — tokenize → embed → transform → sample.
Each stage runs independently, connected by mailboxes. Natural backpressure:
if Stack B is slow, Stack A's mailbox fills up and try_send returns 0.

**Implementation:** Each stack is a standard Loom Engine instance. A **coordinator**
(top-level .flow code) manages the mailboxes and dispatch ordering.

### Topology B: Parallel (Independent)

Multiple stacks running simultaneously, no direct dependency, optional sync.

```
STACK A (Agent 1: Coder)     ─┐
STACK B (Agent 2: Reviewer)   ├──► SYNC MAILBOX ──► Merge Logic
STACK C (Agent 3: Tester)    ─┘
```

**Use case:** Swarm intelligence — N agents with different models or prompts,
each in an isolated stack, voting/consensus through mailboxes.

**Implementation:** Boot N stacks, dispatch all in round-robin, collect results
via mailboxes when all stacks signal completion.

### Topology C: Hierarchical (Nested)

A Main Loom's work is itself a full Loom Engine stack. The outer stack orchestrates,
the inner stack does heavy lifting.

```
OUTER STACK (Orchestrator)
├── Support Loom (outer)
├── Main Loom 0 ──────► INNER STACK (Inference Engine)
│                       ├── Support Loom (inner)
│                       ├── Main Loom 0 (decompress)
│                       └── Main Loom 1 (matmul)
└── Main Loom 1 (monitor/metrics)
```

**Use case:** Meta-agent that spawns specialized sub-agents. Outer stack manages
task allocation, inner stacks handle computation.

**Implementation:** Inner stack created by `loom_stack_new()`. Outer Support Loom
delegates work to inner stack's Support Loom via mailbox. Inner stack runs
autonomously, returns results via mailbox when done.

### Multi-Stack API

```
// Stack lifecycle
let stack = loom_stack_new()                    // create new Loom Engine stack
loom_stack_boot(stack, instances, regs, globals) // boot VMs within stack
loom_stack_shutdown(stack)                      // shutdown all VMs in stack

// Stack topology
loom_stack_chain(stack_a, stack_b)              // sequential: A's output → B's input
loom_stack_sync(stack_a, stack_b, mailbox)      // parallel: sync via shared mailbox

// Stack control
loom_stack_dispatch(stack, spv, params, wg)     // dispatch to stack's Main Loom
loom_stack_run(stack)                           // execute all pending dispatches
loom_stack_poll(stack)                          // check if stack is done

// Resource
loom_stack_vram(stack)                          // VRAM usage of this stack
loom_stack_vm_count(stack)                      // number of VMs in stack
```

### Resource Sharing Between Stacks

All stacks share one physical GPU. Resource contention is managed by:

1. **VRAM Budget:** Each stack has a VRAM allocation. Sum of all allocations ≤ 85% of total.
2. **Queue Scheduling:** All stacks submit through the same queue mutex. Round-robin by default,
   priority levels for latency-sensitive stacks (inference > background > analysis).
3. **Compute Unit Sharing:** GPU hardware scheduler handles this. Stacks with more dispatches
   naturally get more compute time.

### Edge Cases

| Edge Case | Risk | Mitigation |
|-----------|------|------------|
| VRAM fragmentation — many stacks alloc/free small VMs | OOM despite free space | Slab allocator: pre-allocate large region, sub-allocate per stack |
| Queue saturation — too many stacks submitting | Throughput collapse | Queue-aware scheduler: batch submissions per stack, round-robin across stacks |
| Hierarchical depth bomb — stack spawns stack spawns stack... | Resource exhaustion | Max nesting depth = 4 levels, enforced at `loom_stack_new()` |
| Parallel starvation — one stack hogs GPU | Other stacks starved | Per-stack dispatch quota per frame; homeostasis applies per-stack |
| Sequential deadlock — Stack B waits for A, A's mailbox waits for B's recv | Hang | Acyclic constraint: sequential chains must be DAG; cycle detection at `loom_stack_chain()` |
| Error propagation — inner stack fails, outer doesn't know | Silent failure | `loom_stack_poll()` returns error code; inner Support Loom writes status to mailbox |
| Shutdown ordering — parallel stacks with mailbox deps | Use-after-free | Dependency graph tracked by coordinator; topological shutdown order |
| Hot-swap — replacing a stack at runtime | State loss | Checkpoint: serialize stack state to OctoZip before swap; restore after |

### Test Plan
- `test_sequential_pipeline` — 3 stacks chained, data flows A→B→C correctly
- `test_parallel_independent` — 4 stacks running simultaneously, verify no interference
- `test_hierarchical_nest` — outer stack delegates to inner stack, results propagate back
- `test_depth_limit` — verify nesting beyond depth 4 is rejected
- `test_cross_stack_mailbox` — mailbox works between stacks (not just between VMs within a stack)
- `test_stack_shutdown_order` — verify clean shutdown of chained stacks in reverse order
- `test_stack_vram_budget` — verify total allocation across stacks respects GPU limit

---

## Primitive 6: Varying Bit Compute

### What Exists
All Loom Engine computation is FP32 (SPIR-V `OpTypeFloat 32`). The IR builder
emits 32-bit floats for all arithmetic. `float_to_bits()` / `bits_to_float()`
allow bit reinterpretation but not native reduced-precision arithmetic.

### What's New
Per-loom precision selection. Each Main Loom runs JIT-compiled kernels at the
optimal precision for its workload. Support Loom decides precision per-loom
based on workload analysis.

### Precision Tiers

| Tier | Bits | SPIR-V | Use Case | VRAM Savings |
|------|------|--------|----------|-------------|
| FP32 | 32 | OpTypeFloat 32 | Norms, variance, loss | Baseline |
| FP16 | 16 | OpTypeFloat 16 (requires capability) | Attention, softmax | 2x |
| INT8 | 8 | Pack 4 per u32, integer math via float | MLP, embeddings | 4x |
| INT4 | 4 | Pack 8 per u32, lookup table decode | Weight storage, gather | 8x |
| 1-bit | 1 | Pack 32 per u32, bitwise ops | Binary masks, sieve (proven) | 32x |

### Per-Loom Precision Assignment

```
SUPPORT LOOM (precision controller)
│
├── Loom 1: Attention
│   Precision: FP16
│   Reason: Softmax needs reasonable precision, but FP16 is sufficient
│   JIT: emit_matmul("attn", H, H, H, PRECISION_FP16)
│
├── Loom 2: MLP (gate + up + down)
│   Precision: INT8
│   Reason: ReLU/SiLU activation tolerates quantization
│   JIT: emit_matmul("mlp", H, FFN, H, PRECISION_INT8)
│
├── Loom 3: Norms (RMSNorm)
│   Precision: FP32
│   Reason: Variance calculation needs full precision
│   JIT: emit_rmsnorm("norm", H, PRECISION_FP32)
│
├── Loom 4: OctoZip Decompress
│   Precision: Mixed (coefficients FP16, iteration FP32)
│   Reason: Coefficients compact, iteration needs stability
│   JIT: emit_octozip_fractal_dec("dec", PRECISION_MIXED)
│
└── Loom 5: Embedding Gather
    Precision: INT4 (packed)
    Reason: Pure lookup, no arithmetic precision needed
    JIT: emit_gather("embed", VOCAB, H, PRECISION_INT4)
```

### Precision on the Prime Sieve Precedent

`sieve_gpu_v3.flow` already does 1-bit compute on 16 VMs:
- Each VM allocates a bitmap (8192 u32 words = 262,144 bits)
- Kernels use `OpBitwiseAnd`, `OpBitwiseOr`, `OpBitCount` in SPIR-V
- Each bit represents one number — 1-bit compute, 32x density vs FP32

This is the proof that varying-bit compute works in the Loom Engine. The sieve
just hardcodes it. Phase 4 makes it **adaptive** — Support Loom decides the
precision, JIT compiles the kernel, and the Main Loom runs it.

### Precision Selection Algorithm

```
fn select_precision(data_stats)
  // data_stats from analysis kernel: min, max, mean, variance, sparsity

  if data_stats.needs_variance        // norms, loss computation
    return PRECISION_FP32             // full precision required

  if data_stats.max - data_stats.min < 256.0 and data_stats.is_integer
    if data_stats.max < 16.0
      return PRECISION_INT4           // small integer range
    end
    return PRECISION_INT8             // byte-range integers

  if data_stats.variance < 0.01       // low variance, smooth data
    return PRECISION_FP16             // half precision sufficient

  return PRECISION_FP32               // default safe choice
end
```

### Accumulation Rule

Regardless of operand precision, **accumulation always uses FP32**:

```
// INT8 matmul with FP32 accumulation
let a_byte = ir_buf_load(block, 0, gid)     // load INT8 (packed in float)
let b_byte = ir_buf_load(block, 1, kid)     // load INT8
let prod = ir_fmul(block, a_byte, b_byte)   // multiply (result fits in float)
let sum = ir_fadd(block, accumulator, prod)  // accumulate in FP32 (no overflow)
```

This prevents overflow/underflow in reduction operations. Same policy as NVIDIA
Transformer Engine, but enforced by the JIT emitter, not hardware.

### Edge Cases

| Edge Case | Risk | Mitigation |
|-----------|------|------------|
| Precision too low — autodetect picks INT8 where FP16 needed | Quality degradation | Validation: compare JIT output vs FP32 on sample; escalate if MSE > threshold |
| Mixed precision across mailbox — Loom A sends FP16, Loom B expects INT8 | Data corruption | Mailbox header includes precision tag; recv validates match or converts |
| INT4 packing complexity — 8 values per u32 word | Slow decode | JIT emitter generates shift+mask sequences; amortized cost < 1 cycle per value |
| Hardware lacks FP16 — older GPUs | Kernel creation fails | Query `gpu_info()` at boot; fall back to FP32 emulation; warn user |
| Accumulation overflow — very long reductions at FP16 | Wrong results | FP32 accumulation rule; never accumulate at operand precision |
| Precision drift over time — iterative algorithms accumulate error | Gradual degradation | Periodic full-precision checkpoint (every N iterations, recompute at FP32) |

### Test Plan
- `test_varying_precision_select` — verify algorithm picks FP16/INT8/FP32 correctly for known distributions
- `test_fp32_accumulation` — verify INT8 matmul accumulates in FP32, no overflow on large matrices
- `test_int4_pack_unpack` — verify 8 values pack into u32 and unpack correctly
- `test_precision_across_mailbox` — send FP16 data through mailbox to INT8 loom, verify conversion
- `test_precision_fallback` — simulate missing FP16 capability, verify FP32 fallback

---

## The Loom Computer Model

The six primitives above aren't optimizations. Together they form an **operating
system for the GPU**. This reframes the entire Loom Engine architecture:

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     THE LOOM COMPUTER                            │
│                                                                  │
│  APPLICATION LAYER (.flow programs)                             │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ "Describe what you want. The Loom Computer handles how."   ││
│  └──────────────────────────┬─────────────────────────────────┘│
│                              │                                   │
│  GPU OPERATING SYSTEM (Support Loom)                            │
│  ┌──────────────────────────▼─────────────────────────────────┐│
│  │                                                             ││
│  │  SCHEDULER        MEMORY MGR      IPC         FILESYSTEM   ││
│  │  homeostasis      VRAM budget     mailbox     OctoZip      ││
│  │  reactive         park/unpark     ring buf    .octozip     ││
│  │  dispatch         slab alloc      signals     cache        ││
│  │                                                             ││
│  │  PROCESS TABLE (Main Looms)                                ││
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────────┐   ││
│  │  │ ML0 │ │ ML1 │ │ ML2 │ │ ML3 │ │ ML4 │ │ parked  │   ││
│  │  │run  │ │run  │ │run  │ │sleep│ │run  │ │ pool    │   ││
│  │  │FP16 │ │INT8 │ │FP32 │ │     │ │1-bit│ │ (reuse) │   ││
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────────┘   ││
│  │                                                             ││
│  │  JIT COMPILER: ir_new → ir_ops → ir_write_spv → dispatch  ││
│  │  DEVICE DRIVER: Vulkan dispatch + queue mutex              ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │ Vulkan commands                   │
│  GPU HARDWARE                ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  1408 CUDA cores, 6GB VRAM — "the actual computer"         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  CPU AS BIOS + RUNTIME SERVICES                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  BOOT SEQUENCE:                                             ││
│  │    main() → detect GPU → init Vulkan → create queue         ││
│  │    → init thread pool → load .flow → launch Support Loom    ││
│  │                                                              ││
│  │  THREAD POOL (post-boot runtime services):                  ││
│  │  ┌───────────┬───────────┬───────────┬───────────┐         ││
│  │  │ T0: Main  │ T1: I/O   │ T2: JIT   │ T3: Zip   │         ││
│  │  │ interpret │ disk+net  │ compile   │ analyze   │         ││
│  │  ├───────────┼───────────┼───────────┼───────────┤         ││
│  │  │ T4: Mail  │ T5: Input │ T6: Chkpt │ T7-11:   │         ││
│  │  │ mediate   │ keyboard  │ writer    │ workers   │         ││
│  │  └───────────┴───────────┴───────────┴───────────┘         ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

### OS Concept Mapping

Every operating system concept has a direct Loom Engine equivalent:

| OS Concept | Loom Equivalent | Status |
|-----------|----------------|--------|
| Process | Main Loom (VM) | Exists |
| Process spawn / fork | `loom_boot` / `loom_auto_spawn` | Exists + 4A |
| Process kill | `loom_shutdown` | Exists |
| Process sleep | `loom_park` | 4A-04 |
| Process wake | `loom_unpark` | 4A-04 |
| Process scheduler | Homeostasis + reactive dispatch | Exists + EC-7 |
| Virtual memory | OctoZip (compress/decompress on demand) | 4B |
| Physical memory | VRAM | Hardware |
| Memory manager | `loom_vram_budget` + slab allocator | 4A-05 |
| IPC pipe | Mailbox ring buffer | 4A-03 |
| Shared memory | `loom_copy` between VMs | Exists |
| Signals | Mailbox flag word / poll | 4A-03 |
| File system | OctoZip persistent cache (.octozip) | 4B-08 |
| Device driver | Vulkan dispatch + queue mutex | Exists |
| System calls | `loom_*` builtins | Exists |
| Dynamic linking | JIT kernel compilation (ir_write_spv) | Exists |
| BIOS | CPU boot sequence (init Vulkan, discover GPU) | Exists |
| BIOS runtime services | CPU thread pool (I/O, JIT, mediation) | 4A-06 |

### The CPU-GPU Contract

The CPU is not idle after boot. It provides **runtime services** — things the GPU
cannot do, or is inefficient at:

| CPU Service | Why CPU (Not GPU) | Thread |
|-------------|-------------------|--------|
| Scheduling decisions | Branch-heavy if/else, irregular logic | T0 (main) |
| Mailbox routing | Pointer chasing (which VM, which offset, how much) | T4 |
| JIT compilation | SPIR-V byte emission is sequential, can't run on GPU | T2 |
| OctoZip analysis | Self-similarity scan has irregular access patterns | T3 |
| File I/O | OS syscalls, filesystem — GPU has no disk access | T1 |
| Network I/O | TCP/UDP, HTTP — GPU has no network stack | T1 |
| User input | Win32 message pump, event dispatch | T5 |
| Prefix sum (small N) | For N < 64K, CPU is faster than GPU dispatch overhead | T0 |
| KV cache management | Token eviction = linked list / tree operations | T0 |
| Tree traversal | Fractal octree decisions (recursive, irregular) | T6+ |
| Checkpoint writes | Serialize + OctoZip compress + disk write | T6 |

### Impact: Parallel Support Loom

Without thread pool (current — single-threaded Support Loom):
```
Frame: [metrics 0.5ms] → [mail 1.0ms] → [JIT 0.1ms] → [schedule 0.1ms]
       → [dispatch 0.5ms] → [present 1.0ms]
Total Support overhead: ~3.2ms (sequential)
```

With thread pool (CPU multi-threading):
```
T0: [metrics] → [schedule] → [dispatch]        1.1ms (critical path)
T4: [mail send + 4×recv]                        0.4ms (overlapped)
T2: [JIT check/compile]                         0.1ms (overlapped)
T1: [OctoZip prefetch next layers]              0.0ms (background)
T5: [user input]                                0.0ms (background)

Total Support overhead: ~1.1ms (parallel, 3x reduction)
```

For the Quad-View showcase: from ~142 FPS to ~165+ FPS. The CPU clears the
runway so the GPU never stalls waiting for scheduling, I/O, or mediation.

---

## Primitive 7: Support Loom CPU Thread Pool

### What Exists
Phase 3O added ad-hoc background threads: async present (`JoinHandle` for framebuffer
download), SPIR-V prefetch (background file read), batched uploads (deferred staging).
These are individual threads for specific tasks — not a coordinated pool.

### What's New
A systematic thread pool that the Support Loom uses for all CPU-side operations.
Any `loom_*` builtin that involves CPU work can offload to the pool automatically.

### Design

```rust
struct LoomThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: crossbeam::channel::Sender<LoomTask>,
}

enum LoomTask {
    // I/O
    FileRead { path: String, result: oneshot::Sender<Vec<u8>> },
    FileWrite { path: String, data: Vec<u8>, done: oneshot::Sender<()> },

    // Mailbox mediation
    MailboxCopy {
        gpu: *const VulkanCompute,  // safe: VulkanCompute is pinned
        src_vm: u32, src_off: u32,
        dst_vm: u32, dst_off: u32,
        count: u32,
        done: oneshot::Sender<()>,
    },

    // JIT compilation
    JitCompile {
        emitter_source: String,  // .flow emitter code
        output_path: String,
        done: oneshot::Sender<Result<(), String>>,
    },

    // OctoZip
    OctoZipDecompress {
        handle: u32,
        block_index: u32,
        target_vm: u32,
        target_offset: u32,
        done: oneshot::Sender<()>,
    },

    // Checkpoint
    Checkpoint {
        vm_ids: Vec<u32>,
        output_path: String,
        done: oneshot::Sender<()>,
    },
}
```

### API

```
// Thread pool lifecycle
loom_threads(n)              // initialize pool with n worker threads (default: cpu_count - 1)
loom_cpu_count()             // query available CPU cores (0-arg builtin)

// Explicit async (optional — most builtins auto-offload)
let handle = loom_async(task_type, args...)  // submit task to pool
loom_await(handle)                           // wait for completion
loom_async_poll(handle)                      // check if done (non-blocking)

// Batch async (submit multiple, wait for all)
loom_await_all(handles_array)                // wait for all handles to complete
```

### Auto-Offload Builtins

These builtins automatically use the thread pool when available, with no .flow
code changes needed:

| Builtin | Without Pool | With Pool |
|---------|-------------|-----------|
| `loom_mail_send` | Main thread copies | Worker thread copies (overlapped) |
| `loom_mail_recv` | Main thread copies | Worker thread copies (overlapped) |
| `octozip_decompress` | Main thread decompresses | Worker thread decompresses |
| `ir_write_spv` | Main thread emits SPIR-V | Worker thread compiles |
| `gguf_prefetch_layer` | Already uses bg thread | Uses pool (shared workers) |
| `octozip_save` | Main thread writes disk | Worker thread writes |

**Backwards compatible:** If `loom_threads()` is never called, everything runs on
the main thread exactly as today. The pool is opt-in.

### Thread Safety

- All Vulkan submissions already go through `queue_mutex` (Phase 3O T-03)
- VM state access goes through thread-local `GPU_VMS` map — worker threads receive
  raw `VkBuffer` handles (Copy types, safe per Vulkan spec)
- OctoZip analysis runs on CPU data (no Vulkan objects)
- JIT compilation writes to filesystem (SPIR-V cache handles concurrent reads)

### Edge Cases

| Edge Case | Risk | Mitigation |
|-----------|------|------------|
| Thread pool exhaustion — all workers busy | Main thread blocks on loom_await | Pool size default = cpu_count - 1; main thread never submits to pool |
| Worker crashes — panic in thread | Pool degraded | Catch panic per-worker, respawn, log error |
| Over-subscription — more workers than CPU cores | Context switching overhead | Cap at cpu_count - 1; leave 1 core for OS + main thread |
| Vulkan object lifetime — worker holds VkBuffer after VM shutdown | Use-after-free | Workers hold u32 VM IDs, resolve to VkBuffer at execution time; check validity |
| Thread pool initialized mid-computation | Inconsistent state | Pool can only be initialized before first `loom_boot`; error if called after |
| Multiple pools | Resource waste | Enforce singleton — second `loom_threads()` call resizes existing pool |

### Test Plan
- `test_thread_pool_init` — verify `loom_threads(4)` creates 4 workers, `loom_cpu_count()` returns correct value
- `test_async_file_read` — verify background file read completes correctly
- `test_auto_offload_mailbox` — verify mailbox copies run on worker threads (measure main thread time reduction)
- `test_pool_saturation` — submit more tasks than workers, verify all complete (no deadlock)
- `test_pool_not_required` — verify all builtins work without pool (backwards compatibility)

---

## Implementation Roadmap

### Phase 4A: Foundation (Mailbox + IR Bitwise Ops + Thread Pool)

Everything depends on cross-loom communication, integer bit operations, and
CPU multi-threading for Support Loom operations.

| Task | Description | Depends On | Estimate |
|------|-------------|-----------|----------|
| 4A-01 | Add `ir_iand/ior/ixor/ishl/ishr/popcount` to IR builder | None | Small |
| 4A-02 | Implement mailbox VM + ring buffer layout | loom_copy (exists) | Medium |
| 4A-03 | Implement `loom_mail_send/recv/poll` builtins | 4A-02 | Medium |
| 4A-04 | Implement `loom_park/unpark` for VM reuse pool | None | Small |
| 4A-05 | Implement `loom_max_vms/vram_budget` resource controls | None | Small |

| 4A-06 | Support Loom CPU thread pool | None | Medium |

**Commit target:** 4 commits (IR ops, mailbox, resource controls, thread pool)

### Phase 4B: OctoZip Core

Compression engine with three block types.

| Task | Description | Depends On | Estimate |
|------|-------------|-----------|----------|
| 4B-01 | Define `.octozip` file format + reader/writer | None | Small |
| 4B-02 | Emit `octozip_analyze` kernel (entropy + autocorrelation) | 4A-01 (bitwise) | Medium |
| 4B-03 | Emit `octozip_fractal_enc` kernel (IFS attractor fitting) | 4B-02 | Hard |
| 4B-04 | Emit `octozip_fractal_dec` kernel (IFS iteration) | 4B-02 | Medium |
| 4B-05 | Integrate delta encode/decode (reuse existing kernels) | None | Small |
| 4B-06 | Implement holder (passthrough) blocks | None | Small |
| 4B-07 | Implement `octozip_compress/save/load/decompress` API | 4B-03, 4B-04, 4B-05, 4B-06 | Medium |
| 4B-08 | Persistent cache with source hash validation | 4B-07 | Small |

**Commit target:** 4 commits (format, analysis, encode/decode, API + cache)

### Phase 4C: OctoZip Streaming + Inference Integration

Connect OctoZip to the inference pipeline.

| Task | Description | Depends On | Estimate |
|------|-------------|-----------|----------|
| 4C-01 | Implement `octozip_stream/next/done` streaming API | 4B-07 | Medium |
| 4C-02 | Replace `gguf_prefetch_layer` with OctoZip stream path | 4C-01 | Medium |
| 4C-03 | Benchmark: 70B model on 6GB GPU, measure tok/s and quality | 4C-02 | Validation |
| 4C-04 | OctoZip CLI: `octoflow compress model.gguf` | 4B-07 | Small |

**Commit target:** 3 commits (streaming, inference integration, CLI tool)

### Phase 4D: Multi-Stack + Varying Bit

Compose stacks and assign precision per-loom.

| Task | Description | Depends On | Estimate |
|------|-------------|-----------|----------|
| 4D-01 | JIT precision-parameterized kernel templates | 4A-01 (bitwise) | Medium |
| 4D-02 | Precision analysis kernel (min/max/variance/entropy) | 4A-01 | Medium |
| 4D-03 | Implement `loom_stack_new/boot/shutdown/dispatch/run/poll` | 4A-02 (mailbox) | Hard |
| 4D-04 | Sequential topology: `loom_stack_chain` with mailbox wiring | 4D-03, 4A-03 | Medium |
| 4D-05 | Parallel topology: independent stacks with sync mailbox | 4D-03, 4A-03 | Medium |
| 4D-06 | Hierarchical topology: nested `loom_stack_new` within stack | 4D-03 | Hard |
| 4D-07 | Per-loom precision assignment via Support Loom | 4D-01, 4D-02 | Medium |
| 4D-08 | Stack nesting depth limit (max 4) | 4D-06 | Small |
| 4D-09 | VRAM budget enforcement across stacks | 4A-05 | Medium |

**Commit target:** 5 commits (JIT precision, stack core, sequential, parallel, hierarchical)

### Phase 4E: Swarm Intelligence Showcase

Application-layer demonstration of the full architecture.

| Task | Description | Depends On | Estimate |
|------|-------------|-----------|----------|
| 4E-01 | Recipe: single-agent inference with OctoZip | 4C-02, 4D-07 | Medium |
| 4E-02 | Recipe: 2-agent pipeline (decompress + infer) | 4D-04 | Medium |
| 4E-03 | Recipe: 4-agent swarm (parallel 7B models, consensus) | 4D-05 | Hard |
| 4E-04 | Showcase: 70B on 6GB GPU (streaming OctoZip + varying bit) | 4E-01 | Demo |
| 4E-05 | Showcase: swarm-of-agents collaborative task | 4E-03 | Demo |

**Commit target:** 3 commits (recipes, inference showcase, swarm showcase)

---

## Summary: What Makes This Unique

```
CUDA:    Single program, fixed precision, manual memory, no IPC, CPU is host
Metal:   Unified memory, no multi-VM, no JIT kernels from language
OpenCL:  Portable but verbose, no adaptive runtime, no compression
Loom:    GPU Operating System — adaptive JIT + on-demand processes (VMs)
         + mailbox IPC + fractal compression + multi-stack topologies
         + varying-bit compute + CPU thread pool for runtime services
         = self-optimizing GPU computer on consumer hardware
```

Seven primitives. The Loom Computer Model: CPU boots the system (BIOS), Support
Loom runs as the OS, Main Looms run as processes, GPU is the hardware. CPU
multi-threading provides runtime services the GPU can't do — I/O, scheduling,
JIT compilation, coordination.

No other engine treats the GPU as the primary computer with the CPU as its
bootstrap and service layer. Everyone else treats GPU as an accelerator that
the CPU drives. The Loom Engine inverts this.

**The benchmark that proves it:** 70B parameter LLM inference on a 6GB GPU,
at usable token rates, with acceptable quality. OctoZip + Varying Bit +
Multi-Stack + Mailbox + CPU Thread Pool, all orchestrated by the Support Loom OS.
One command. Zero configuration. Consumer hardware.

---

## Appendix: Dependency Graph

```
4A-01 (IR bitwise ops)
  ├── 4B-02 (OctoZip analysis kernel)
  │     ├── 4B-03 (fractal encode)
  │     │     └── 4B-07 (OctoZip API)
  │     │           ├── 4B-08 (persistent cache)
  │     │           ├── 4C-01 (streaming)
  │     │           │     ├── 4C-02 (inference integration)
  │     │           │     │     └── 4C-03 (benchmark)
  │     │           │     └── 4C-04 (CLI)
  │     │           └── 4E-01 (single-agent recipe)
  │     └── 4B-04 (fractal decode)
  │           └── 4B-07
  └── 4D-01 (JIT precision kernels)
        └── 4D-07 (per-loom precision)

4A-02 (mailbox VM)
  └── 4A-03 (mail_send/recv/poll)
        ├── 4D-04 (sequential topology)
        │     └── 4E-02 (2-agent pipeline recipe)
        ├── 4D-05 (parallel topology)
        │     └── 4E-03 (4-agent swarm recipe)
        └── 4D-06 (hierarchical topology)

4A-04 (park/unpark)
4A-05 (budget controls)
  └── 4D-09 (cross-stack budget)

4A-06 (CPU thread pool)
  ├── accelerates 4A-03 (mailbox auto-offload)
  ├── accelerates 4B-07 (OctoZip decompress offload)
  └── accelerates 4D-01 (JIT compile offload)

4B-05 (delta reuse) ── no deps
4B-06 (holder blocks) ── no deps
```

---

## Appendix: Emergent Capabilities from Primitive Composition

The six primitives are individually valuable. Stacked together, they unlock
capabilities that no individual primitive provides. Every capability below is
achievable with the Phase 4 primitives — no new fundamental architecture needed.

These are documented as patterns to implement, validate, and showcase once the
core primitives ship.

---

### EC-1: Fractal Spatial Indexing

**Stacks:** Mailbox + Multi-Stack (Hierarchical) + JIT Kernels

**Problem:** The N-Body showcase uses a flat spatial hash grid — 32K cells, fixed
resolution. But particle density is non-uniform. Dense clusters waste compute in
sparse regions; sparse regions waste memory in dense areas.

**Solution:** Replace flat grid with adaptive octree — a fractal data structure
where each node subdivides identically. Different Main Looms handle different
tree levels. Support Loom reads particle density per cell and decides which cells
to subdivide.

```
FLAT GRID (current):              FRACTAL OCTREE (adaptive):
┌──┬──┬──┬──┬──┬──┬──┬──┐       ┌─────────────┬──┬──┐
│  │  │  │  │  │  │  │  │       │             │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┤       │   empty     ├──┼──┤
│  │  │  │  │  │  │  │  │       │   (1 cell)  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┤       ├──┬──┬───────┼──┼──┤
│  │  │  │  │  │  │  │  │       │  │  │       │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┤       ├──┼──┤ medium├──┼──┤
│  │  │  │  │  │  │  │  │       │●●│●●│density │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┤       ├──┼──┤       ├──┼──┤
│  │  │  │  │  │  │  │  │       │●●│●●│       │  │  │
└──┴──┴──┴──┴──┴──┴──┴──┘       └──┴──┴───────┴──┴──┘
 64 cells, same cost everywhere    12 cells, resolution where needed
```

**Loom pattern:**
- Main Loom 0: Level 0 (coarsest, whole domain, few cells)
- Main Loom 1: Level 1 (subdivided regions that exceeded density threshold)
- Main Loom 2: Level 2 (fine detail, only the densest clusters)
- Support Loom: reads cell counts, decides subdivision, routes particles between levels
- Mailbox: boundary particles between levels

**Key insight:** Same kernel at every level — fractal self-similarity. JIT emitter
generates it once, parameterized by cell size. The tree depth adapts every frame
based on actual particle distribution.

**Applies to:** N-Body, fluid simulation, molecular dynamics, collision detection,
neural field rendering — any grid-based spatial computation.

**Impact:** 50-75% fewer cells computed in non-uniform distributions.

**Depends on:** Mailbox (4A-03), Multi-Stack Hierarchical (4D-06), JIT Kernels (4D-01)

---

### EC-2: Fractal Memory Layout (Space-Filling Curves)

**Stacks:** IR Bitwise Ops + JIT Kernels

**Problem:** GPU memory access patterns matter enormously. Threads reading 2D/3D
grid data in row-major order have poor cache locality for spatial neighbors.

**Solution:** Lay out data along Z-order (Morton) or Hilbert curves — fractals that
preserve spatial locality at every scale. Neighboring points in 2D/3D space map
to neighboring addresses in memory.

```
ROW-MAJOR:                     Z-ORDER (fractal):
 0  1  2  3  4  5  6  7        0  1  4  5 16 17 20 21
 8  9 10 11 12 13 14 15        2  3  6  7 18 19 22 23
16 17 18 19 20 21 22 23        8  9 12 13 24 25 28 29

Reading 2×2 block at (0,0):    Reading 2×2 block at (0,0):
  Indices: 0, 1, 8, 9            Indices: 0, 1, 2, 3
  Stride: 8 (cache miss)         Stride: 1 (perfect locality)
```

**Implementation:** Z-order encoding is pure bit interleaving — uses `ir_iand`,
`ir_ior`, `ir_ishl`, `ir_ishr` (Phase 4A-01 bitwise ops):

```
fn z_encode_2d(block, x, y)
  // Spread x bits: 0b abcd → 0b 0a0b0c0d
  let sx = ir_iand(block, ir_ior(block, ir_ishl(block, x, 8), x), 0x00FF00FF)
  let sx = ir_iand(block, ir_ior(block, ir_ishl(block, sx, 4), sx), 0x0F0F0F0F)
  let sx = ir_iand(block, ir_ior(block, ir_ishl(block, sx, 2), sx), 0x33333333)
  let sx = ir_iand(block, ir_ior(block, ir_ishl(block, sx, 1), sx), 0x55555555)
  // Same for y, then interleave
  let sy = ... // same spread sequence on y
  return ir_ior(block, sx, ir_ishl(block, sy, 1))
end
```

**Applicable kernels:** Ray tracing (pixel neighbors access similar geometry),
fluid simulation (cell neighbors interact), any 2D/3D grid computation.

**Impact:** 20-40% throughput improvement, zero algorithm change — just remap `gid`
through the space-filling curve at kernel entry.

**Depends on:** IR Bitwise Ops (4A-01)

---

### EC-3: Fractal Memoization (Coarse-to-Fine Computation)

**Stacks:** Multi-VM + Mailbox + JIT Kernels

**Problem:** Many computations at resolution N look similar to resolution N/2. Computing
the full resolution from scratch wastes GPU cycles where the coarse answer was adequate.

**Solution:** Multi-grid approach — compute coarse first (fast), upscale, then only
recompute the fine detail where the coarse answer was insufficient.

```
BRUTE FORCE:                       FRACTAL MEMOIZATION:
1024×1024 = 1,048,576 threads      256×256 = 65,536 threads (coarse)
                                   + upscale to 1024×1024
                                   + error check (sparse sample)
                                   + recompute ~200K high-error pixels
                                   = ~265K threads total (75% savings)
```

**Loom pattern:**
- Main Loom 0: Coarse pass (256×256, fast)
- Support Loom: reads coarse result, identifies high-error regions via sampling
- Main Loom 1: Fine pass (1024×1024, masked to only compute flagged regions)
- Mailbox: passes coarse result + error mask from Loom 0 to Support to Loom 1

**Applies to:** Fluid simulation (coarse timestep → refine turbulent regions),
ray tracing (coarse pixel → refine edges/detail), any iterative solver.

**Impact:** 50-75% fewer GPU dispatches for workloads with spatial coherence.

**Depends on:** Mailbox (4A-03), Multi-VM (exists)

---

### EC-4: Fractal Error Correction for OctoZip

**Stacks:** OctoZip + JIT Kernels

**Problem:** Lossy fractal compression introduces per-block error. Across 80
transformer layers with similar structure, the error pattern is often self-similar.

**Solution:** Measure the error on the first layer (compare compressed output vs
full-precision reference). If the error pattern is self-similar across layers,
store one error model and apply it as a correction to all subsequent layers.

```
Layer 1: full decode → compare → error = [+0.02, -0.01, +0.03, -0.02, ...]
Layer 2: fast decode + apply error correction → near-lossless
Layer 3: fast decode + apply error correction → near-lossless
...
Layer 79: fast decode + apply error correction → near-lossless
```

**Cost:** One extra layer decode at full precision (calibration). Error model is
itself fractal-compressible — costs almost nothing to store.

**Impact:** Push OctoZip from "acceptable lossy" to "near-lossless" at the same
compression ratio. Could reduce perplexity impact from ~0.5% to ~0.05%.

**Depends on:** OctoZip (4B-07)

---

### EC-5: Self-Modifying Compute Graphs

**Stacks:** JIT Kernels + Homeostasis Metrics + SPIR-V Cache

**Problem:** Today, compute graphs are fixed — the programmer chooses which kernel
to dispatch. The GPU may be running a dense matmul on data that's 90% zeros.

**Solution:** Support Loom reads intermediate results (metrics buffer) and JIT-compiles
optimized kernels mid-computation:

```
Frame 1:
  Support Loom dispatches generic matmul kernel
  Reads result → metrics show 87% sparsity

Frame 2:
  Support Loom calls emit_sparse_matmul("sparse.spv", sparsity=0.87)
  JIT compiles sparse kernel (skip zero blocks)
  Dispatches sparse kernel → 5x speedup
  SPIRV_FILE_CACHE stores it (zero cost on reuse)

Frame N: (sparsity changes)
  Support Loom re-analyzes → sparsity dropped to 30%
  Switches back to dense kernel (also cached)
```

**The compute graph evolves as it runs.** No human chose "sparse kernel" — the
Loom Engine discovered the optimization from the data.

**Implementation:** Pure .flow pattern. Support Loom reads `loom_read_metrics()`,
calls an emitter function, gets a new .spv, dispatches it. Everything needed exists.

**Applies to:** Any workload with varying data characteristics: sparse attention in LLMs,
variable particle density in physics, adaptive mesh refinement.

**Impact:** Automatic optimization without human analysis. The engine gets smarter
about the data as it runs.

**Depends on:** JIT Kernels (4D-01), Homeostasis metrics (exists), SPIR-V cache (exists)

---

### EC-6: Speculative Multi-Draft Decoding

**Stacks:** On-Demand Looms + Mailbox + Park/Unpark + Varying Bit

**Problem:** LLM inference is sequential — each token depends on the previous. GPU
is mostly idle waiting for the sequential dependency chain.

**Solution:** Spawn speculative looms that guess the next token and start computing
ahead. If the guess is right, the result is already available.

```
Token N generated: "The"

Support Loom spawns 4 speculation looms:
├── Loom A: assume "cat"  → compute token N+2 for "The cat"
├── Loom B: assume "dog"  → compute token N+2 for "The dog"
├── Loom C: assume "new"  → compute token N+2 for "The new"
└── Loom D: assume "old"  → compute token N+2 for "The old"

Main inference generates actual token N+1: "cat"

Loom A was right! Token N+2 is already computed.
Looms B, C, D: park for reuse (zero allocation cost next time)

Result: 2 tokens in the time of ~1.3 tokens
```

**Loom Engine advantages for speculative decoding:**
- `loom_auto_spawn`: spawn speculator VMs (or reuse parked ones)
- Mailbox: broadcast KV cache state to all speculators
- `loom_park`: discard wrong speculations without deallocation overhead
- Varying bit: speculators run at INT4 (draft quality), main runs at FP16 (full quality)

**Impact:** 2-4x inference throughput improvement. Well-studied technique (used in
vLLM, Medusa) but the Loom Engine makes it architecturally natural.

**Depends on:** On-Demand Looms (4A-04), Mailbox (4A-03), Varying Bit (4D-07)

---

### EC-7: Reactive Pipelines (Data-Driven Dispatch)

**Stacks:** Homeostasis Metrics + On-Demand Looms + JIT Kernels

**Problem:** Fixed compute pipelines waste GPU cycles. If no collisions occurred
this frame, the collision response kernel runs for nothing.

**Solution:** Support Loom reads intermediate metrics and makes dispatch decisions
per-frame:

```
Support Loom reads collision_count from metrics buffer:

  collision_count == 0:
    SKIP collision response kernel (save ~2ms)
    Dispatch rendering directly

  collision_count > 1000:
    Spawn extra Main Loom, split collision response across 2 VMs
    (reactive scale-up)

  collision_count < 10:
    JIT simple brute-force kernel (skip spatial hash overhead)
    Park the spatial hash loom this frame
```

**The pipeline adapts every frame.** Calm scene → minimal GPU work. Chaotic scene
→ more VMs, more precise kernels. Zero overhead in the common case.

**Applies to:** Game physics, simulation, interactive visualization — any workload
with variable per-frame intensity.

**Impact:** 30-60% GPU savings in variable-intensity workloads by skipping unnecessary work.

**Depends on:** Homeostasis metrics (exists), On-Demand Looms (4A-04), JIT (4D-01)

---

### EC-8: Checkpoint and Resume

**Stacks:** OctoZip + Loom State Serialization

**Problem:** Long-running scientific simulations lose all progress on crash or
power loss. Multi-hour GPU compute with no recovery option.

**Solution:** Periodically serialize all VM state via OctoZip, write to disk as
checkpoint. On recovery, decompress and restore.

```
Every N frames:
  Support Loom reads all VM buffers
  OctoZip compress (simulation state is highly self-similar across time)
  Write checkpoint.octozip to disk

On crash recovery:
  Load checkpoint.octozip
  OctoZip decompress → restore all VM buffers
  Resume simulation from last checkpoint
```

OctoZip makes this practical — 100MB of simulation state compresses to 10-20MB.
Checkpoint every 60 seconds costs < 0.5s overhead.

**Also enables live migration:** Serialize → OctoZip compress → transfer compressed
(10x less data) → decompress on target GPU → resume. Move computation between GPUs
without stopping.

**Impact:** Crash recovery for long simulations. Migration between GPUs/machines.

**Depends on:** OctoZip (4B-07), loom_read_globals (exists)

---

### EC-9: Ternary Compute (1.58-bit)

**Stacks:** Varying Bit + JIT Kernels

**Problem:** Even INT4 uses multiply instructions. For inference-only workloads,
recent research (BitNet) shows ternary weights {-1, 0, +1} maintain quality.

**Solution:** JIT emitter generates ternary matmul kernel. Multiplication by
{-1, 0, +1} is not multiplication — it's conditional add/subtract/skip:

```
weight = -1: output -= input    (subtract)
weight =  0: skip               (zero, no operation)
weight = +1: output += input    (add)
```

**No multiply instruction emitted.** Just `ir_fadd`, `ir_fsub`, and `ir_select`.

**Memory impact:**
```
FP16:    2 bytes/weight    70B model = 140 GB
INT8:    1 byte/weight     70B model = 70 GB
INT4:    0.5 bytes/weight  70B model = 35 GB
Ternary: 0.125 bytes/weight 70B model = 8.75 GB
  + OctoZip on ternary (extreme regularity): ~2 GB
```

A 70B ternary model with OctoZip fits **entirely in 6GB VRAM**. No streaming needed.
No layer-by-layer loading. Instant access to all weights. This eliminates the
memory bottleneck completely.

**Quality tradeoff:** Ternary loses some quality vs FP16. But combined with
adaptive precision (EC-10), critical layers can run FP16 while most run ternary.

**Depends on:** Varying Bit (4D-07), JIT Kernels (4D-01)

---

### EC-10: Adaptive Precision Per Token

**Stacks:** Varying Bit + Homeostasis Metrics + JIT Kernel Swap

**Problem:** Uniform precision wastes quality budget. Easy tokens (predictable next
word) don't need FP16. Hard tokens (ambiguous reasoning) suffer at INT8.

**Solution:** Support Loom monitors output entropy after each token and adjusts
inference precision dynamically:

```
Prompt processing (prefill):
  Known text, errors average out → INT8 → fast prefill

Easy generation (confident, low entropy):
  Next token is predictable → INT8 → fast generation

Hard generation (uncertain, high entropy):
  Softmax distribution is flat → UPGRADE to FP16 for this token
  → better quality decision on the ambiguous token

Return to easy:
  Entropy drops → DOWNGRADE to INT8 → fast again
```

**Implementation:** Support Loom reads softmax distribution from metrics buffer.
If entropy > threshold: swap inference loom to FP16 kernel (already cached by JIT).
If entropy drops: swap back to INT8.

**Result:** Better quality than uniform FP16 for some workloads — precision budget
is spent where it matters most, not wasted on trivial tokens.

**Depends on:** Varying Bit (4D-07), Homeostasis metrics (exists), JIT cache (exists)

---

### EC-11: Population-Based GPU Training

**Stacks:** Multi-VM Parallel + Varying Bit (1-bit/ternary) + Mailbox

**Problem:** Training neural networks requires backpropagation, gradient computation,
and massive frameworks (PyTorch, JAX). Not accessible to everyone.

**Solution:** With 1-bit/ternary weights, a neural network is tiny enough to run
many copies simultaneously. Use evolutionary optimization instead of gradients:

```
Population of 4 model variants (ternary 7B = 875MB each):
  Total VRAM: 4 × 875MB = 3.5 GB ← fits on 6GB GPU

Each generation:
  1. Evaluate all 4 on same prompt (parallel stacks, one Main Loom each)
  2. Rank by quality (Support Loom compares outputs via mailbox)
  3. Keep top 2, mutate (random bit flips) to create 2 new variants
  4. Repeat

No gradients. No backprop. No training framework.
Just: mutate → evaluate → select → repeat.
```

**Loom Engine advantages:**
- 4 Main Looms: one per model variant, parallel evaluation
- Mailbox: collect evaluation results from all looms
- Ternary weights: 1000 bit flips = 1000 random mutations per generation
- Park/unpark: swap population members in/out efficiently

**This is evolutionary optimization of neural networks on a consumer GPU.**
Not competitive with gradient training for large models, but viable for fine-tuning,
prompt optimization, and small model adaptation.

**Depends on:** Multi-VM (exists), Varying Bit (4D-07), Mailbox (4A-03)

---

### EC-12: Self-Optimizing Inference Engine

**Stacks:** ALL six primitives combined

This is the convergence — every Phase 4 primitive working together to deliver
zero-configuration inference on consumer hardware.

```
USER: octoflow chat model.gguf

LOOM ENGINE (fully automatic):

STAGE 1: ANALYZE (first run, cached forever)
  └── OctoZip scans model weights
      ├── Layers 0-20: high self-similarity → fractal compress (12:1)
      ├── Layers 21-60: moderate similarity → delta compress (4:1)
      └── Layers 61-79: low similarity → holder (1:1)
      Result: model.octozip written to disk

STAGE 2: PROFILE (first 10 tokens)
  └── Support Loom measures per-layer behavior
      ├── Attention layers: need FP16 (softmax stability)
      ├── MLP layers: INT8 sufficient (quality checked)
      ├── Norms: FP32 required (variance precision)
      └── Embeddings: INT4 lookup (no arithmetic needed)
      Result: JIT precision-optimal kernels, cached in SPIRV_FILE_CACHE

STAGE 3: CONFIGURE (automatic topology)
  └── Support Loom spawns optimal loom configuration
      ├── Main Loom 1: OctoZip decompression (streams layers from cache)
      ├── Main Loom 2: Attention (FP16 JIT kernels)
      ├── Main Loom 3: MLP (INT8 JIT kernels)
      └── Mailboxes: decompressor → attention → MLP → Support
      VRAM budget: ~3 GB for a 70B model

STAGE 4: INFER (every token, pipelined)
  └── Three-stage pipeline running continuously
      ├── Loom 1 decompresses layer N+1 (OctoZip → full weights)
      ├── Loom 2 computes attention on layer N (FP16)
      ├── Loom 3 computes MLP on layer N-1 (INT8)
      └── All three overlapped — pipeline always full

STAGE 5: ADAPT (continuous, every token)
  └── Support Loom monitors and reacts
      ├── High entropy? → upgrade attention to FP32 this token (EC-10)
      ├── Sparsity detected? → JIT sparse kernel (EC-5)
      ├── Long prompt? → spawn extra attention loom, split heads
      ├── Simple generation? → park extra looms, save power
      └── Homeostasis: auto-pace entire pipeline for thermal stability
```

**User experience:** Type one command. Get inference. The Loom Engine handles
compression, precision, topology, pipelining, and adaptation automatically.
70B model on a $200 GPU. No CUDA. No Python. No configuration.

**Depends on:** All Phase 4 primitives (4A through 4E)

---

### Emergent Capabilities Summary

| # | Capability | Primitives Stacked | New Code | Impact |
|---|-----------|-------------------|----------|--------|
| EC-1 | Fractal Spatial Indexing | Mailbox + Multi-Stack + JIT | Octree kernel emitter | 50-75% fewer cells in non-uniform distributions |
| EC-2 | Fractal Memory Layout | IR Bitwise Ops | Z-order GID remapper (~15 lines) | 20-40% throughput, zero algorithm change |
| EC-3 | Fractal Memoization | Multi-VM + Mailbox | Coarse/fine dispatch pattern | 50-75% fewer dispatches |
| EC-4 | Fractal Error Correction | OctoZip | Error model per-layer | Near-lossless at lossy ratios |
| EC-5 | Self-Modifying Compute | JIT + Metrics + Cache | .flow orchestration pattern | Auto-discovered optimizations |
| EC-6 | Speculative Decoding | On-Demand + Mailbox + Park + VaryBit | Multi-draft orchestration | 2-4x inference throughput |
| EC-7 | Reactive Pipelines | Metrics + On-Demand + JIT | Per-frame dispatch logic | 30-60% GPU savings in variable workloads |
| EC-8 | Checkpoint / Resume | OctoZip + State Serialization | Serialize + compress + write | Crash recovery, live migration |
| EC-9 | Ternary Compute | Varying Bit + JIT | Ternary matmul emitter | 0 multiplies, 70B in ~2GB with OctoZip |
| EC-10 | Adaptive Precision/Token | Varying Bit + Metrics + JIT Swap | Entropy monitoring | Better quality than uniform FP16 |
| EC-11 | Population-Based Training | Multi-VM + VaryBit + Mailbox | Evolutionary loop in .flow | Neural network optimization without gradients |
| EC-12 | Self-Optimizing Inference | ALL primitives | Full pipeline orchestration | Zero-config 70B on 6GB GPU |

Every capability is a **composition** of Phase 4 primitives stacked on Phase 3
foundations. No new fundamental architecture required. The power comes from
combining what's already proven.

---

## Appendix B: Extended Application Domains

Ten high-value application patterns enabled by the Loom Engine's primitive
composition. Organized by coverage (what we have vs what's new).

### Covered by Existing/Planned Primitives (recipes, not new architecture)

**B-1: Real-Time Algorithm Evolution Engines**
Evolve thousands of algorithm variants simultaneously at millisecond scale.
CPU threads mutate kernels/hyperparameters/memory layouts. GPU benchmarks all
variants in parallel. Support Loom selects winners, parks losers, mutates next
generation. *Covered by:* EC-11 (Population-Based Training) + Multi-Stack
Parallel + park/unpark. General-purpose mutation beyond neural nets is a .flow
recipe pattern.

**B-2: Brute Force at Previously Impossible Scale (Meta-Bruteforce)**
Run different search *strategies* in parallel, not just data-parallel copies of
one algorithm. Example: SAT solver — dispatch DPLL, CDCL, local search, random
walk simultaneously across 4 stacks. Take first to finish, park the rest.
*Covered by:* Multi-Stack Parallel topology + on-demand looms + park (discard
losers). Pure recipe pattern.

**B-3: Massive Agent Simulation**
10M+ economic agents, city-scale emergent behavior, synthetic market ecosystems.
GPU threads handle reactive agent rules (simple agents = one thread each). VMs
handle complex strategic agents. Mailbox aggregates agent decisions. Support Loom
updates world state. *Covered by:* Multi-VM + mailbox + N-Body patterns
(4096 particles already proven; GPU threads naturally scale to millions).
Domain-specific agent kernels needed, not new infrastructure.

**B-4: Real-Time Market Reflex Systems**
Simulate 1000 futures per second. GPU evaluates trade path trees, computes
correlated asset cascades. Support Loom feeds live market data, evaluates
strategies, routes orders. *Covered by:* SC-4 (Monte Carlo, 1M paths x 252
days) + time series stdlib (SMA/EMA/RSI/MACD) + web builtins for market data.
Making it live = .flow recipe + external API integration.

**B-5: Phase-Space Computation Engines**
Instead of computing one answer, compute the *shape* of solution space.
Parameterize programs across phase-space coordinates. GPU evaluates entire
parameter regions. Support Loom aggregates into topology. *Covered by:*
Multi-Stack Parallel (each VM = different parameter point) + mailbox (collect
results) + OctoUI (visualize landscape). Recipe pattern.

### Partially Covered (planned primitives close the gap)

**B-6: Live Transformer Layer Routing (Sparse Activation)**
Dynamic routing through 10,000 micro-layers. Only activated subgraphs execute.
Layers become dynamically composable compute units. Phase-space activated
inference. *Partially covered by:* EC-6 (Speculative Decoding), EC-7 (Reactive
Pipelines), EC-10 (Adaptive Precision/Token). *Gap:* Our architecture thinks in
whole layers (1 VM = 1 layer). Sub-layer micro-VM routing at 10K scale needs
Vulkan overhead analysis — may not be practical on consumer GPU, but the concept
of dynamic subgraph activation IS our EC-7 at coarser granularity.

**B-7: Microsecond Reinforcement Learning**
Millions of RL episodes per second on GPU. Evaluate reward, update policy, repeat.
*Partially covered by:* EC-11 (evolutionary optimization achieves similar goals
without gradients). Multi-VM parallel handles the episode evaluation side.
*Gap:* True RL (policy gradients, PPO, DQN) requires backpropagation kernels,
gradient accumulation, and optimizer state — IR emitters that don't exist yet.

### Not Covered (new architecture needed — Phase 5+)

**B-8: GPU-Native Query Engines (Fractal Databases)**
Every SQL clause becomes a GPU dispatch chain. WHERE = parallel filter,
GROUP BY = histogram + scatter, JOIN = parallel hash probe, ORDER BY = gpu_sort,
aggregates = reduction. Speculative query planning: generate N plans, dispatch all
on GPU, take fastest. *Gap:* We have GPU primitives (sort, reduce, matmul) but
no relational algebra layer, no query planner, no SQL parser. Significant new
subsystem. Individual query kernels could be IR emitters on existing infrastructure.

**B-9: Distributed GPU VM Mesh (Edge Compute Swarms)**
Multiple GPUs as a logical graph. Network dispatch. Decentralized AI mesh. Peer-to-
peer GPU supercomputer. *Gap:* Our Vulkan layer talks to one physical device.
Multi-GPU needs multi-device Vulkan. Multi-machine needs network protocols,
serialization, fault tolerance, consensus. OctoZip + mailbox + checkpoint/resume
give building blocks for migration, but a distributed runtime is Phase 6+ scope.

**B-10: Compiler Auto-Parallelism Discovery**
Compiler splits programs into micro-tasks, speculatively parallelizes everything,
benchmarks live on GPU, keeps fastest execution plan. *Gap:* Auto-parallelization
is a compiler research problem requiring dependency analysis, speculative
execution with rollback, and benchmark-driven plan selection. Beyond Phase 4 scope.

### Coverage Summary

| Domain | Coverage | Primitive Stack |
|--------|----------|----------------|
| Algorithm Evolution | 80% | EC-11 + Multi-Stack + park/unpark |
| Meta-Bruteforce | 90% | Multi-Stack Parallel + park |
| Agent Simulation | 70% | Multi-VM + mailbox + spatial hash |
| Market Reflex | 75% | SC-4 + time series + web builtins |
| Phase-Space Compute | 40% | Multi-Stack Parallel + mailbox |
| Sparse Layer Routing | 60% | EC-6 + EC-7 + EC-10 |
| Microsecond RL | 50% | EC-11 (evolutionary alternative) |
| GPU Query Engine | 20% | GPU primitives exist, no query layer |
| Distributed GPU Mesh | 5% | Single-GPU only |
| Auto-Parallelism | 30% | JIT + metrics, not auto-parallel |

**Key insight:** 7/10 domains are application patterns on existing primitives.
The Loom Engine's 7 primitives compose into far more capabilities than were
explicitly designed. Only 3 require genuinely new architecture (distributed GPU,
query engine, auto-parallelism) — all Phase 5+ candidates.
