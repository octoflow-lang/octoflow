# GPU VM Learnings: Seven Versions of Getting It Wrong

**What the sieve taught us about building a GPU compute VM.**

The GPU parallel sieve of Eratosthenes started at 70 seconds for pi(10^9) and
ended at 2.5 seconds — exact, past the uint32 wall, with zero CPU round-trips
during execution. Every speedup came from a bug, a crash, or a wrong assumption.

This document captures what we learned about GPU VM engineering: the bit-level
tricks, the precision landmines, the architectural patterns, and the surprising
results that apply to any domain running compute through OctoFlow's GPU VMs.

---

## Part 1: The f32 Precision Minefield

OctoFlow represents all values as `Value::Float(f32)`. f32 has a 24-bit
mantissa — exact integers only up to 2^24 = 16,777,216. Everything in GPU VM
engineering flows from this constraint.

### Bug Catalog: Eight Precision Bugs and Their Fixes

**Bug 1: Large uint32 constants round to wrong values**

```
ir_const_u(body, 4294967295.0)   // intended: 0xFFFFFFFF
// f32 rounds 4294967295 → 4294967296.0 = 2^32
// uint32(2^32) = 0  ← WRONG
```

Fix — now a standard pattern:
```
let c0 = ir_const_u(body, 0.0)
let all_ones = ir_not(body, c0)    // ~0 = 0xFFFFFFFF  ✓
```

**Rule: Any uint32 constant above 2^24 must be computed on-GPU via bitwise ops,
never passed as an f32 literal.** Common constants:
- `0xFFFFFFFF` = `ir_not(c0)`
- `0xFFFFFFFE` = `ir_not(c1)`
- `0x80000000` = `ir_shl(c1, c31)` where c31 = `ir_const_u(body, 31.0)`

---

**Bug 2: Push constant precision for computed values**

```
// CPU side: seg_start = seg_idx * 2000000 + 1
// Passed as f32 push constant
// At seg_idx = 9: 18000001.0 → f32 rounds to 18000000.0
// The +1 vanishes silently
```

Fix:
```
// Pass seg_idx (small, exact in f32) as push constant
let seg_idx = ir_push_const(entry, 0.0)    // small int, exact
let seg_u   = ir_ftou(entry, seg_idx)
let seg_start = ir_iadd(body, ir_imul(body, seg_u, seg_range), c1)
// GPU computes the full value in uint32 — exact
```

**Rule: Never pass a computed large number as a push constant. Pass the small
inputs and let the GPU compute.** Push constants are f32 — they lose precision
above 2^24. The GPU's uint32 ALU is exact.

---

**Bug 3: N encoding above 2^24**

N = 10^10 = 10,000,000,000 can't be represented exactly in f32 (or even in a
single uint32). How do you pass it to the GPU?

Fix — split into two f32-exact halves:
```
let SPLIT = 16777216.0              // 2^24
let N_HI = floor(N / SPLIT)        // fits in f32 exactly
let N_LO = N - N_HI * SPLIT        // fits in f32 exactly
// GPU reconstructs: N64 = N_hi × 2^24 + N_lo
let n_hi64 = ir_u32_to_u64(block, ir_ftou(block, pc_nhi))
let n_lo64 = ir_u32_to_u64(block, ir_ftou(block, pc_nlo))
let split64 = ir_u32_to_u64(block, ir_const_u(block, 16777216.0))
let n64     = ir_iadd64(block, ir_imul64(block, n_hi64, split64), n_lo64)
```

Exact for N up to 2^48 (~281 trillion). Each half fits in f32's 24-bit mantissa.

**Rule: Any value exceeding 2^24 that needs to reach the GPU should be split
into f32-exact components and reconstructed on-GPU.**

---

**Bug 4: Accumulation overflow in f32**

Summing per-VM prime counts that exceed 2^24 causes silent precision loss.
At pi(10^9) = 50,847,534, adding per-VM counts like 3,724,577 + 3,430,054 + ...
eventually crosses 2^24 where adjacent integers aren't representable.

Fix: Use `float_to_bits()` on readback to extract the uint32 accumulator value,
which was stored via `OpAtomicIAdd` (exact uint32 arithmetic on GPU). The GPU
accumulates in uint32 — the precision loss only happens if you treat the result
as f32.

**Rule: GPU-side uint32 arithmetic is always exact. The precision bugs happen
at the f32 boundary — push constants going in, readback values coming out.
Keep values in uint32 on the GPU as long as possible.**

---

**Bug 5: NaN canonicalization on readback**

Large uint32 values whose bit pattern has IEEE 754 exponent = 0xFF (e.g.,
`~255 = 0xFFFFFF00`) are interpreted as NaN by the GPU and canonicalized
to a standard NaN bit pattern, destroying the actual uint32 value.

This affects any uint32 in the range `0x7F800000`–`0x7FFFFFFF` (positive NaN)
or `0xFF800000`–`0xFFFFFFFF` (negative NaN).

Fix: Never read such values directly as f32. Use `float_to_bits()` which
bypasses f32 interpretation, or use indirect computation:
```
// Instead of reading NOT(x) directly:
let val = ir_buf_load_u(body, 0.0, idx)      // read uint32 raw
let inverted = ir_not(body, val)               // compute NOT on GPU
let count = ir_bitcount(body, inverted)        // use count, not raw bits
```

**Rule: If a GPU buffer contains uint32 values that overlap IEEE 754 NaN bit
patterns, never interpret them as f32. Read via `float_to_bits()` or compute
derived values on-GPU before readback.**

---

**Bug 6: uint32 overflow before narrowing**

In the v6 init kernel, `N - seg_base` can exceed 2^32 for segments in the
middle of the range. If narrowed directly from uint64 to uint32 via
`ir_u64_to_u32`, it truncates to garbage — the wrong bitmap bits get zeroed.

At N=10^10, exactly 2 segments crossed 2^32 boundaries, losing ~24,831 primes.

Fix: Clamp before narrowing:
```
// n_sub_base = N64 - seg_base64 (can be > 2^32)
// seg_range64 is known to fit in uint32
let clamp_flag = ir_ugte64(block, n_sub_base, seg_range64)
let clamped    = ir_select(block, IR_TYPE_UINT64, clamp_flag, seg_range64, n_sub_base)
let narrow     = ir_u64_to_u32(block, clamped)  // now safe
```

**Rule: Always clamp uint64 values to a known-safe range before narrowing to
uint32. The truncation is silent — no error, no crash, just wrong results in
specific segments that are hard to diagnose.**

---

**Bug 7: f32 sqrt and p*p comparison at scale**

CPU-side bucket sieve boundary: "which primes have multiples in this segment?"

Approach A: `floor(sqrt(seg_end)) + 1.0` — sqrt has f32 precision limits.
Approach B: `p * p > seg_end` — but f32 can't represent p*p exactly when p > 4096
(products above 2^24). At N=10^10, products near 10^10 have ULP ~1024.

Both approaches silently include or exclude primes at the boundary. v7 pre-sentinel
was off by +128 primes at pi(10^10) due to this.

Fix: The sentinel design (see Part 3) makes the boundary non-critical — a prime
dispatched one segment too early self-corrects on-GPU via exact uint64 arithmetic.

**Rule: Any CPU-side f32 comparison that gates GPU correctness is a ticking bomb
at scale. Design so that the GPU-side computation is authoritative and the CPU-side
comparison only affects performance (early/late dispatch), not correctness.**

---

**Bug 8: Round numbers mask precision bugs**

10^9, 2×10^9, 3×10^9, 4×10^9 are all exactly representable in f32. Tests at
these values pass. The precision bug only surfaces at N=4.3×10^9 (which rounds
to 4,300,000,256 in f32).

This is insidious: a test suite using only round numbers will see 100% pass
rate while the code has a latent precision bug.

**Rule: Test with adversarial N values — primes near 2^24, 2^32, values that
are NOT round numbers. If your tests only use 10^k, you will miss precision
bugs that appear between those values.**

---

### Precision Summary: The Three Boundaries

| Boundary | Threshold | Symptom | Fix Pattern |
|----------|-----------|---------|-------------|
| **f32 mantissa** | 2^24 = 16,777,216 | Silent rounding of integers | Compute on GPU in uint32 |
| **uint32 range** | 2^32 = 4,294,967,296 | Overflow → crash or wrap | uint64 for address math, uint32 for inner loops |
| **NaN bit pattern** | Exponent = 0xFF | Value destroyed on readback | `float_to_bits()` or indirect computation |

These three boundaries define the engineering space for any OctoFlow GPU VM
program that works with large integers or addresses.

---

## Part 2: Bit-Level GPU Tricks

### Bitmap Operations: The Foundation

The single biggest optimization in the sieve (48x GPU speedup, 1000x VRAM
reduction) was switching from f32-per-boolean to bit-packed uint32.

**Core operations:**

```
// Set bit at position `pos` in word at index `word_idx`
let one_shl = ir_shl(body, c1, bit_pos)      // 1 << pos
let mask    = ir_not(body, one_shl)            // ~(1 << pos)
let _       = ir_buf_atomic_and(body, 0.0, word_idx, mask)  // clear bit
```

**Hardware popcount for O(1) counting:**
```
let word = ir_buf_load_u(body, 0.0, gid)
let bits = ir_bitcount(body, word)             // hardware popcount
let _    = ir_atomic_iadd(body, count_off, bits)  // accumulate
```

**Boundary masking for partial words:**
```
// Last word in segment may have fewer than 32 valid bits
let valid_bits = ir_isub(body, total_cands, ir_imul(body, gid, c32))
let full_mask  = ir_not(body, c0)              // 0xFFFFFFFF
let partial    = ir_isub(body, ir_shl(body, c1, valid_bits), c1)  // (1<<n)-1
let use_full   = ir_ugte(body, valid_bits, c32)
let mask       = ir_select(body, IR_TYPE_UINT, use_full, full_mask, partial)
```

**Why atomic AND for bit-clearing:** Multiple threads may clear bits in the same
uint32 word concurrently. `OpAtomicAnd` is the only safe operation — read-modify-
write is atomic. Contention is low for large primes (~1-16 multiples per segment
per word range), but small primes need the word-centric approach (one thread owns
the whole word).

### Shared Memory Patterns

**Cooperative prime cache (small-prime marking):**
```
// Thread `local_id` loads one prime from global → shared memory
if local_id < num_small
    ir_shared_store(body, local_id, ir_load_input_at(body, 2.0, local_id))
end
ir_barrier(body)  // sync workgroup

// Now all 256 threads read primes from shared memory (fast)
while pi < num_small
    let prime = ir_shared_load(body, pi)
    // ... mark multiples ...
end
```

Impact: 256 threads × 53 primes = 13,568 global memory reads per workgroup
reduced to 53 global reads + 13,568 shared memory reads. Shared memory is
~10x faster than global (on-chip vs DRAM).

**Tree reduction for counting (shared memory):**
```
// Each thread writes its popcount to shared memory
ir_shared_store(body, local_id, my_count)
ir_barrier(body)

// Tree reduction: 8 rounds for 256 threads
// Round k: thread i adds slot [i + 2^k] if i < 2^(8-k)
let stride = 1
while stride < 256
    if local_id < (256 / stride / 2)
        let a = ir_shared_load(body, local_id * stride * 2)
        let b = ir_shared_load(body, local_id * stride * 2 + stride)
        ir_shared_store(body, local_id * stride * 2, ir_iadd(body, a, b))
    end
    ir_barrier(body)
    stride = stride * 2
end
// Thread 0 has the total → write to global accumulator
```

This replaces 8,192 global atomics with 8 barrier+reduce rounds and 1 global
atomic. At scale, the shared memory tree reduction is 256x fewer global atomic
operations.

### uint64 Arithmetic in SPIR-V

SPIR-V supports 64-bit integers via `OpCapability Int64`. The IR builder
provides 10 uint64 operations:

| IR Op | SPIR-V | Purpose |
|-------|--------|---------|
| `ir_u32_to_u64(block, val)` | OpUConvert | Widen for address math |
| `ir_u64_to_u32(block, val)` | OpUConvert | Narrow after computation |
| `ir_iadd64(block, a, b)` | OpIAdd (64-bit) | 64-bit addition |
| `ir_isub64(block, a, b)` | OpISub (64-bit) | 64-bit subtraction |
| `ir_imul64(block, a, b)` | OpIMul (64-bit) | 64-bit multiply |
| `ir_udiv64(block, a, b)` | OpUDiv (64-bit) | 64-bit unsigned divide |
| `ir_umod64(block, a, b)` | OpUMod (64-bit) | 64-bit unsigned modulo |
| `ir_ugte64(block, a, b)` | OpUGreaterThanEqual | 64-bit comparison |
| `ir_ulte64(block, a, b)` | OpULessThanEqual | 64-bit comparison |
| `ir_uequ64(block, a, b)` | OpIEqual | 64-bit equality |

All operations auto-enable `OpCapability Int64` in the SPIR-V header via
`ir_uses_uint64[0] = 1.0`.

**Design principle: minimal uint64 footprint.** Only the initial address
computation needs uint64. Inner loops stay in uint32 using segment-relative
offsets. At pi(10^9), the uint64 overhead is +20% GPU time — tolerable for
breaking the 4B wall but not free.

---

## Part 3: Architectural Patterns (Transferable to Any Domain)

### Pattern 1: Bit-Packing for Dense Boolean State

**What:** Pack 32 booleans per uint32 word + hardware popcount.

**Impact:** 48x GPU speedup, 1000x VRAM reduction (v1→v2).

**Why it works:** GPU memory hierarchy rewards compactness. A 32 KB working set
fits in L1 cache (4-cycle latency). A 2 GB working set hits DRAM (~400 cycles).
The savings multiply: less data to transfer, higher cache hit rate, more threads
share the same cache line.

**Transfer domains:**
- Bloom filters, bitmap indices, set membership queries
- Cellular automata (Game of Life — 32 cells per word)
- Binary image processing, feature presence masks in ML
- Graph adjacency matrices (dense boolean)
- Collision detection grids (occupied/empty cells)

---

### Pattern 2: L1-Sized Segmentation

**What:** Partition work so the hot working set fits in L1 cache (32-48 KB).

**Impact:** ~4x lower memory latency vs L2-sized segments.

**The numbers:** L1 latency ~4 cycles, L2 ~12-20 cycles, DRAM ~400 cycles.
For algorithms that touch the same working set repeatedly (sieve: each bitmap
word hit by multiple primes per segment), the latency difference compounds.

**Design detail:** Segment size is a push constant — tunable without recompiling
SPIR-V. In the sieve, `SEG_RANGE = NUM_WORDS × 64` where NUM_WORDS = 8192
gives exactly 32 KB.

**Transfer domains:**
- Tiled GPU GEMM (32 KB tiles fit in L1)
- Blocked convolution in signal/image processing
- Chunk-based merge sort (intermediate buffers in L1)
- Sliding window analytics (time series)
- Any iterative algorithm over a fixed working set

---

### Pattern 3: Bucket Dispatch (Work-Proportional Scheduling)

**What:** Monotonic work pointer. Per segment, dispatch only work items relevant
to that segment's range. The pointer never rewinds.

**Impact:** 7.4x fewer dispatches at pi(10^9). Segment 1 dispatches ~128 primes
instead of all 3,400.

**The code pattern:**
```
let mut work_ptr = start
for each segment:
    let seg_max = segment_boundary(seg_idx)
    while work_ptr < total_items:
        if work_items[work_ptr] > seg_max: break
        work_ptr += 1
    // dispatch work_items[start..work_ptr] for this segment
```

**Why it scales:** The benefit grows with N. At pi(10^9), only ~4% of primes are
relevant in the first segment. Without bucket dispatch, 100% are dispatched.

**Transfer domains:**
- N-body simulation (near-field interactions per spatial tile)
- Collision detection (broadphase: only check nearby object pairs)
- Event simulation (process events in current time window only)
- Database predicate pushdown per partition
- Stream joins (overlapping time windows)

---

### Pattern 4: Carry-Forward State

**What:** Store resume state in GPU registers (SSBO). Next invocation reads
directly — no recomputation.

**Impact at pi(10^9):** Eliminates ~15 uint64 ops per prime per segment. But
inner loop dominates, so net GPU speedup is within noise (+2%). Value grows
at extreme N where setup cost dominates.

**The surprising result:** At pi(10^9), large primes average ~4 bit-clears per
segment. Setup (15 ops) is ~27% of per-prime work. Not enough to move the
needle on total GPU time because the atomic AND operations (bit-clearing) have
~100-cycle latency while the arithmetic setup is ~4 cycles each. **The cheap
ops get saved; the expensive ops remain.**

**When carry-forward pays off:** When setup cost is proportionally large. At
N=10^12, large primes would have ~0.5 bit-clears per segment, making setup 30x
the marking work. Carry-forward would be transformative there.

**Lesson: Measure the ratio of setup cost to inner-loop cost before implementing
carry-forward. If the inner loop is dominated by high-latency operations
(atomics, memory), saving low-latency arithmetic setup won't be visible.**

**Transfer domains:**
- Streaming aggregation (running sums across chunks)
- Incremental graph BFS (frontier carried across waves)
- Time series sliding windows (carry partial aggregates)
- Iterative solvers (resume from last convergence point)
- Event sourcing (carry projection state, replay delta only)

---

### Pattern 5: Hybrid Kernel Strategy

**What:** Split work at a data-dependent threshold. Dispatch different,
specialized kernels for each category.

**Impact:** v4's unified kernel was 60x slower than v3/v5's hybrid split.

**The disaster that proved the point (v4):** One kernel, one thread per prime.
Small primes (p=3) loop 87,000 times per segment. Large primes (p=31623) loop
once. Same warp, same instruction pointer — the warp stalls on thread p=3 while
thread p=31623 idles for 86,999 iterations.

GPU threads execute in warps (32 threads in lockstep). Thread divergence isn't a
minor penalty — the warp throughput drops to 1/max_iterations of best case.

**The threshold:** p = 256. Below: word-centric (one thread per bitmap word, all
small primes). Above: prime-centric (one thread per prime, few iterations).

**Rule: If work items in the same dispatch have >10x different iteration counts,
split them into separate kernels.** This isn't premature optimization — it's the
difference between 765ms and 45,736ms.

**Transfer domains:**
- NLP batch processing (short vs long sequences — pad-and-mask wastes 90%+ compute)
- Sparse matrix ops (dense rows vs sparse rows)
- Particle simulation (near-field O(N) vs far-field O(1))
- Ray tracing (coherent primary rays vs scattered secondary rays)
- Graph processing (high-degree hubs vs low-degree nodes)

---

### Pattern 6: Selective JIT vs Pre-Compiled SPIR-V

**What:** Pre-compile static kernels to `.spv` files. Reserve `vm_dispatch_mem`
for kernels whose structure (not just parameters) changes at runtime.

**Impact:** v5's single JIT kernel costs 621ms. If all 5 kernels were JIT, startup
would be ~3 seconds — comparable to the sieve time itself.

**When JIT is justified:**
- Kernel structure depends on data discovered at runtime
- Number of loop iterations baked into the kernel
- Specialization for specific data shapes

**When JIT is NOT justified:**
- Kernel parameterized by push constants (segment index, array bounds, thresholds)
- These change per-dispatch but the kernel code is identical

**Transfer domains:**
- Database queries: JIT only complex filter UDFs, not standard scan/join
- ML inference: JIT only dynamic shapes, not fixed-architecture layers
- Scientific computing: JIT only user-defined functions in solvers

---

### Pattern 7: Async VM Swarm + Zero-Roundtrip Execution

**What:** Boot N independent VMs. Record complete dispatch chains. Build command
buffers. Launch all async. Poll completion.

**Impact:** GPU stays saturated — while one VM waits for a pipeline barrier,
another computes. At pi(10^10): 95,370 dispatches executed with zero CPU
intervention across 16 VMs.

**The structural advantage over CUDA:** CUDA kernel launches require a CPU-GPU
round-trip (~5-20μs each). OctoFlow pre-records the entire chain into a Vulkan
command buffer and submits once. At 95,370 dispatches, CUDA would need ~95K
round-trips. OctoFlow needs 16 (one per VM).

At extreme N (10^12+), this advantage compounds: CUDASieve's total time is
dominated by dispatch overhead, while OctoFlow's GPU time scales linearly with
work and dispatch overhead is fixed.

**Transfer domains:**
- Multi-query database execution (independent query plans)
- Ensemble ML inference (multiple models simultaneously)
- Monte Carlo simulation (independent random walks)
- Multi-channel signal processing (independent filter banks)

---

### Pattern 8: The Sentinel (Self-Initializing State)

**What:** Write a special marker value (sentinel) to state slots. The compute
kernel detects the sentinel and self-initializes on first encounter. All
subsequent invocations use the carried-forward state.

**Why it was needed:** The original v7 plan computed initial offsets in a separate
uint64 kernel. But this required knowing exactly when each prime activates —
a computation that depends on f32 sqrt/p*p comparisons that lose precision at
scale. Three different approaches to computing the activation boundary all
produced wrong results at N=10^10.

**The sentinel insight:** Don't try to predict activation from the CPU side.
Let the GPU kernel be authoritative:

```
// Init: write sentinel to all carry-forward slots
let sentinel = ir_not(body, c0)    // 0xFFFFFFFF
ir_buf_store_u(body, 0.0, carry_idx, sentinel)

// Mark kernel: check sentinel
let carry = ir_buf_load_u(pre_loop, 0.0, carry_idx)
let is_sentinel = ir_uequ(pre_loop, carry, sentinel_val)
// If sentinel → compute from scratch (uint64, exact)
// If not sentinel → use carry value (uint32, fast)
```

**Self-correcting property:** If the bucket sieve dispatches a prime one segment
too early (due to f32 imprecision), the sentinel triggers the uint64 computation,
which finds `half_off >= max_bits` → zero bit-clears happen → a valid
carry-forward residual is stored. The system corrects itself.

**General principle: When initialization depends on a precision-sensitive
computation, make the compute kernel self-initializing. The kernel runs in
exact integer arithmetic on the GPU. The CPU-side heuristic only affects
WHEN the kernel activates, not WHETHER the computation is correct.**

**Transfer domains:**
- Lazy initialization in streaming systems (first-touch activation)
- Sparse array operations (sentinel = uninitialized, compute on first access)
- Graph algorithms with dynamic frontier (node activates on first visit)
- Cache warming patterns (sentinel = cold, populate on first hit)
- Any system where initialization conditions are approximate but execution
  must be exact

---

## Part 4: What Surprised Us

### 1. The biggest optimization was data representation, not algorithms

v1→v2: Bit-packing gave 48x GPU speedup and 1000x VRAM reduction. Every
subsequent algorithmic improvement (bucket sieve, carry-forward, hybrid kernels)
combined for another ~2x. The data layout dwarfed everything else.

**Takeaway:** Before optimizing algorithms, audit your data representation.
If you're storing booleans as floats, integers as floats, or sparse data in
dense arrays — fix that first. Everything else is noise until the data fits
in cache.

### 2. Optimizations have crossover points

v3 (L1 segmentation + bucket sieve) is 1.8x faster than v2 at pi(10^9) but
5x slower at pi(10^7). More segments = more dispatch overhead. The bucket sieve
only helps at large N where early segments skip most primes.

**Takeaway:** Always benchmark at both small and large scale. An optimization
designed for 10^9 may be a regression at 10^7. The crossover point is part of
the design — document it.

### 3. Architectural elegance can be a 60x performance trap

v4's unified kernel was clean, simple, one code path for all primes. It was
60x slower than the "messy" hybrid split because GPU warps enforce lockstep
execution — one slow thread stalls 31 others.

**Takeaway:** On GPUs, heterogeneous workloads in the same dispatch cause thread
divergence that multiplies your worst-case cost across the entire warp. Split
aggressively at natural boundaries.

### 4. Speeding up one layer exposes the next bottleneck

v1: 92% GPU-bound. After v2's 48x GPU speedup, the system became 51% CPU-bound
(interpreter recording dispatches). The GPU was now fast enough that the OctoFlow
interpreter was the bottleneck.

At pi(10^10) in v7, recording 95,370 dispatches takes ~11 seconds of the 22-second
total. The GPU executes in 7.8 seconds. The CPU side is now the majority cost.

**Takeaway:** Every speedup shifts the bottleneck. Profile after every major
optimization — the limiter is probably somewhere you stopped looking.

### 5. Carry-forward saved cheap ops, not expensive ones

v7 eliminates ~15 uint64 arithmetic ops per prime per segment. Expected: 30-40%
GPU speedup. Actual: within noise (+2%).

Why: The eliminated ops are integer arithmetic (~4 cycles each). The remaining
ops are `OpAtomicAnd` (~100 cycles each). Carry-forward saved 60 cycles of
arithmetic but left 400+ cycles of atomics untouched.

**Takeaway:** Measure the latency profile of what you're optimizing vs what
remains. Saving low-latency ops doesn't help when high-latency ops dominate.
This is Amdahl's Law applied to instruction-level parallelism.

### 6. Round test values mask precision bugs

10^9, 2×10^9, 3×10^9 are all exactly representable in f32. Tests passed at
every round number. The bug only appeared at 4.3×10^9 (which rounds to
4,300,000,256 in f32).

**Takeaway:** Adversarial test values are mandatory. Include values near 2^24,
near 2^32, and values that are NOT multiples of large powers of 2. If all your
test inputs are round numbers, you have a false sense of correctness.

### 7. The batch dispatch model has a structural advantage at scale

OctoFlow executes 95,370 Vulkan dispatches with zero CPU-GPU round-trips.
CUDA would need ~95K round-trips at ~5-20μs each. At pi(10^12), the dispatch
count would be ~1M — OctoFlow's pre-recorded command buffer architecture would
outperform round-trip-based systems on dispatch overhead alone.

**Takeaway:** Pre-recorded batch dispatch chains amortize submission overhead
over arbitrary dispatch counts. This is the fundamental architectural bet of the
GPU VM: sacrifice flexibility (no CPU-side branching mid-chain) for throughput
(zero-roundtrip execution at any scale).

---

## Part 5: The Optimization Sequence That Worked

| Step | Version | What Changed | GPU Speedup | Principle |
|------|---------|-------------|-------------|-----------|
| 1 | v1→v2 | Bit-pack booleans | **48x** | Data representation > algorithms |
| 2 | v2→v3 | L1 segments + bucket sieve | **1.8x** | Cache locality + work-proportional dispatch |
| 3 | v3→v4 | Unified carry-forward | **0.017x** (60x slower) | Don't unify heterogeneous workloads |
| 4 | v4→v5 | Restore hybrid + selective JIT | **60x** (back to v3 level) | Hybrid split is mandatory |
| 5 | v5→v6 | uint64 addressing | **0.83x** (+20% overhead) | Break scaling walls; minimal uint64 footprint |
| 6 | v6→v7 | Sentinel carry-forward | **~1x** (parity) | Self-initializing state for correctness |

Total: **86x GPU speedup** from v1 to v7 (64,994ms → 792ms at pi(10^9)).

The lesson: **step 1 was 48x. Steps 2-6 combined were 1.8x.** The data
representation decision dominated everything. After that, each step was about
removing a specific class of waste (dispatch overhead, thread divergence,
scaling walls, precision bugs) — important but incremental compared to the
foundational choice.

### The Mantra

> **Compact data. Cache-sized segments. Dispatch only what's relevant.
> Split heterogeneous work. Let the GPU be authoritative.
> Measure what you think you're saving — it might not be the bottleneck.**

---

## Quick Reference: When to Apply Each Pattern

| Situation | Pattern | Expected Impact |
|-----------|---------|-----------------|
| Storing booleans/flags in f32 arrays | Bit-packing (#1) | 10-50x GPU, 32x memory |
| Working set > 48 KB per workgroup | L1 segmentation (#2) | 2-4x from cache locality |
| Dispatching all work to all segments | Bucket dispatch (#3) | 2-10x fewer dispatches |
| Recomputing startup state per segment | Carry-forward (#4) | 0-30% (depends on ratio) |
| Work items with >10x iteration variance | Hybrid kernels (#5) | 2-60x from reduced divergence |
| JIT-compiling parameterized kernels | Selective JIT (#6) | Save 100-600ms startup |
| Sequential VM execution | Async VM swarm (#7) | Near-linear GPU utilization |
| Init depends on imprecise CPU computation | Sentinel init (#8) | Correctness at scale |
| Passing large numbers as push constants | N-split encoding | Correctness above 2^24 |
| Reading uint32 with NaN bit patterns | `float_to_bits()` | Correctness on readback |
