# Annex U: Mersenne Prime Trial Factoring — GPU VM Swarm

**Status: Viable project. Research complete, architecture mapped to existing infrastructure.**

OctoFlow's GPU VM swarm can contribute to the Great Internet Mersenne Prime
Search (GIMPS) by performing trial factoring (TF) — the one GIMPS work type
that maps directly onto our proven sieve + parallel dispatch architecture.

---

## Why This Project

### For OctoFlow

Trial factoring is the first real-world distributed computing workload for
OctoFlow. It validates the GPU VM beyond benchmarks — producing results that
are verified by GIMPS/PrimeNet infrastructure against a global network of
participants. A correct TF result from OctoFlow proves the language can do
useful scientific computation.

### For GIMPS

GIMPS has a multi-year backlog of double-checks and trial factoring work.
Every GPU that contributes TF results saves expensive PRP primality tests
(hours/days per exponent) by finding factors cheaply (minutes per assignment).
New participants are welcome at any scale.

### For the Sieve Heritage

The sieve v1-v7 journey built exactly the infrastructure TF needs:
- Bit-packed sieve with hardware popcount (Pattern 1)
- L1-sized segmentation (Pattern 2)
- Batch dispatch chains with zero CPU round-trips (Pattern 7)
- Async VM swarm across 16+ VMs (Pattern 7)
- uint64 SPIR-V arithmetic (v6/v7)
- Sentinel-based lazy initialization (v7)

Trial factoring adds one new primitive — modular exponentiation — on top of
everything we already have.

---

## The Mathematics

### What is a Mersenne Prime?

A Mersenne number is 2^p - 1 where p is prime. If 2^p - 1 is also prime, it's
a Mersenne prime. As of February 2026, 52 Mersenne primes are known. The
largest, M136279841 (41 million digits), was discovered in October 2024 by
Luke Durant using thousands of NVIDIA GPUs — the first Mersenne prime found
by GPU.

### Trial Factoring: The Idea

Before running an expensive primality test (days of compute), try to find a
small factor. If 2^p - 1 has a factor, it's composite — no primality test
needed. Trial factoring is the cheapest way to eliminate candidates.

**Key theorem:** Any factor q of 2^p - 1 must satisfy:
1. q = 2kp + 1 for some integer k
2. q must be 1 or 7 (mod 8)

This dramatically restricts the search space. Instead of testing all numbers
up to some bound, we only test numbers of the form 2kp + 1 that pass the
mod-8 filter.

### The Two-Phase Algorithm

**Phase 1: Sieve (eliminate candidates divisible by small primes)**

Generate candidates q = 2kp + 1 for k = 1, 2, 3, ...
For each small prime s (up to ~40,000):
  - Eliminate candidates where q mod s = 0

This is a modified sieve of Eratosthenes. Each bit represents a candidate k
value. The sieve clears bits where 2kp + 1 is divisible by any small prime.
After sieving, ~5% of candidates survive.

**Phase 2: Modular exponentiation (test survivors)**

For each surviving candidate q:
  - Compute 2^p mod q using binary square-and-multiply
  - If result = 1, then q divides 2^p - 1 (factor found!)

For p ~ 140,000,000 (current GIMPS wavefront), the binary method requires
~27 iterations (log2 of p). Each iteration is one modular squaring plus a
conditional multiply-by-2. That's ~27 modular multiplications per candidate.

### Bit Levels

GIMPS assigns TF work in "bit levels" — the bit-length of the factor being
tested. Current assignments are typically 73-80 bits. This means q values
range from 2^72 to 2^80.

| Bit Level | Factor Range | Arithmetic Needed | Status in GIMPS |
|-----------|-------------|-------------------|-----------------|
| 60-63 | 2^59 to 2^63 | Pure uint64 | Long completed |
| 64-72 | 2^63 to 2^72 | uint64 + overflow handling | Mostly completed |
| **73-76** | **2^72 to 2^76** | **Barrett76 (fastest mfaktc kernel)** | **Active wavefront** |
| 77-79 | 2^76 to 2^79 | Barrett87 | Active |
| 80+ | 2^79+ | Multi-word | Frontier |

---

## Architecture: Mapping TF to OctoFlow GPU VM

### Overview

```
Phase 1: Sieve Kernel (GPU)          Phase 2: PowMod Kernel (GPU)
+--------------------------+         +---------------------------+
| Bit-packed bitmap        |         | For each surviving k:     |
| Each bit = one k value   |  --->   |   q = 2*k*p + 1           |
| Clear bits where 2kp+1   |         |   result = powmod(2,p,q)  |
| divisible by small prime  |         |   if result == 1: FOUND   |
+--------------------------+         +---------------------------+

           VM Swarm: 16 VMs, each handles a range of k values
           Entire pipeline pre-recorded as batch dispatch chain
           Zero CPU round-trips during execution
```

### Kernel Lineup (5 kernels per segment)

| # | Kernel | Purpose | Reuse |
|---|--------|---------|-------|
| 1 | tf_sieve_init | Set all bits in bitmap (L1-sized segment) | Adapt sieve_init_v6 |
| 2 | tf_sieve_mark | Clear bits for candidates divisible by small primes | Adapt sieve_mark_v6_small |
| 3 | tf_compact | Extract surviving k values from bitmap to array | New (scan + scatter) |
| 4 | tf_powmod | Modular exponentiation: 2^p mod (2kp+1) | New (core algorithm) |
| 5 | tf_check | Compare powmod results to 1, flag factors | New (trivial) |

### Memory Layout (B0 per VM)

```
B0[0 .. NUM_WORDS-1]               = sieve bitmap (32KB, L1-sized)
B0[NUM_WORDS]                       = survivor count
B0[NUM_WORDS+1 .. +1+MAX_SURV]     = survivor k-values (compacted)
B0[RESULT_BASE .. +MAX_SURV]       = powmod results
B0[FACTOR_OFF]                      = factor found flag (0 or q)
```

### The Sieve Phase (Adapting Existing Infrastructure)

The TF sieve is structurally identical to the prime sieve:
- Bit-packed bitmap where each bit represents a candidate k
- Small primes mark composites via bit-clearing
- L1-sized segments (32KB = 262K candidates per segment)

Key differences from the prime sieve:
- Candidates are k values, not odd numbers
- The marking condition is `(2kp + 1) mod s == 0`, not `n mod s == 0`
- This simplifies to: `k ≡ -(p^(-1)) × ((s-1)/2) (mod s)` for each small prime s
- The modular inverse `p^(-1) mod s` is computed once per small prime on CPU

The word-centric marking kernel (v6 small-prime) adapts directly — same shared
memory prime cache, same atomic AND bit-clear, different stride computation.

### The PowMod Phase (New Core Kernel)

Binary left-to-right modular exponentiation:

```
function powmod(base=2, exp=p, mod=q):
    result = 1
    for each bit of exp (from MSB to LSB):
        result = (result * result) mod q       // square
        if bit is 1:
            result = (result * base) mod q     // multiply
    return result
```

For p ~ 140,000,000 (27 bits), this is a loop of 27 iterations.
Each iteration: 1-2 modular multiplications.

**One GPU thread per candidate.** Each thread:
1. Reads its k value from the compacted survivor array
2. Computes q = 2kp + 1
3. Runs 27 iterations of square-and-multiply mod q
4. Writes result to B0[RESULT_BASE + thread_id]

No thread divergence: every thread runs exactly 27 iterations (the exponent
bits are the same for all threads — they differ only in q).

### Modular Multiplication for 73+ Bit Numbers

This is the one genuinely new capability needed. Three approaches:

**Approach A: Barrett reduction with uint64 pairs (recommended)**

Represent q as two uint64 values: `q_hi` (high bits) and `q_lo` (low bits).
Modular multiplication via:
1. Full multiply: `a * b` → 128-bit product (4 partial uint64 products)
2. Barrett reduction: divide by q using precomputed reciprocal
3. Subtract: result = product - q * quotient_estimate

Needs new IR op: `ir_mulhi64(block, a, b)` — upper 64 bits of 64x64 multiply.
SPIR-V doesn't have a direct mulhi instruction, but it can be emulated:
- Use `OpIMul` on uint64 for the lower 64 bits
- Use the "shift and add" decomposition for the upper 64 bits
- Or: split each uint64 into two uint32 halves, do 4 schoolbook multiplies

**Approach B: Three-word uint32 (mfaktc's approach)**

Represent q as three uint32 words (96 bits). Barrett reduction using 32-bit
multiplies. Directly mirrors mfaktc's Barrett76/Barrett87 kernels.

More IR ops per multiplication but each op is 32-bit (fast). Well-proven by
mfaktc's decade of production use.

**Approach C: Start with pure uint64 (immediate, limited)**

For bit levels up to 63, no multi-word arithmetic needed. Pure `ir_imul64` +
`ir_umod64`. This gets us running immediately for supplementary work (tf1G
project — exponents above 1 billion where lower bit levels still have work).

**Recommended path: Start with C (immediate results), then implement A or B
for 73+ bits (production GIMPS contribution).**

---

## The Swarm Advantage

mfaktc runs one exponent on one GPU with a single CUDA kernel. OctoFlow's
VM swarm enables patterns that mfaktc cannot express:

### Multi-Exponent Batching

Assign different exponents to different VMs. While VM 0 sieves candidates
for exponent p1, VM 1 runs powmod for exponent p2. The dispatch chains are
independent — perfect for async execution.

mfaktc must finish one exponent before starting the next. OctoFlow can
interleave sieve and powmod phases across multiple exponents, keeping the
GPU saturated even when individual phases have low occupancy.

### Adaptive Candidate Flow

The sieve phase produces a variable number of survivors. If segment A
produces 200 survivors and segment B produces 50, the powmod phase for A
has 4x more work. In mfaktc, this creates warp underutilization for B.

With OctoFlow's dispatch chain, we can:
1. Run sieve for multiple segments
2. Read survivor counts
3. Batch powmod dispatches proportional to survivor count
4. Record the full chain before execution

This is the "bucket dispatch" pattern (Pattern 3) applied to candidate flow
rather than prime distribution.

### Pre-Recorded Pipeline

The entire TF pipeline for an assignment — sieve all segments, compact
survivors, powmod all candidates, check results — records as a single
dispatch chain per VM. At current bit levels (~73 bits), a typical assignment
tests ~10^9 candidates. With 262K candidates per sieve segment, that's
~4,000 segments × 5 kernels = 20,000 dispatches per VM.

This is exactly the scale where our batch dispatch chain excels — the sieve
v7 at pi(10^10) already handles 95,370 dispatches with zero CPU intervention.

---

## Performance Projections

### Baseline: mfaktc on GTX 1660 Ti

- Barrett76 kernel: ~1,650 GHz-days/day at default power
- Barrett87 kernel: ~1,530 GHz-days/day
- GIMPS minimum assignment: 5.0 GHz-days
- Time per assignment: ~4-7 minutes

### Projected: OctoFlow on GTX 1660 SUPER

| Factor | Gap vs mfaktc | Notes |
|--------|---------------|-------|
| Dispatch overhead | 5-10x | Vulkan vs CUDA kernel launch |
| Arithmetic efficiency | 2-3x | IR-generated vs hand-tuned Barrett |
| Sieve efficiency | ~1x | Our sieve infrastructure is proven |
| Parallelism | ~1x | VM swarm vs CUDA thread blocks |
| **Combined** | **~10-20x slower** | |

**Projected throughput: ~80-160 GHz-days/day**

- Time per minimum assignment (5 GHz-days): ~45-90 minutes
- Assignments per day: ~16-32
- **This is useful.** GIMPS accepts any valid result above the 5.0 GHz-day threshold.

### Path to Competitive (3-5x of mfaktc)

| Optimization | Expected Gain | When |
|-------------|---------------|------|
| Compiled dispatch recording | 3-5x total time | Future (self-hosting compiler) |
| Tuned Barrett kernel | 1.5-2x powmod | After initial implementation |
| Workgroup-level sieve | 1.3x sieve | Shared memory optimization |
| Persistent kernels (Vulkan) | 2-3x dispatch | Requires runtime extension |

With compiled dispatch + tuned Barrett: projected **400-800 GHz-days/day**,
bringing OctoFlow within 2-4x of mfaktc.

---

## Implementation Roadmap

### Phase 1: Pure uint64 TF (immediate)

Build the pipeline with uint64 arithmetic (bit levels up to 63).
Validates the architecture end-to-end without multi-word arithmetic.

**New kernels:**
- `tf_sieve_init` — adapt sieve_init_v6 (bitmap init with candidate range)
- `tf_sieve_mark` — adapt sieve_mark_v6_small (mark composites of small primes)
- `tf_compact` — parallel prefix sum + scatter (extract survivors)
- `tf_powmod64` — binary modular exponentiation, pure uint64
- `tf_check` — compare results to 1

**New IR ops:**
- None! Pure uint64 powmod uses existing `ir_imul64`, `ir_umod64`, `ir_iadd64`.

**Validation:**
- Known factor database: GIMPS publishes all known factors
- Test against known factors for small exponents
- Cross-validate with mfaktc on the same ranges

**Deliverable:** Working TF at bit level 63. Valid results for tf1G project.

### Phase 2: Barrett76 (73-bit, production GIMPS)

Extend arithmetic to 73+ bits using Barrett reduction.

**New IR ops (one of):**
- `ir_mulhi64` — upper 64 bits of 64x64 multiply (emulated via uint32 schoolbook)
- Or: direct 96-bit Barrett using three uint32 words

**New kernel:**
- `tf_powmod76` — Barrett reduction modular exponentiation, 73-76 bits

**Validation:**
- GIMPS PrimeNet submission with CRC32 checksums (mfaktc 0.24+ format)
- Cross-validate against known factors at 73-bit level

**Deliverable:** Production GIMPS TF contribution. Valid PrimeNet results.

### Phase 3: Swarm Optimizations

Multi-exponent batching, adaptive candidate flow, optimized Barrett kernels.

**Deliverable:** Demonstrated throughput within 3-5x of mfaktc.

---

## GIMPS Integration

### PrimeNet Protocol

GIMPS uses PrimeNet to assign and collect work. For trial factoring:

1. **Get assignment:** Request exponent p and bit level range via PrimeNet API
   (or manual via mersenne.org)
2. **Do work:** Sieve + powmod for the assigned bit range
3. **Report result:** Submit factor (if found) or "no factor" with CRC32 checksum

For initial development, manual assignments via mersenne.org are sufficient.
Automated PrimeNet integration (via AutoPrimeNet scripts) can come later.

### Result Format

mfaktc 0.24+ reports results with CRC32 checksums for validation. OctoFlow
must produce compatible result files. The format is documented in mfaktc source.

Starting 2026, GIMPS requires CRC32 checksums for all TF results.

### Verification

Every TF result is independently verified by GIMPS. A wrong result (missed
factor or false positive) is detected and flagged. This provides automatic
correctness validation for OctoFlow's arithmetic — the global GIMPS network
acts as our test oracle.

---

## What We Already Have vs What We Need

### Existing (from sieve v1-v7)

| Capability | Sieve Version | TF Application |
|-----------|---------------|----------------|
| Bit-packed sieve + popcount | v2 | Candidate sieve |
| L1-sized segments (32KB) | v3 | Segment candidate ranges |
| Shared memory prime cache | v3 | Small-prime sieve optimization |
| Bucket dispatch | v3 | Adaptive candidate flow |
| Async VM swarm (16 VMs) | v3+ | Multi-exponent batching |
| Batch dispatch chains | v2+ | Full TF pipeline per VM |
| uint64 SPIR-V arithmetic | v6 | 64-bit modular exponentiation |
| Sentinel initialization | v7 | Lazy powmod dispatch |
| f32 precision workarounds | v2-v7 | Push constant encoding |
| `float_to_bits()` readback | v2+ | Factor extraction |

### New (to build)

| Capability | Complexity | Dependency |
|-----------|-----------|------------|
| Modular inverse (CPU, one-time) | Low | None |
| tf_sieve_mark stride computation | Low | Modular inverse |
| tf_compact (prefix sum + scatter) | Medium | New kernel pattern |
| tf_powmod64 (binary mod-exp) | Medium | Existing uint64 IR |
| tf_powmod76 (Barrett reduction) | Medium-High | New multi-word arithmetic |
| Result file generation | Low | Format documentation |

### Estimated Effort

| Phase | New Kernels | New IR Ops | Estimated Work |
|-------|------------|-----------|----------------|
| Phase 1 (uint64) | 5 | 0 | Moderate — sieve adapts, powmod is new |
| Phase 2 (Barrett76) | 1 | 1-2 | Moderate — Barrett is well-documented |
| Phase 3 (swarm) | 0 | 0 | Light — orchestrator changes only |

---

## Risk Assessment

### Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Barrett arithmetic bugs | High | Cross-validate against mfaktc on known factors |
| Dispatch overhead dominates | Medium | Batch more candidates per segment |
| VRAM limits per VM | Low | 32KB sieve + small result buffers |
| f32 precision in k computation | Medium | Apply sieve learnings (compute on GPU, not push constants) |
| CRC32 checksum mismatch | Medium | Implement CRC32 in .flow, validate against mfaktc |

### Strategic Risks

| Risk | Assessment |
|------|-----------|
| "Too slow to be useful" | At 80+ GHz-days/day, we clear minimum thresholds. Useful. |
| "PRP/LL would be more impressive" | True, but requires NTT — a much larger project. TF is the right entry point. |
| "Already solved by mfaktc" | Yes, but we're proving OctoFlow can do it, not replacing mfaktc. The showcase value is the point. |

---

## Success Criteria

### Minimum Viable

- [ ] Produce a correct "no factor found" TF result for one GIMPS assignment
- [ ] Cross-validate against mfaktc for 10+ known factors
- [ ] Throughput > 50 GHz-days/day (clears GIMPS minimum thresholds)

### Full Success

- [ ] Submit 100+ valid TF results to PrimeNet
- [ ] Throughput > 200 GHz-days/day
- [ ] Multi-exponent batching demonstrated (swarm advantage)
- [ ] Find an actual factor (luck-dependent, but ~1 in 50 assignments finds one)

### Aspirational

- [ ] Within 5x of mfaktc throughput
- [ ] Automated PrimeNet integration
- [ ] Published as OctoFlow showcase (blog post / mersenneforum thread)

---

## References

- [GIMPS How It Works](https://www.mersenne.org/various/works.php)
- [GIMPS The Math](https://www.mersenne.org/various/math.php)
- [GIMPS Assignment Rules](https://www.mersenne.org/thresholds/)
- [mfaktc — CUDA Trial Factoring](https://github.com/primesearch/mfaktc)
- [mfakto — OpenCL Trial Factoring](https://github.com/Bdot42/mfakto)
- [GIMPS Factoring and Sieving](https://www.rieselprime.de/ziki/GIMPS_factoring_and_sieving)
- [GPU TF Performance Chart](https://www.mersenne.ca/mfaktc.php)
- [gpuowl / PRPLL — GPU Primality Test](https://github.com/preda/gpuowl)
- [GIMPS Current Status](https://www.mersenne.ca/)
- [Efficient NTT on GPU (paper)](https://eprint.iacr.org/2021/124.pdf)
- [GPU VM Learnings](gpu-vm-learnings.md) — OctoFlow sieve optimization patterns
- [GPU Sieve Documentation](gpu-sieve.md) — v1-v7 implementation details
