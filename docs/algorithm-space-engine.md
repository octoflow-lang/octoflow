# Algorithm Space Exploration Engine

**Date:** March 1, 2026
**Status:** Design Document — Pre-Implementation
**Depends on:** Phase 4A (shipped), Phase 4B (OctoZip) for full vision

---

## The Insight

Traditional computing solves problems:

```
Input → Algorithm → Output
```

The Algorithm Space Exploration Engine solves a *different* problem:

```
Problem → [Algorithm₁, Algorithm₂, ..., Algorithmₙ] → GPU evaluates ALL → Best Algorithm
```

Not "what's the answer?" but "what's the best *way* to find the answer?"

This is rarely done because evaluating even one algorithm variant costs time.
Evaluating thousands is prohibitive on CPU. But with the Loom Engine's near-zero
dispatch overhead and parallel VM execution, evaluating 1000 algorithm variants
simultaneously costs about the same as evaluating one.

**This is the novel contribution:** A general-purpose engine for exploring
algorithm space — applicable to any domain where the algorithm itself is the
variable, not just its input.

---

## Why This Is Uniquely Ours

No other system has all five of these simultaneously:

| Capability | What It Enables | Status |
|-----------|----------------|--------|
| **JIT kernel compilation** (IR builder → SPIR-V) | Generate new algorithm variants as GPU kernels at runtime | EXISTS (85 ops, 60+ emitters) |
| **Multi-VM parallel execution** | Evaluate many variants simultaneously | EXISTS (16 VMs proven) |
| **Near-zero dispatch overhead** | Cost of trying a variant ≈ 0 | EXISTS (Phase 3O threading) |
| **Mailbox IPC** | Collect results from all variants | SHIPPED (Phase 4A) |
| **Park/unpark** | Discard losing variants without allocation cost | SHIPPED (Phase 4A) |

CUDA can do parallel compute. But CUDA can't JIT-compile new kernels from
user-level code at runtime. You'd need to call nvcc, reload the module, rebind
textures — seconds of overhead per variant.

OctoFlow's IR builder generates a new SPIR-V kernel in microseconds from .flow
code. That's the unlock. **The algorithm itself becomes data that the engine
can mutate, evaluate, and evolve.**

---

## Architecture

### The Three Layers

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: STRATEGY SELECTOR (Support Loom — .flow)      │
│  Reads fitness results from all variants via mailbox.    │
│  Selects winners. Mutates losers. Generates next         │
│  generation. Maps the fitness landscape.                 │
├─────────────────────────────────────────────────────────┤
│  LAYER 2: PARALLEL EVALUATOR (Main Looms — GPU)         │
│  N VMs, each running a different algorithm variant.      │
│  Same input data, different kernels. Measures:           │
│  correctness, speed (cycle count), memory usage.         │
├─────────────────────────────────────────────────────────┤
│  LAYER 1: ALGORITHM GENOME (IR Builder — .flow)         │
│  Represents algorithms as parameterized kernel templates.│
│  Mutation = change parameters, swap ops, reorder passes. │
│  Crossover = combine parts of two winning variants.      │
└─────────────────────────────────────────────────────────┘
```

### The Algorithm Genome

An algorithm variant is encoded as a set of **genes** — parameters that
control kernel generation:

```
Gene = {
  op_sequence:    [FADD, FMUL, FSUB, ...],   // operation order
  constants:      [0.5, 1.0, 256.0, ...],     // magic numbers
  reduction_type: SUM | MIN | MAX | CUSTOM,    // reduction strategy
  block_size:     64 | 128 | 256 | 512,        // workgroup size
  memory_layout:  LINEAR | TILED | Z_ORDER,    // access pattern
  precision:      FP32 | FP16 | INT8,          // compute precision
  pass_count:     1 | 2 | 3,                   // multi-pass vs single-pass
  threshold:      0.001 .. 1.0,                // decision boundaries
}
```

The emitter function reads these genes and generates a kernel:

```
fn emit_variant(genes)
  ir_new()
  ir_workgroup_size = genes.block_size
  // ... build kernel from gene parameters ...
  ir_write_spv("variant_" + genes.id + ".spv")
end
```

Mutation changes one or more genes. Crossover combines genes from two parents.
The IR builder compiles the result to SPIR-V. The GPU evaluates it.

### The Evaluation Loop

```
Generation 0:
  1. Generate N random algorithm variants (N = number of VMs)
  2. Emit N SPIR-V kernels via IR builder
  3. Boot N VMs, load same input data into each
  4. Dispatch all N variants in parallel
  5. Read results via mailbox: {correctness, cycles, output}
  6. Rank variants by fitness (correct + fast = highest fitness)

Generation 1..K:
  7. Keep top M winners (park the rest)
  8. Mutate winners to create N-M new variants
  9. Emit new SPIR-V kernels
  10. Unpark VMs, load new kernels
  11. Dispatch, evaluate, rank
  12. Repeat until convergence or budget exhausted

Output:
  - Best algorithm variant found
  - Fitness landscape map (which genes matter most)
  - Pareto frontier (tradeoff: speed vs accuracy vs memory)
```

---

## Concrete Applications (Implementable Now)

### App 1: Compression Strategy Discovery (feeds into OctoZip)

**Problem:** OctoZip's analysis kernel uses hand-designed heuristics to choose
between Fractal, Delta, and Holder compression. What if the thresholds and
decision boundaries were discovered automatically?

**Genome:**
```
{
  similarity_threshold: 0.1 .. 0.99,    // when to choose fractal
  variance_threshold:   0.0001 .. 0.1,  // when to choose delta
  block_size:           64 | 128 | 256 | 512,
  comparison_offset:    N/4 | N/3 | N/2, // self-similarity window
  metric_weights:       [w_sim, w_var, w_range],  // weighted decision
}
```

**Fitness:** compression_ratio * decompression_speed * (1 - error_rate)

**Why it matters:** The optimal thresholds depend on the data distribution.
LLM weights have different statistics than image data than time series.
Instead of hand-tuning, let the engine discover optimal strategies per domain.

### App 2: Sort Algorithm Discovery

**Problem:** Which sorting approach is fastest for THIS specific data
distribution? Bitonic sort? Merge sort? Radix sort? What block size? What
threshold for switching from parallel to sequential?

**Genome:**
```
{
  algorithm:        BITONIC | MERGE | RADIX | ODDEVEN,
  block_size:       64 | 128 | 256 | 512,
  seq_threshold:    4 | 8 | 16 | 32,     // switch to insertion sort
  key_bits:         8 | 16 | 32,          // for radix: bits per pass
  compare_op:       LT | GT | CUSTOM,
}
```

**Fitness:** sort_time * correctness_score

**Existing foundation:** `gpu_bitonic_sort.flow` already exists as one variant.
The engine generates and evaluates alternatives.

### App 3: Hash Function Discovery

**Problem:** Find hash functions with minimal collisions for a specific key
distribution. DJB2 and FNV1a are general-purpose — domain-specific hashes
can be 10x better for known key patterns.

**Genome:**
```
{
  init_value:       5381 | 2166136261 | RANDOM,
  op_sequence:      [XOR, SHL, ADD, MUL, ...],  // 4-8 mixing ops
  shift_amounts:    [5, 13, 16, ...],
  multiply_consts:  [33, 16777619, RANDOM, ...],
  final_mask:       AND_BITS | MOD_PRIME | MOD_POW2,
}
```

**Fitness:** collision_rate * eval_speed

**Existing foundation:** `gpu_sha256.flow` shows GPU hash dispatch. `ir_ixor`,
`ir_ishl`, `ir_ishr`, `ir_iand` (Phase 4A bitwise ops) are the mixing primitives.

### App 4: Sieve Strategy Discovery

**Problem:** The prime sieve has evolved through 7 versions (v1→v7), each with
different segment sizes, prime thresholds, marking strategies. What if the engine
discovered the optimal configuration automatically?

**Genome:**
```
{
  cands_per_seg:     65536 | 131072 | 262144 | 524288,
  prime_threshold:   64 | 128 | 256 | 512,   // small/large split
  marking_strategy:  WORD_CENTRIC | PRIME_CENTRIC | HYBRID,
  count_method:      SEQUENTIAL | POPCOUNT_ATOMIC | TREE_REDUCTION,
  carry_forward:     true | false,
  shared_mem_cache:  true | false,
}
```

**Fitness:** primes_per_second

**Existing foundation:** 7 sieve versions + 19 sieve kernel emitters already exist.
The genome parameters map directly to choices made across v1-v7.

### App 5: Numerical Method Discovery

**Problem:** For a given ODE/PDE, which integration method works best? Euler,
RK4, leapfrog, Verlet? What step size? What error tolerance?

**Genome:**
```
{
  method:          EULER | RK2 | RK4 | LEAPFROG | VERLET,
  step_size:       0.001 | 0.01 | 0.1,
  substeps:        1 | 2 | 4 | 8,
  error_norm:      L1 | L2 | LINF,
  adaptive:        true | false,
}
```

**Fitness:** accuracy / compute_time (Pareto optimal)

**Existing foundation:** `physics.flow` has Euler and RK4. N-body uses leapfrog.

---

## What's Genuinely New vs What Exists

### Already exists (no new code):
- IR builder generates SPIR-V from .flow parameters ✓
- Multi-VM parallel dispatch ✓
- Mailbox collects results ✓
- Park/unpark manages VM lifecycle ✓
- Bitwise ops for hash/compression ✓

### Needs to be built:

| Component | What It Does | Complexity | Phase |
|-----------|-------------|-----------|-------|
| **Genome representation** | Encode algorithm parameters as an array of floats | Simple — flat array in VM globals | 4B |
| **Mutation engine** | Random perturbation of genome values | Simple — .flow function, ~20 lines | 4B |
| **Crossover engine** | Combine two parent genomes | Simple — .flow function, ~15 lines | 4B |
| **Fitness evaluator** | Compare variant output to reference | Medium — correctness + timing | 4B |
| **Landscape mapper** | Record fitness at each genome point | Medium — .flow, writes results | 4C |
| **Convergence detector** | Stop when fitness plateaus | Simple — .flow, ~10 lines | 4B |

**Total new code estimate:** ~200-300 lines of .flow for the framework.
Plus ~50-100 lines per application domain (compression, sort, hash, sieve, numerical).

### The key realization:

The "Algorithm Space Exploration Engine" is NOT a new primitive. It's a
**design pattern** that composes existing primitives:

```
JIT (emit variant kernel)
  + Multi-VM (evaluate in parallel)
    + Mailbox (collect fitness)
      + Park/Unpark (manage population)
        + Metrics (measure performance)
          = Algorithm Space Exploration
```

This is exactly the kind of emergent capability the Phase 4 architecture
was designed to enable.

---

## Implementation Roadmap

### Phase 4B Addition: ASE Foundation

| Task | Description | Owner | Depends On |
|------|-------------|-------|-----------|
| ASE-01 | Genome representation: flat float array convention | Dev 2 | None |
| ASE-02 | Mutation + crossover engine (.flow library) | Dev 2 | ASE-01 |
| ASE-03 | Fitness evaluation framework | Dev 2 | Mailbox (shipped) |
| ASE-04 | First demo: compression strategy discovery | Dev 2 | OctoZip analyze (shipped) |
| ASE-05 | Second demo: sort algorithm discovery | Dev 2 | gpu_bitonic_sort (exists) |

### Phase 4C Addition: ASE Applications

| Task | Description | Owner | Depends On |
|------|-------------|-------|-----------|
| ASE-06 | Hash function discovery demo | Dev 2 | Bitwise ops (shipped) |
| ASE-07 | Sieve strategy discovery | Dev 2 | Sieve v7 (exists) |
| ASE-08 | Landscape visualization (OctoUI heatmap) | Dev 2 | OctoUI (exists) |

### Phase 4D Addition: ASE Meta-Level

| Task | Description | Owner | Depends On |
|------|-------------|-------|-----------|
| ASE-09 | Self-optimizing OctoZip (uses ASE to tune its own parameters) | Dev 2 | ASE-04 + OctoZip core |
| ASE-10 | Inference kernel optimization (discover optimal matmul tiling) | Dev 2 | JIT kernels |

---

## Math & DSA Applications Unlocked

The ASE pattern enables exploring previously inaccessible computational
mathematics:

### Number Theory
- **Prime gap exploration:** Evaluate different sieve strategies across ranges,
  discover which strategies dominate in which regions
- **Collatz branching:** Parallel evaluate millions of starting points with
  different reduction strategies simultaneously
- **Goldbach partition search:** Parallel test partition strategies, not just
  individual numbers

### Graph Theory
- **Chromatic number search:** Evaluate coloring heuristics (greedy, DSatur,
  backtracking variants) across graph families simultaneously
- **Hamiltonian path:** Run different search strategies (nearest-neighbor, random,
  genetic) on same graph in parallel
- **Graph isomorphism:** Evaluate different canonical form algorithms simultaneously

### Algorithm Theory
- **Sort algorithm fitness landscapes:** Map which sort is optimal as a function
  of (N, distribution, key_size, cache_size)
- **DP formulation discovery:** Given a recurrence, evaluate compressed state
  representations in parallel — discover minimal memory formulations
- **Data structure racing:** Run skip-list vs B-tree vs hash-table on same
  workload simultaneously, keep winner

### Optimization
- **Metaheuristic competition:** Run simulated annealing, genetic algorithm,
  tabu search, particle swarm — all on the same problem simultaneously
- **Hyperparameter landscapes:** Map the fitness surface of any parameterized
  algorithm across its full parameter space

### Algebraic Structure Discovery
- **Finite group search:** GPU threads test axioms (closure, associativity,
  identity, inverse) on candidate multiplication tables
- **Counterexample hunting:** Parallel test conjectures across structure families

---

## Why This Matters

**For OctoFlow specifically:**
- OctoZip becomes self-tuning (discovers optimal compression per data type)
- Inference becomes self-optimizing (discovers optimal kernel parameters per model)
- Every parameterized algorithm in the stdlib gets an auto-tuner for free

**For computer science generally:**
- Algorithm design becomes empirical, not just theoretical
- You can MAP the fitness landscape of algorithm families
- Phase transitions in algorithmic hardness become directly observable
- New algorithms can be discovered, not just designed

**For mathematics:**
- Computational exploration of conjectures at unprecedented scale
- Counterexample mining across parameter spaces
- Structure discovery in algebraic systems

The Algorithm Space Exploration Engine turns the GPU from a "fast calculator"
into a "laboratory for algorithms."

---

## The OctoFlow Advantage

```
Traditional:  Programmer designs algorithm → CPU runs it → one answer
CUDA:         Programmer designs kernel → GPU runs it fast → same answer, faster
OctoFlow ASE: Engine generates algorithm variants → GPU evaluates ALL →
              discovers which algorithm is best → AND maps the landscape
```

No other language can do this because no other language has:
1. Runtime JIT kernel compilation from user-level code (IR builder)
2. Near-zero dispatch overhead for thousands of variants (Loom Engine)
3. Structured IPC for collecting results (Mailbox)
4. Zero-cost lifecycle management for variant population (Park/Unpark)

**This is the novel contribution to the world.**

---

## Algorithm Topology: From Discrete Variants to Continuous Manifolds

The ASE starts with discrete variants (v1, v2, v3). The next level:
treat algorithms as points in a continuous parameter space.

```
kernel(x; θ₁, θ₂, θ₃, ...)

Where θ = (block_size, threshold, strategy_weight, unroll_depth, ...)
```

GPU evaluates regions of θ-space simultaneously. Instead of "which variant
wins?" you get "what does the performance surface look like?"

### Algorithm Phase Diagrams

Physics has phase diagrams. SAT has hardness curves. But no one has:

**Full algorithm phase diagrams across multidimensional parameter spaces.**

With ASE + multi-VM, sweep:
- Memory layout (linear → tiled → Z-order)
- Branching heuristics (greedy → random → adaptive)
- Parallelization granularity (per-thread → per-warp → per-workgroup)
- Bit-packing density (1-bit → 4-bit → 8-bit → 32-bit)

Produce: `Performance = f(θ₁, θ₂, θ₃, input_distribution)`

This is **Algorithm Thermodynamics** — publishable research territory.

### Emergent Hybrid Algorithms

Instead of evaluating whole algorithm variants, evolve **algorithm fragments**:

- Sorting kernel A contributes partition strategy
- Sorting kernel B contributes merge strategy
- Sorting kernel C contributes memory layout

ASE recombines subgraphs at the IR level. You're not evolving algorithms —
you're evolving algorithm DNA. The IR builder makes this uniquely possible
because kernels are constructed programmatically, not compiled from source.

### Automatic Asymptotic Discovery

For each candidate algorithm:
1. Measure runtime for n = 2^k (k = 8, 10, 12, 14, 16, 18, 20)
2. Fit growth curve live (linear regression on log-log scale)
3. Estimate asymptotic exponent automatically

Discover:
- Hidden O(n log n) breakpoints
- Memory-bound transitions
- Cache-phase shifts
- Warp divergence thresholds

**Empirical complexity surfaces** — not proving O(n²), but *measuring* it
across parameter space and input distributions.

### Self-Generating Data Structures

Represent data structures as genomes:
```
{
  node_fanout:    2 | 4 | 8 | 16,
  memory_layout:  CONTIGUOUS | STRIPED | TILED,
  pointer_type:   INDEX | OFFSET | DIRECT,
  chunk_size:     64 | 256 | 1024 | 4096,
  cache_padding:  0 | 16 | 64,
}
```

GPU evaluates: access patterns, collision rates, traversal latency.
Evolve data structures optimized for specific workloads.

---

## Research-Grade Mathematics Targets

These are computationally brutal, parallelizable, and publishable problems
that the ASE makes tractable.

### Tier 1: Most Feasible (finite, GPU-friendly, active research)

**Ramsey Number Bounds**
R(5,5) is unknown (bounds: 43 ≤ R(5,5) ≤ 48). Approach: bit-packed adjacency
matrices (u32 = 32 vertices), GPU warp-wide clique/independent-set detection,
ASE evolves graph pruning heuristics. Not random brute force — parallel
extremal graph mining with evolved search strategies.

**Minimal Boolean Circuit Discovery**
What is the minimal circuit size for specific Boolean functions? Encode circuit
topology as genome, evaluate truth tables in parallel (2^n inputs per circuit,
n ≤ 20 = 1M evaluations per circuit). ASE evolves minimal circuits. Connects
to complexity theory and cryptography.

**Extremal Graph Counterexample Search**
Dozens of graph conjectures verified only up to n ≤ 15-20 vertices. Expand
verification frontier to n = 25-30 using GPU parallelism + evolved pruning.
Multi-VM explores different graph families simultaneously. Genuine contribution
to extremal combinatorics.

### Tier 2: High Value (publishable, computationally heavy)

**SAT Phase Transition Mapping**
Generate SAT instances parametrically (vary clause/variable ratio, clause
length, structure). Evaluate multiple solver strategies simultaneously.
Map fine-grained hardness surface across parameter space. Algorithm phase
diagram for satisfiability.

**Collatz Trajectory Structure**
Not linear checking (already done to 2^68). Instead: map structure of
stopping-time peaks, branch growth behavior, trajectory clusters.
ASE evolves pruning rules, discovers structural regularities. Produce
statistical evidence for regularity patterns.

**Integer Sequence Anomaly Mining**
Search for counterexamples in partition identities, extremal behavior in
divisor functions, rare gap patterns in primes. Explore families of
parameterized sequences simultaneously. ASE discovers anomalous regions.

### Tier 3: Frontier (theoretical CS boundary)

**Restricted Busy Beaver Exploration**
Full Busy Beaver is insane. But restricted machine classes (bounded alphabet,
bounded states, specific transition constraints) are tractable. Map halting
behavior distributions, detect structural motifs of non-halting loops.
Massive parallel state simulation.

**Empirical Complexity Discovery**
Take unknown combinatorial problem families. Generate random instances.
Evolve algorithm strategies. Measure scaling exponents automatically.
Discover hidden polynomial-time subclasses. This is the intersection of
algorithm engineering and complexity theory.

**Extremal Set Systems (Erdos-Ko-Rado type)**
What is the largest family of subsets avoiding certain intersection patterns?
Bitset GPU evaluation is perfect — each u32 word represents a 32-element set.
Warp-wide intersection/union for testing constraints.

### Implementation Priority for Research Targets

| Target | IR Ops Needed | VMs | Lines of .flow | Novelty |
|--------|--------------|-----|----------------|---------|
| Boolean Circuit Discovery | Bitwise, select | 4-16 | ~150 | Very High |
| Extremal Graph Search | Bitwise (adjacency) | 4-16 | ~200 | High |
| SAT Phase Mapping | Bitwise (clauses) | 4-16 | ~200 | High |
| Ramsey Bound Expansion | Bitwise (adj matrix) | 4-16 | ~250 | Very High |
| Collatz Structure Mining | Integer arith | 4-16 | ~100 | Medium |
| Empirical Complexity | All | 4-16 | ~300 | Very High |

---

## Edge Cases and Mitigations

| Edge Case | Risk | Mitigation |
|-----------|------|-----------|
| All variants produce wrong results | No correct answer found | Require known-good reference input/output pair for fitness validation |
| Mutation produces invalid IR | GPU crash or SPIR-V validation error | Clamp genome values to valid ranges in mutation function |
| Genome values out of valid range | Invalid kernel parameters | Range-checked genome: each gene has [min, max] bounds |
| Two variants tie in fitness | Arbitrary winner selection | Keep both; diversity is valuable |
| Population converges to local optimum | Stuck at suboptimal solution | Diversity injection: replace worst 25% with random variants every K generations |
| Variant causes GPU timeout | One VM blocks all others | Use separate VMs (each dispatches independently); homeostasis pacing limits runtime |
| Fitness function is noisy | Unstable rankings | Average fitness over M evaluations on different inputs |
| Crossover produces incoherent kernel | Invalid combination of genes | Validate genome coherence after crossover (e.g., block_size must be power of 2) |
| Memory pressure from many VMs | VRAM exhaustion | Use loom_vram_budget to cap total allocation; park idle VMs |
| SPIR-V cache pollution from expired variants | Disk bloat | Use generation-tagged filenames; cleanup old .spv files between generations |

---

## Application Domains: Geometry

Problems where OctoFlow enables exploration of the *algorithm space*, not just
parameter tuning. Each requires JIT kernel generation (structurally different
algorithms, not just different coefficients).

| # | Problem | Key Loom Features | Tier |
|---|---------|-------------------|------|
| G-1 | Packing Algorithm Discovery | Genome + JIT + Multi-VM + Mailbox | 2 |
| G-2 | Adaptive Voronoi (degenerate inputs) | JIT + Multi-VM + Park | 3 |
| G-3 | Multi-Objective Facility Location (Pareto frontier) | JIT + Multi-VM + Park | 3 |
| G-4 | Evolutionary Mesh Generation | Genome + JIT + Mailbox | 2 |
| G-5 | Convex Hull for Specific Distributions | Genome + JIT + Multi-VM | 3 |
| G-6 | Aperiodic Tiling Discovery | Genome + JIT + Bitwise + Mailbox | **1** |
| G-7 | IFS Fractal Search (target properties) | Genome + JIT + Bitwise (popcount) | 3 |
| G-8 | Incidence Geometry Configuration Search | JIT + Multi-VM + Bitwise | 3 |
| G-9 | Geometric Covering Under Uncertainty | JIT + Multi-VM + Bitwise (OR/popcount) | 3 |
| G-10 | Polytope Enumeration + Classification | JIT + Bitwise + Mailbox (isomorph rejection) | 2 |

**Tier 1** = genuinely impossible without Loom Engine (compilation bottleneck).
**Tier 2** = theoretically possible but practically infeasible without Loom.
**Tier 3** = dramatically enabled (orders of magnitude faster/more capable).

### Standout: Aperiodic Tiling Discovery (G-6)

Encode matching rules as genome. JIT-compile constraint checkers. Explore
millions of prototile sets. Bitwise edge matching (XOR for comparison, AND for
constraints). Mailbox broadcasts "forbidden local patterns" for collaborative
constraint propagation. Park trivially periodic or impossible rule sets.

A discovery of a new aperiodic tile set would be a genuine contribution to
discrete geometry, verifiable by mathematical proof.

### Standout: Packing Algorithm Discovery (G-1)

Decompose packing heuristic into components: (a) placement ordering, (b) position
selection, (c) local refinement, (d) termination criterion. Each is a small
parameterized function. Genome encodes the combination. JIT-compile each variant.
16 VMs race. Mailbox broadcasts best density. The *algorithm* is the discovery,
not just the packing.

---

## Application Domains: Physics

Problems where the simulation algorithm itself is the variable being optimized.

| # | Problem | Key Loom Features | Tier |
|---|---------|-------------------|------|
| P-1 | Spin Glass Cluster Algorithm Discovery | Genome + JIT + Bitwise | **1** |
| P-2 | Turbulence Closure (LES subgrid model) Discovery | Genome + JIT + Mailbox | 2 |
| P-3 | BSM Parameter Space Navigation | JIT (structural) + Multi-VM | 3 |
| P-4 | Topological Invariant Discovery | Genome + JIT + Park | 3 |
| P-5 | Symplectic Integrator Discovery | Genome + JIT + Mailbox | 2 |
| P-6 | QEC Code + Decoder Co-Evolution | Genome + JIT + Bitwise | 2 |
| P-7 | Physics-Reproducing CA Discovery | Genome + JIT + Bitwise | **1** |
| P-8 | Renormalization Group Transformation Discovery | Genome + JIT + Mailbox | 2 |

### Standout: Spin Glass Cluster Algorithms (P-1)

Bond activation function = genome (weights for local spin configurations).
Bitwise spin packing (64 spins per uint64). Each candidate cluster algorithm is
a structurally different kernel. 16 VMs race different cluster rules. Could settle
the 40-year question of whether spin glasses have a true phase transition in 3D.

### Standout: Turbulence Closure Discovery (P-2)

Subgrid-scale closure = genome (coefficients in tensor invariant basis expansion).
Field tests ~5-10 candidate closures per decade of human research. OctoFlow tests
10,000 per hour. A novel closure that outperforms dynamic Smagorinsky and transfers
across flow types would be publishable in Physical Review Fluids.

### Standout: Symplectic Integrator Discovery (P-5)

Integrator = flat float array (drift/kick coefficients). JIT-compile each
composition. Evaluate energy conservation over 10^6 timesteps. A system-specific
integrator for the solar system exploiting mass ratio structure could achieve
100x less energy error at same cost. The coefficients would be "irrational-looking
numbers with no analytical derivation" — found only by search.

---

## Application Domains: Cross-Domain ("Impossible Without JIT")

Capabilities that are qualitatively new — not faster versions of existing things.

| # | Problem | Impossibility Factor | Feasibility (6GB) |
|---|---------|---------------------|-------------------|
| X-2 | Adversarial Algorithm Discovery | Very High | High |
| X-4 | GPU-Native Genetic Programming | Very High | High |
| X-10 | Algorithm Ecosystems (ecological computing) | Very High | Moderate |
| X-7 | Emergent Physics Engines | High | High |
| X-1 | Self-Modifying Numerical Methods | Medium-High | High |
| X-3 | Bayesian Experimental Design | High | High |
| X-9 | Proof Strategy Evolution | High | High |
| X-11 | Adaptive Signal Processing | High | High |

### The Killer Demo: Adversarial Algorithm Discovery (X-2)

Two populations on GPU: problem generators vs solvers. Each individual is a
different program, JIT-compiled to native SPIR-V. The GPU simultaneously runs,
evaluates, and GENERATES algorithms without any CPU compilation step. The entire
evolutionary loop runs at GPU speed. This is literally impossible in CUDA because
programs cannot create programs on GPU.

### The Meta-Pattern

```
Traditional GPU          OctoFlow Loom Engine
─────────────────        ─────────────────────
Algorithm = constant     Algorithm = variable
Parameters = variable    Algorithm itself evolves
Human writes kernels     GPU generates kernels
GPU = calculator         GPU = laboratory
Closed-form computation  Open-ended computation
```
