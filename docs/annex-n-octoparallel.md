# Annex N — OctoParallel: GPU-Native Parallel Computing Domain

**Date:** February 17, 2026
**Status:** Design — implementation post-self-hosting (Phase 52+)
**Depends on:** Phase 52 (self-hosting), Annex M (Neural Networks)

---

## The Insight

Every domain in OctoFlow needs the same foundation:

- Neural Networks (Annex M) needs matrix multiply, convolution, softmax
- HyperGraphDB (Annex L) needs sparse matrix ops, graph Laplacian, BFS
- OctoMedia (Annex X) needs FFT, convolution, image transforms
- OctoEngine (Annex Q) needs matrix transforms, physics simulation, collision detection
- Data science needs PCA, SVD, covariance, regression

Today, each domain will reimplement these independently.
That is wrong. The math layer should be written once, GPU-accelerated, and shared.

**OctoParallel is that layer.**

It is to OctoFlow what NumPy is to Python — except GPU-native by default,
zero external dependencies, and written in `.flow` after self-hosting.

---

## What OctoParallel Covers

### Layer 1: Tensor Primitives
The fundamental data structure. Everything is a tensor.

```flow
// N-dimensional array with shape metadata
let t = tensor([2, 3])            // 2x3 tensor, zero-initialized
let t = tensor_from(data, [4, 4]) // reshape flat array into 4x4
let shape = t.shape               // [4, 4]
let rank  = t.rank                // 2
let elem  = t[1, 2]               // element access
```

GPU maps directly: a tensor IS a GPU buffer with shape metadata.
No copy. No boxing. The tensor lives on the device.

### Layer 2: Linear Algebra (BLAS equivalent)
GPU-accelerated, written in OctoFlow, zero dependencies.

```flow
// Matrix multiply — the most important operation in computing
let C = matmul(A, B)              // GEMM: C = A × B

// Element-wise
let S = A + B                     // broadcast-aware addition
let P = A * B                     // element-wise multiply (Hadamard)

// Reductions
let s = tensor_sum(A)             // sum all elements
let m = tensor_mean(A)            // mean
let n = tensor_norm(A)            // L2 norm

// Transforms
let T = transpose(A)              // matrix transpose
let R = reshape(A, [6, 2])        // reshape without copy

// Solvers
let x = solve(A, b)               // Ax = b
let Ai = inverse(A)               // A⁻¹
let eig = eigenvalues(A)          // eigenvalues (power iteration on GPU)
```

### Layer 3: Parallel Algorithms (GPU-native DSA)
The algorithms that are theoretically known but poorly supported elsewhere.
See: Research Catalogue below.

```flow
// Sorting — GPU radix sort (optimal for GPU, not just "parallel quicksort")
let sorted = gpu_sort(data)
let sorted = gpu_sort_by(data, fn(x) { x.key })

// Prefix operations — fundamental parallel primitive
let scan   = prefix_sum(data)     // parallel prefix scan (Blelloch)
let segsum = segment_sum(data, flags) // segmented scan

// Parallel search
let idx = parallel_find(data, fn(x) { x > threshold })
let idx = parallel_lower_bound(sorted_data, target)

// Parallel hash table — GPU-native cuckoo hash
let table = gpu_hash_new()
gpu_hash_insert(table, key, value) // warp-level parallel insert
let v     = gpu_hash_get(table, key)

// Parallel graph primitives
let dist  = sssp(graph, source)   // single-source shortest path (GPU)
let cc    = connected_components(graph)
let rank  = pagerank(graph, iters)
```

### Layer 4: Signal Processing
FFT and convolution — used by audio, image, physics.

```flow
let spectrum   = fft(signal)      // Fast Fourier Transform (GPU Cooley-Tukey)
let filtered   = ifft(spectrum)   // Inverse FFT
let convolved  = convolve(signal, kernel)  // 1D convolution
let convolved2 = convolve2d(image, kernel) // 2D convolution (image filters)
```

### Layer 5: Statistics and Decomposition
For data science and ML.

```flow
let cov  = covariance(data)       // covariance matrix
let U, S, V = svd(A)             // Singular Value Decomposition
let P, components = pca(data, k)  // Principal Component Analysis
let mu, sigma = kmeans(data, k)   // K-means clustering (GPU Lloyd's)
```

---

## The Cross-Domain Value

```
OctoParallel (Annex N) — the foundation
       │
       ├─ Annex M: Neural Networks
       │     matmul → linear layers
       │     convolve2d → CNNs
       │     fft → transformer positional encoding
       │
       ├─ Annex L: HyperGraphDB
       │     sssp → shortest path queries
       │     pagerank → relevance scoring
       │     svd → graph embedding
       │
       ├─ Annex X: OctoMedia
       │     convolve2d → image filters, blur, sharpen
       │     fft → audio spectrum, equalizer
       │     gpu_sort → palette extraction
       │
       ├─ Annex Q: OctoEngine
       │     matmul → 3D transforms, MVP matrix
       │     parallel_find → broad-phase collision
       │     sssp → pathfinding
       │
       └─ Data Science (general)
             svd, pca, covariance → analysis
             kmeans → clustering
             gpu_sort → ranking
```

One implementation. GPU-accelerated. Written in pure `.flow`.
Every domain imports it. Zero code duplication.

---

## Research Catalogue: GPU DSA Status

*Survey updated February 2026. Sources: Euro-Par 2024, ICPP 2024, PPoPP 2023,
VLDB 2024, SC24, arXiv 2025. Full citations at end of document.*

---

### Production-Ready (algorithms to implement in OctoParallel post-Phase 52)

**Sorting**
| Algorithm | Perf | Notes |
|-----------|------|-------|
| Radix sort — Onesweep (CUB/Thrust) | ✅ Best | LSD, 8-bit digits, 4 passes for 32-bit. State of art for uniform distributions. The default. |
| Radix sort — FidelityFX (AMD) | ✅ | AMD's production sort for games/graphics |
| Bitonic sort | ✅ Niche | O(n log²n) work — slower than radix for large n, good for n < 64K or as sub-primitive |
| Sample sort | ✅ | Better than radix for non-uniform key distributions |
| RadiK — radix Top-K (ICS 2024) | ✅ New | Scalable radix-based top-K. 2.5x single-query, 4.8x batch over WarpSelect. No shared-memory limit. |

**Hash Tables**
| Structure | Perf | Notes |
|-----------|------|-------|
| WarpCore (HiPC 2020) | ✅ Baseline | Warp-cooperative probing: 1.6B inserts/s, 4.3B lookups/s on GV100. Still the benchmark baseline. |
| WarpSpeed (arXiv Sep 2025) | ✅ Newest | 8 designs: Iceberg, Power-of-Two, Cuckoo, Double Hash, Chaining + fingerprint variants. 10x+ over locked designs via lock-free vector loads. |
| Hive Hash Table (arXiv Oct 2025) | ✅ Latest | Warp-cooperative, dynamically resizable. WABC + WCME protocols, one atomic per warp. 95% load factor. 1.5-2x over SlabHash/WarpCore. |
| Compact cuckoo hash (Euro-Par 2024) | ✅ | 10-20% throughput over standard cuckoo. Supports insert/delete. |
| cuCollections / cuco (NVIDIA) | ✅ | Open-source production library. `cuco::static_map` with flat memory, `cuda::std::atomic`. |

**Graph Analytics**
| Algorithm | Perf | Library |
|-----------|------|---------|
| BFS (level-synchronous) | ✅ | cuGraph, Gunrock |
| SSSP (delta-stepping, Near-Far) | ✅ | cuGraph, Gunrock — 14-340x over sequential |
| PageRank | ✅ | cuGraph, nvGraph |
| Betweenness Centrality | ✅ | cuGraph (~60 algorithms total) |
| APSP (Floyd-Warshall variant) | ✅ | 40-51x over CPU. Multi-GPU: 64 nodes → 1.66M vertices |
| Connected Components | ✅ | cuGraph |

**Numerics / Signal**
| Operation | Perf | Notes |
|-----------|------|-------|
| Dense GEMM | ✅ | cuBLAS, CUTLASS — near-peak FLOPS |
| Sparse SpMV / SpGEMM | ✅ | cuSPARSE — 30-150x over CPU |
| FFT (CUDA) | ✅ | cuFFT — industry standard, supports 16 GPUs |
| FFT (Vulkan/cross-platform) | ✅ | **VkFFT** — Vulkan + CUDA + HIP + OpenCL + Metal. Highly relevant for OctoFlow. |
| 2D Convolution | ✅ | cuDNN — heavily optimized for ML |
| Prefix scan | ✅ | CUB Blelloch scan — foundation of all parallel primitives |
| K-means clustering | ✅ | cuML (RAPIDS), multiple GPU implementations |
| Algebraic Multigrid (AMG) | ✅ | AmgT (2024): Tensor Core AMG on H100/MI210, 19% over cuSPARSE |

---

### Research Stage (published papers, limited production use)

**Dynamic Trees**
| Structure | Paper | Notes |
|-----------|-------|-------|
| GPU B-Tree | PPoPP 2019 | Supports insert/delete/range. Beats LSM for batch < 100K. Not production-distributed. |
| GPU LSM Tree | arXiv 2017 | 225M inserts/s on K40c. Beats B-Tree for large-batch insert-heavy. Standard LSM lookup tradeoff. |
| GPU B+ Tree | Research | Concurrent structural modifications (splits/merges) expensive. No standard library. |

**Dynamic Graphs**
| Structure | Paper | Notes |
|-----------|-------|-------|
| CSR++ | arXiv Feb 2025 | Combines CSR + adjacency lists. Within 10% of CSR reads, order-of-magnitude faster mutations. |
| GraphVine | arXiv 2023 | Self-managing vertex/edge insertions/deletions without CPU coordination. |
| GPU Merkle Patricia Trie | VLDB 2024 | PhaseNU + LockNU parallel update algorithms. First GPU MPT with meaningful speedup. Blockchain/verifiable DB. |

**Tries and Radix Trees**
| Structure | Paper | Notes |
|-----------|-------|-------|
| CuART (GPU Adaptive Radix Tree) | ICPP 2021 | GPU-optimized ART. Pointer following is latency-bound, not bandwidth-bound — hard on GPU. |
| GPU MPT | VLDB 2024 | See above |

**Priority Queues**
| Structure | Paper | Notes |
|-----------|-------|-------|
| WarpSelect | Research | Merge-based top-K. Limited by shared memory size. Outperformed by RadiK for large K. |
| BGPQ (heap-based) | ICPP 2021 | Heap-based concurrent GPU PQ. Research prototype. |
| Parallel Bucket Heap | arXiv 2019/2023 | Bulk insert support. Cache-efficient variant. |

**Concurrent Queues**
| Structure | Paper | Notes |
|-----------|-------|-------|
| Agile Queue | ICPP Workshops 2024 | Warp-centric enqueue/dequeue. Moves away from CAS-heavy lock-free (contention kills performance). **40x faster** than prior GPU queues. |
| Warp-centric FIFO | Research | Lock-free with amortized atomics per warp. Prototype. |

**Computational Geometry**
| Algorithm | Status | Notes |
|-----------|--------|-------|
| Convex Hull (QuickHull) | ✅ Batch | CudaHull — 30-40x over CPU. Static only. |
| 3D Delaunay Triangulation | ✅ 2025 breakthrough | **First fully GPU-parallel 3D Delaunay** (2025 paper). Massively parallel point insertion + bilateral flipping. 10x over sequential CPU. Previous approaches required CPU post-processing. |
| R-Tree (spatial indexing) | 🔬 Research | Static construction studied. Dynamic insert/delete remains research. MDPI Mathematics 2024. |

**KD-Trees and ANN**
| Structure | Status | Notes |
|-----------|--------|-------|
| Static KD-tree build | ✅ | Near-optimal parallelism at every build stage |
| Buffer KD-tree (batch queries) | 🔬 Research | ICML 2014. Buffers requests at nodes to circumvent pointer-chasing. |
| FAISS (flat + IVF) | ✅ Production | **The** production GPU ANN. Sidesteps tree traversal via quantization + vector search. |
| GPU-HNSWLIB | 🔬 Research | HNSW pointer structure is pathologically bad for GPU (pointer chasing + irregular graph). Active research. |

**String Algorithms**
| Algorithm | Status | Notes |
|-----------|--------|-------|
| Suffix array construction | ✅ | Skew + prefix-doubling hybrid. 7.9x over prior art. Used in GPU bzip2. |
| CUSMART (64 algorithms) | ✅ Batch | Parallelizes 64 string matching algorithms via inter-query parallelism. |
| Smith-Waterman GPU | ✅ | Anti-diagonal wavefront. >70% hardware efficiency. ACM 2023. |

**Dynamic Memory Allocation**
| Allocator | Status | Notes |
|-----------|--------|-------|
| ScatterAlloc | 🔬 Research | Scatters requests across memory regions to reduce contention. |
| Halloc | 🔬 Research | Fast scalable allocator. Available on GitHub. |
| AlignMalloc | 🔬 2025 | Warp-aware rearrangement aligned with UVM prefetching. Addresses large-scale dynamic allocations. |
| CUDA device malloc | 🔴 Slow | Serializes through a global heap. Do not use in performance-critical code. |

---

### Theoretically Known, Poorly Implemented on GPU

| Problem | Sequential | GPU Status | Root Blocker |
|---------|-----------|------------|--------------|
| Optimal comparison sort | O(n log n) | 🟡 Suboptimal | Branching diverges warps; data-dependent memory access destroys coalescing |
| Linked list traversal | O(n) | 🔴 Terrible | Every pointer dereference = full DRAM latency. GPU threads stall completely on each load. |
| Dynamic tree traversal (LCT) | O(log n) | 🔴 Not done | Highly sequential path structure |
| Intra-query parallel KMP/Aho-Corasick | O(n) | 🟡 Partial | Failure function construction has sequential dependencies |
| Work-efficient BFS on power-law graphs | O(V+E) | 🟡 Partial | Degree variance causes warp divergence. WER paper (2024) partially addresses. |
| GPU-native GC | N/A | 🔴 Not done | No preemptive scheduling — a thread that spins prevents others from running |
| Irregular DP (tree DP, DAG DP) | O(n) | 🔴 Not done | Non-wavefront dependency graphs have no GPU parallelism model |
| Online/streaming sort with insertion | O(log n per insert) | 🔴 Not done | Dynamic ordering while parallel |
| Parallel backtracking (SAT, CSP) | Exponential | 🟡 Partial | Work stealing needed; GPU scheduling not preemptive |

---

### Open Problems (unsolved for GPU, 2026)

| Problem | Notes |
|---------|-------|
| Fast GPU device malloc | CUDA device-side malloc serializes. AlignMalloc (2025) helps but no production solution. The single largest blocker for GPU-native dynamic data structures. |
| Pointer-chasing structures | Fundamental SIMT mismatch. Each dereference = 200-800 cycle stall. HBM reduces bandwidth latency, not pointer-chain latency. |
| True wait-free GPU data structures | Lock-free = some thread progresses. Wait-free = every thread progresses in bounded steps. GPU scheduler is not fair/preemptive — spin-wait can deadlock. |
| Persistent GPU data structures | Copy-on-write at GPU scale: no GC, no efficient variable-size allocator, no reference counting at warp granularity. |
| GPU transactional memory (STM) | Research stage only. Contention at GPU scale (4096+ concurrent threads) unsolved. |
| Work-efficient irregular graph algorithms | No solution works well for ALL graph topologies. Load imbalance is architecturally unsolved. |
| Dynamic ANN / streaming ANN index | FAISS is batch-only. Streaming insert + query while maintaining search quality: open problem. |
| Algorithm discovery for GPU ops | AlphaTensor found 8.5% faster matmul. The search space for FFT, sort, BFS, sparse ops is entirely unexplored. |

---

## Architectural Constraints: Why These Problems Are Hard

Five fundamental properties of SIMT GPU architecture explain the entire "not yet" list:

**1. Pointer Chasing is Fatal**
Each pointer dereference incurs full DRAM latency (200-800 cycles). A GPU thread
stalls completely — unlike a CPU superscalar that issues many outstanding loads.
HBM3 on H100 (3+ TB/s) improves bandwidth, not latency. This kills: linked lists,
skip lists, tries, B-trees, HNSW graph traversal.

**2. SIMT Divergence Serializes Warps**
All 32 threads in a warp execute the same instruction. Data-dependent branches
cause divergent threads to execute sequentially. Irregular structures (trees,
power-law graphs) cause severe divergence — threads process 1 neighbor while
others process 1,000,000. Warp Equalizer (WER 2024) partially addresses this.
Algorithms designed around SIMT (radix sort, warp-cooperative hashing, bitonic
sort) dramatically outperform naive parallelizations.

**3. Memory Bandwidth is the Dominant Resource**
Most GPU data structure operations are memory-bound, not compute-bound. Roofline
analysis confirms they sit below the bandwidth ceiling. Every algorithm must
minimize memory traffic and maximize coalescing (adjacent threads access adjacent
memory addresses). Hash tables that guarantee coalesced access patterns
(WarpCore, WarpSpeed) dominate those that don't.

**4. Synchronization Cost Scales with Scope**
- Global synchronization: requires kernel launch (microseconds)
- Block synchronization (`__syncthreads()`): fast
- Warp synchronization (`__ballot_sync`, `__shfl_sync`): register-speed

Modern algorithms push coordination to warp level. Warp-level primitives
(`ballot`, `shfl`, `vote`, `match`) are the key to 2025-era GPU data structures.
Hardware support for these (Vortex RISC-V GPU, 2025) achieves ~4x speedup over
software emulation. Anything requiring global coordination pays a large penalty.

**5. Dynamic Allocation is Broken**
CUDA device-side `malloc()` serializes through a global heap — catastrophic
for concurrent allocations from 4096+ threads. This is the single largest
blocker for GPU-native dynamic data structures. All research alternatives
(ScatterAlloc, Halloc, AlignMalloc) address parts but none solve it comprehensively.

---

## Key References (2020-2025)

- WarpSpeed: 8 GPU hash table designs (arXiv Sep 2025)
- Hive Hash Table: warp-cooperative resizable hash (arXiv Oct 2025)
- Compact Parallel Hash Tables (Euro-Par 2024)
- WarpCore (HiPC 2020)
- Engineering a GPU B-Tree (PPoPP 2019)
- RadiK: radix Top-K selection (ICS 2024)
- Agile Queue (ICPP Workshops 2024)
- 3D Delaunay on GPU (2025 — first fully parallel)
- GPU Merkle Patricia Trie (VLDB 2024)
- CuART (ICPP 2021)
- CSR++ dynamic graph representation (arXiv Feb 2025)
- GraphVine (arXiv 2023)
- PASGAL graph library (UCR 2024)
- GPU Load Balancing abstraction (PPoPP 2023)
- WER Warp Equalizer for irregular graphs (2024)
- AlignMalloc (2025)
- VkFFT: cross-platform Vulkan/CUDA/HIP/OpenCL/Metal FFT
- AmgT: Tensor Core AMG solver (2024)
- AlphaTensor: algorithm discovery (Nature 2022)
- AlphaSparseTensor (2024)
- Hardware warp-level primitives in Vortex RISC-V GPU (arXiv 2025)

---

## The Self-Discovery Loop

This is the long-range payoff — and it is not theoretical.

AlphaTensor proved it: GPU compute can discover better algorithms than humans
have found in 50 years. The method is:
1. Represent an algorithm as a mathematical object (tensor decomposition, graph, etc.)
2. Define a reward signal (fewer operations, lower latency, better GPU throughput)
3. Run RL or evolutionary search over the algorithm space — on GPU
4. Evaluate candidates — on GPU
5. Converge to algorithms humans did not find

OctoFlow post-Phase-52 is exactly the stack needed to run this loop:

```
// OctoFlow self-improvement sketch (post Phase 52):
use octoparallel
use neural_network

let candidates = generate_algorithm_variants(matmul_kernel, 1000)
let scores = map_each(candidates, fn(c) {
    benchmark_gpu(c, test_tensors)      // GPU evaluates GPU algorithms
})
let best = sort_by(candidates, fn(a, b) { scores[a] > scores[b] })
let next = mutate_crossover(best, 50)
// ... iterate
```

The program that improves GPU algorithms runs ON the GPU it is improving.
No other language can do this with zero external dependencies, in a < 2MB binary,
on any GPU (AMD/Intel/NVIDIA/future).

The research catalogue in this document is the seed list — the known algorithms
that OctoFlow will validate, reimplement in pure `.flow`, and eventually improve
upon through algorithmic discovery.

---

## Why This Domain Exists After Self-Hosting

Before Phase 52: These algorithms would be Rust code in the bootstrap.
After Phase 52: These algorithms are `.flow` files that compile to `.fgb`.

The difference:
- Pre-self-hosting: adding GPU k-means means writing Rust + SPIR-V
- Post-self-hosting: adding GPU k-means means writing `kmeans.flow`

That's why OctoParallel waits. The domain should be written in the language,
not bolted onto the bootstrap. When OctoFlow compiles OctoFlow, the standard
library becomes a living thing that OctoFlow programs can extend.

---

## The Quantum Bridge

OctoParallel's parallel primitives map directly to quantum algorithms:

| Classical (GPU parallel) | Quantum equivalent |
|--------------------------|-------------------|
| Parallel search O(√n × n) | Grover's search O(√n) |
| FFT O(n log n) | Quantum Fourier Transform O(log²n) |
| Sampling / Monte Carlo | Quantum amplitude estimation |
| Linear system solve | HHL algorithm (exponential speedup for sparse) |
| Matrix multiply | Quantum matrix multiplication (open problem) |

OctoParallel's abstraction layer — `dispatch_compute(kernel, data)` — maps
identically to quantum circuit submission. When quantum hardware matures,
OctoParallel gains quantum backends without changing the API.

---

## Implementation Priority (post-Phase 52)

```
Phase 53: Tensor primitives + matmul (foundation for Annex M rewrite)
Phase 54: Linear algebra — SVD, solve, eigenvalues
Phase 55: Parallel DSA — gpu_sort, prefix_scan, gpu_hash, parallel BFS
Phase 56: Signal processing — FFT, convolve, convolve2d
Phase 57: Statistics — PCA, k-means, covariance
```

Each phase produces `.flow` modules that other domains can `use`.
The GPU SPIR-V kernels for each operation are compiled from OctoFlow itself.

---

*"The math layer written once, in the language, for every domain."*
