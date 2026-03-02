# OctoFlow GPU VM — Workload Classes (Follow-Up Context)

The GPU VM is not an inference runtime. It is a **general-purpose dispatch graph executor**. Transformer inference is one application. Any workload that fits the pattern — structured data in GPU memory, parallel kernel chains, inter-stage communication, runtime self-monitoring — runs on the same VM with no architectural changes.

The primitives are workload-agnostic:
- **Registers (binding 0):** Per-stage working memory
- **Globals (binding 2):** Shared mutable state (KV cache, DB tables, weights, parameters)
- **Control (binding 3):** Runtime adaptation — regulator comms + indirect dispatch params
- **Heap (binding 4):** Persistent immutable bulk data (quantized weights, embeddings, compressed datasets)
- **R30→R31:** Inter-stage message passing with metadata
- **Dispatch chain:** Ordered kernel sequence with barriers

What changes per workload: the kernels in the chain and the Heap layout. Nothing else.

---

## Workload Dispatch Patterns

### Attention (Transformer Inference)
- **Heap:** Quantized weight matrices, KV cache in Globals
- **Dispatch chain:** Dequant → Q projection → K projection → QK matmul → Softmax → V projection → Output projection
- **Message (R30):** Hidden state + activation norm + anomaly flag
- **Regulator role:** Activation stability scaling, adaptive quantization per layer

### Graph Neural Networks
- **Heap:** Adjacency lists/matrices (compressed sparse), node feature vectors, edge weights
- **Dispatch chain:** Node embedding lookup → Neighbor aggregation (message passing) → Feature transform → Activation → Readout/pooling
- **Message (R30):** Updated node embeddings + convergence metric + iteration count
- **Regulator role:** Over-smoothing detection (if node features converge too aggressively, signal early termination or residual scaling). Track message magnitude decay across GNN layers.
- **VM mapping:** Each GNN layer = one VM instance. For heterogeneous graphs, multiple VM instances per layer (one per edge type) with Heap-resident routing tables.

### Physics Simulation
- **Heap:** Particle positions, velocities, masses, spatial hash grid / BVH
- **Dispatch chain:** Spatial indexing rebuild → Neighbor search → Force accumulation → Integration (Verlet/RK4) → Collision detection → Constraint resolution
- **Message (R30):** Updated state vector + max velocity + energy drift + constraint violation count
- **Regulator role:** Energy conservation monitoring (if total energy drifts beyond threshold, signal timestep reduction via Control word). Detect runaway particles (max velocity anomaly).
- **VM mapping:** Each simulation substep = one VM instance. For multi-physics (fluid + rigid body), separate VM instance chains with message passing at coupling points.

### Database Query Plans
- **Globals:** Mutable columnar table storage, hash indexes, B-tree nodes, counters. Globals (binding 2) is the shared mutable SSBO — supports inserts/deletes/updates. Heap (binding 4) for immutable compressed lookup tables and precomputed bloom filters.
- **Dispatch chain:** Index lookup → Fused scan+filter (vm_where family, dual output: masked values + mask) → Join (hash probe, same fused pattern) → Aggregate (parallel reduction → Metrics) → Sort (bitonic) → Output marshal
- **Message (R30):** Result set reference (Globals offset + row count) + error code + rows scanned. Scalar aggregates (SUM, COUNT) in Metrics — naturally feed regulator.
- **Regulator role:** Cardinality monitoring (if intermediate result sets blow up unexpectedly, signal plan adaptation — switch from nested loop to hash join). Track scan-to-output ratio for query optimization feedback. COUNT=0 triggers anomaly signal via Metrics.
- **VM mapping:** Each query stage = one VM instance. Subqueries spawn nested VM chains. Joins connect two input VM chains into one output. Indirect dispatch (Step 7) enables adaptive query plans — scheduler reads intermediate row counts and adjusts aggregation workgroup size.

### Game AI (Decision & Behavior)
- **Heap:** Game state tensor (board/world state), evaluation weights, Monte Carlo tree nodes, behavior tree definitions
- **Dispatch chain:** State encoding → Feature extraction → Tree search / Policy evaluation → Move scoring → Action selection → State update
- **Message (R30):** Selected action + confidence score + search depth reached + novelty flag
- **Regulator role:** Time budget enforcement (if compute ticks exceed allocation, signal early termination with best-so-far action). Detect degenerate search (all moves score equally → signal exploration boost).
- **VM mapping:** Each search depth level = one VM instance. For multi-agent games, parallel VM chains per agent with shared Heap (game state) and isolated registers (private strategy).

### DSP Pipelines (Audio/Signal Processing)
- **Heap:** Sample buffers (circular), filter coefficients, FFT twiddle factors, spectral data
- **Dispatch chain:** Input windowing → FFT → Spectral processing (filter/convolve/denoise) → IFFT → Overlap-add → Output marshal
- **Message (R30):** Processed frame + peak amplitude + clipping flag + latency ticks
- **Regulator role:** Clipping prevention (if peak amplitude approaches ceiling, signal gain reduction to downstream stages). Latency monitoring (if a stage exceeds tick budget, signal quality reduction — lower FFT resolution or skip non-essential processing).
- **VM mapping:** Each processing stage = one VM instance. For multi-channel audio, parallel VM chains per channel with shared Heap (shared coefficients) and isolated registers (per-channel state).

---

## Cross-Workload Composition

The real power: these aren't siloed. Because they share the same VM primitives, workloads compose within a single command buffer.

**Agent + RAG + Reasoning:**
LLM VM chain → DB VM (vector retrieval) → LLM VM chain continues with retrieved context. All one `vkQueueSubmit`.

**Game AI + Physics:**
AI VM chain selects action → Physics VM chain simulates result → AI VM chain evaluates outcome. The R30 message from physics carries state + energy metrics directly into the AI chain's R31.

**GNN + Database:**
DB VM scans for relevant subgraph → GNN VM processes it → Results written to R30 for downstream consumption. Graph structure and query results live in the same Heap.

**LLM + DSP (multimodal):**
Audio DSP VM chain extracts features → Feature vector lands in R30 → LLM VM chain reads it as input embedding via R31. Speech understanding without CPU-mediated feature transfer.

The dispatch graph is the universal connector. If a workload produces output in R30 and another consumes input from R31, they compose. The regulator doesn't care what the kernels do — it monitors the same signals (norms, ticks, anomaly flags) regardless.

---

## Architectural Implications for Implementation

When building VM features, always ask: **"Does this decision work for all six workload classes, or am I baking in a transformer assumption?"**

Specific checkpoints:
- **Register width:** Must accommodate GNN embeddings (variable), physics state vectors (3-6 floats per particle × thousands), DSP frames (512-4096 samples), not just transformer hidden dims
- **Heap layout:** Must support columnar storage (DB), sparse matrices (GNN), spatial grids (physics), not just dense weight matrices
- **Regulator signals:** The 4-field metadata (norm, max_abs, ticks, status) is generic enough for all workloads. Don't add transformer-specific fields — use the status field + Control words for workload-specific signaling
- **Barrier granularity:** Physics sims may need sub-step barriers within a single VM instance (force accumulation must complete before integration). The barrier model must support intra-VM barriers, not just inter-VM
- **Indirect dispatch (step 7):** Game AI's early termination and DB's adaptive query plans both need this. Don't design it as "adaptive LLM compute" — design it as "conditional dispatch based on Control word state"

---

## Design Principle

The GPU VM succeeds when adding a new workload requires only:
1. Writing new kernels
2. Defining a Heap layout
3. Plugging them into `vm_program()`

No changes to the VM core. No new SSBO bindings. No new message format. The architecture is complete when it disappears — when the workload author thinks only about their domain, not about the VM.
