# OctoFlow — Annex L: HyperGraphDB — GPU-Native Hypergraph Database for AI/ML

**Parent Document:** OctoFlow Strategic Vision
**Status:** Architecture Specification
**Version:** 0.1
**Date:** February 17, 2026

---

## The Positioning

```
In Python:
  SQLite    = simple embedded analytics database
  PyTorch   = neural network framework with tensor operations

In OctoFlow:
  OctoDB         = simple GPU-native analytics database (Annex R)
  HyperGraphDB   = GPU-native hypergraph for AI/ML (THIS DOCUMENT)
  Neural Network = built on HyperGraphDB primitives (Annex M)
```

**OctoDB** is for when you need a lightweight database for analytics — fast scans, time-travel queries, append-only simplicity.

**HyperGraphDB** is for when you need to represent knowledge, relationships, and neural networks — where the data structure IS the computation graph.

---

## Table of Contents

1. Thesis
2. What Is a Hypergraph Database?
3. Why GPU Changes Everything
4. The Core Architecture
5. Storage: Sparse Matrices on GPU
6. The Incidence Matrix as Universal Primitive
7. Fundamental Operations (6 GPU Kernels)
8. Database Queries as Hypergraph Traversal
9. Neural Networks as Hypergraph Operations
10. The Type System (Polymorphic Entity-Relation)
11. Schema Definition in .flow
12. Example: Knowledge Graph
13. Example: GNN Training
14. Example: Transformer Attention
15. Threading Model: Hyperedges as Units of Work
16. Comparison to Existing Systems
17. Implementation Roadmap
18. Integration with OctoFlow Ecosystem

---

## 1. Thesis

**A hypergraph database and a neural network framework are the same thing.**

Both are:
- Sparse graph structures where operations propagate signals across edges
- Representable as sparse matrices (incidence matrix)
- Executed via the same GPU kernels (SpMM - Sparse Matrix-Matrix Multiply)
- Trained/queried via the same pattern (gather → transform → scatter)

The difference is semantic, not architectural:
- **Database query:** pattern matching on hyperedge membership → retrieve vertices
- **Neural network:** message passing on hyperedge membership → compute features

OctoFlow's HyperGraphDB unifies both. The incidence matrix B (vertices × hyperedges) is:
1. How the graph is stored
2. How queries execute
3. How neural networks propagate
4. How gradients backpropagate
5. How threading parallelizes work

**One data structure. One set of kernels. Everything else is composition.**

---

## 2. What Is a Hypergraph Database?

### Standard Graph vs Hypergraph

```
Standard Graph (Neo4j, ArangoDB):
  Edge connects exactly 2 nodes:
    (Person:Alice) --WORKS_AT--> (Company:Acme)

Hypergraph (TypeDB, HyperGraphDB):
  Hyperedge connects N nodes (N ≥ 2):
    TRANSACTION {Alice, Bob, BankA, BankB, Contract42, DateTime}
    All 6 nodes participate in ONE relationship
```

### Why Hyperedges Matter for AI/ML

1. **N-ary relationships are native**
   - Multi-party transactions, co-authorship, protein complexes
   - No reification (creating intermediate nodes) needed

2. **Higher-order structure**
   - Transformer attention: one query attends to ALL keys (hyperedge)
   - Convolution: one output depends on ALL kernel inputs (hyperedge)
   - Knowledge graphs: predicate connects subject + object + context

3. **Tensor alignment**
   - A k-uniform hyperedge = k-th order tensor
   - Regular edge (k=2) = matrix
   - Attention head (k=sequence_length) = attention matrix
   - Hypergraph operations = tensor operations

4. **Meta-modeling**
   - Relationships can participate in other relationships
   - Types, constraints, provenance on edges themselves

### Existing Hypergraph Databases

**TypeDB** (production, Rust/Java):
- Polymorphic Entity-Relation-Attribute (PERA) model
- TypeQL query language (ACM SIGMOD 2024 Best Newcomer Award)
- Relations are first-class objects
- Storage: RocksDB (CPU-based)

**HyperGraphDB** (open-source, Java):
- Oldest implementation (mid-2000s, actively maintained)
- Storage: BerkeleyDB
- Designed for AI and semantic web

**Gap:** Both are CPU-based. No GPU-native hypergraph database exists.

---

## 3. Why GPU Changes Everything

### The Fundamental Kernel: SpMM

Every hypergraph operation reduces to **Sparse Matrix-Matrix Multiply (SpMM)**:

```
B: incidence matrix (vertices × hyperedges)
   B[i,j] = 1 if vertex i is in hyperedge j, else 0

Scatter (vertex → hyperedge):
  messages = B^T @ vertex_features
  Each hyperedge aggregates features from its member vertices
  Parallel: one CUDA thread block per hyperedge

Gather (hyperedge → vertex):
  new_features = B @ edge_messages
  Each vertex aggregates messages from incident hyperedges
  Parallel: one CUDA thread block per vertex
```

### Why This Is Fast on GPU

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| SpMM (100M edges) | 1000ms | 10ms | 100x |
| BFS (graph traversal) | 500ms | 10ms | 50x |
| PageRank (iterative SpMM) | 2000ms | 20ms | 100x |
| GNN forward pass (SpMM + GEMM) | 1500ms | 15ms | 100x |

**NVIDIA cuGraph** achieves these speedups on real graphs. The secret: sparse matrix stored in CSR (Compressed Sparse Row) format on GPU device memory, with coalesced memory access during neighbor gathering.

### The Killer App: Query and Train Are the Same

```flow
// Database query: find all papers by authors in "Stanford"
let result = match_pattern(
    graph,
    (?paper, ?author, AUTHORED_BY),
    (?author, "Stanford", AFFILIATED_WITH)
)

// Internally: BFS traversal via gather/scatter
// Kernel: SpMM on incidence matrix B

// Neural network: one GNN layer
fn gnn_layer(features: Tensor, graph: HyperGraph, weights: Tensor) -> Tensor {
    let messages = scatter(features, graph.B)     // SpMM: B^T @ features
    let aggregated = gather(messages, graph.B)    // SpMM: B @ messages
    return relu(aggregated @ weights)             // GEMM + activation
}

// Internally: message passing via gather/scatter
// Kernel: SpMM on incidence matrix B (THE SAME KERNEL)
```

Query execution and neural network training use the **same GPU kernels** on the **same data structure**. This is not an analogy. It is architectural identity.

---

## 4. The Core Architecture

### Two-Layer Architecture: Persistent + Hot

**Layer 1: Persistent Storage (Disk)**

```flow
struct HyperGraphDB {
    // ON DISK (.hypergraph file, append-only format)
    file_path: string,

    // Persistent graph topology
    topology: PersistentTopology {
        vertices: [VertexID],          // all vertex IDs
        hyperedges: [HyperedgeID],     // all edge IDs
        incidence: [(VertexID, HyperedgeID)],  // membership list
        vertex_types: [(VertexID, TypeID)],
        edge_types: [(HyperedgeID, TypeID)],
        vertex_attrs: [(VertexID, key, value)],
        edge_attrs: [(HyperedgeID, key, value)]
    },

    // Versioned embeddings (saved after training)
    embedding_versions: [EmbeddingSnapshot {
        version: string,
        timestamp: u64,
        embeddings: [f32; num_vertices × embedding_dim]
    }],

    // Model checkpoints (saved periodically)
    checkpoints: [Checkpoint {
        epoch: u32,
        model_weights: Map<string, Tensor>
    }]
}
```

**Layer 2: Hot In-Memory (GPU)**

```flow
struct HyperGraphGPU {
    // LOADED FROM DISK → GPU DEVICE MEMORY
    B: SparseMatrix,              // incidence matrix (CSR format on GPU)
    vertex_features: Tensor,      // [num_vertices, feature_dim]
    edge_features: Tensor,        // [num_hyperedges, feature_dim]
    type_info: Tensor,            // type IDs per vertex/edge

    // HOT (allocated during training, not persistent)
    model_weights: Map<string, Tensor>,   // W1, W2, ... (in GPU memory)
    gradients: Map<string, Tensor>,       // dL/dW1, dL/dW2, ...
    optimizer_state: AdamState,           // m, v moments
    gradient_tape: [Operation],           // recorded ops for backprop

    // Scratch space for intermediate results
    scratch: [Tensor]
}
```

### Load/Save Workflow

```flow
// Load from disk to GPU (one-time per session)
fn load_to_gpu(db_path: string) -> HyperGraphGPU {
    // 1. Read .hypergraph file from disk
    let db = hypergraph.load_persistent(db_path)

    // 2. Build CSR incidence matrix from edge list
    let B = build_csr_from_edges(db.topology.incidence)

    // 3. Upload to GPU device memory
    let B_gpu = upload_sparse_matrix(B)
    let features_gpu = upload_tensor(db.topology.vertex_attrs)

    return HyperGraphGPU {
        B: B_gpu,
        vertex_features: features_gpu,
        // ... rest initialized empty
    }
}

// Save from GPU to disk (after training)
fn save_to_disk(graph_gpu: HyperGraphGPU, db_path: string, version: string) {
    // 1. Download embeddings from GPU to CPU
    let embeddings_cpu = download_tensor(graph_gpu.vertex_features)

    // 2. Append new embedding version to .hypergraph file
    hypergraph.append_embedding_snapshot(
        db_path,
        version,
        timestamp=now(),
        embeddings=embeddings_cpu
    )

    // Graph topology is NOT rewritten (it's immutable)
    // Only embeddings and checkpoints are appended
}
```

### Incidence Matrix Format (CSR on GPU)

```
Compressed Sparse Row (CSR):
  row_ptr[V+1]   -- for each vertex, offset into col_ind
  col_ind[E]     -- which hyperedges each vertex belongs to
  values[E]      -- optional edge weights

Example:
  Vertex 0: member of hyperedges [1, 3, 5]
  Vertex 1: member of hyperedges [1, 2]
  Vertex 2: member of hyperedges [3, 4, 5]

  row_ptr  = [0, 3, 5, 8]
  col_ind  = [1, 3, 5,  1, 2,  3, 4, 5]
  values   = [1, 1, 1,  1, 1,  1, 1, 1]  (or weights)

GPU advantage:
  - Coalesced memory access during gather/scatter
  - Warp-level reduction for neighbor aggregation
  - Kernel fusion: gather + compute + scatter in one kernel
```

---

## 5. Storage: Persistent vs In-Memory

### Critical Distinction

**HyperGraphDB = persistent database (disk-backed, durable)**
**Neural Network = hot in-memory computation (ephemeral, GPU device memory)**

They work hand-in-hand:

```
PERSISTENT (HyperGraphDB on disk):
├── Graph topology (incidence matrix B)
│   └── Stored in .hypergraph file (append-only, like OctoDB)
├── Vertex/edge metadata (types, IDs, attributes)
├── Historical embeddings (versioned)
└── Training checkpoints

IN-MEMORY (Neural Network during training):
├── Current feature tensors (GPU device memory)
├── Model weights (GPU device memory)
├── Gradients (GPU device memory)
├── Optimizer state (Adam moments, etc.)
└── Gradient tape (operation history)
```

### The Workflow

```flow
// 1. Load graph from persistent storage
let graph = hypergraph.load("knowledge_graph.hypergraph")
// Disk → RAM → GPU: incidence matrix B loaded to GPU device memory

// 2. Train neural network (hot, in-memory on GPU)
let mut model = init_gnn(input_dim, hidden_dim, num_classes)
for epoch in range(num_epochs)
    let embeddings = train_gnn(graph, model)
    // All tensors live in GPU memory during training
end

// 3. Save results back to persistent storage
hypergraph.save_embeddings(graph, embeddings, version=epoch)
hypergraph.save_checkpoint(model, "model_checkpoint_{epoch}.ckpt")
// GPU → RAM → Disk: persist learned embeddings for later queries
```

### Storage Architecture

```
┌─────────────────────────────────────────────────────┐
│                       DISK                           │
│                                                      │
│  .hypergraph file (persistent, append-only):         │
│  ├── Graph topology (incidence matrix B, CSR)        │
│  ├── Vertex metadata (types, attributes)             │
│  ├── Edge metadata (types, attributes)               │
│  ├── Embedding snapshots (versioned, timestamped)    │
│  └── Model checkpoints (.ckpt files)                 │
│                                                      │
└──────────────────┬───────────────────────────────────┘
                   │ hypergraph.load()
                   ↓
┌─────────────────────────────────────────────────────┐
│                    GPU DEVICE MEMORY                 │
│                                                      │
│  LOADED (from disk):                                 │
│  ├── Incidence matrix B (CSR on GPU)                 │
│  ├── Initial vertex features                         │
│  ├── Type info                                       │
│                                                      │
│  HOT (allocated during training):                    │
│  ├── Model weights (W1, W2, ...)                     │
│  ├── Gradients (dL/dW1, dL/dW2, ...)                 │
│  ├── Optimizer state (Adam m, v)                     │
│  ├── Intermediate activations                        │
│  ├── Gradient tape operations                        │
│  └── Scratch buffers                                 │
│                                                      │
└──────────────────┬───────────────────────────────────┘
                   │ hypergraph.save_embeddings()
                   ↓
┌─────────────────────────────────────────────────────┐
│                       DISK                           │
│  New embedding version appended to .hypergraph file  │
└─────────────────────────────────────────────────────┘
```

### Multi-GPU Scaling (Phase 52+)

**WholeGraph pattern** (NVIDIA):
- Sparse matrix B sharded across GPUs via row-partitioning
- Persistent storage remains on disk (single source of truth)
- During training: B loaded to multi-GPU device memory
- After training: embeddings written back to disk

---

## 6. The Incidence Matrix as Universal Primitive

### All Operations Decompose to SpMM

| Operation | Matrix Form | GPU Kernel |
|-----------|-------------|------------|
| Scatter (vertex → edge) | `B^T @ vertex_features` | SpMM (transpose) |
| Gather (edge → vertex) | `B @ edge_messages` | SpMM |
| BFS traversal (query) | `B @ (B^T @ frontier)` | SpMM twice |
| PageRank iteration | `A @ scores` where `A = B @ B^T` | SpMM |
| GNN forward pass | `ReLU(B @ (B^T @ H) @ W)` | SpMM + GEMM |
| GNN backward pass | `B^T @ (B @ grad_H)` | SpMM (transposed) |
| Attention (transformers) | `softmax(QK^T/√d) @ V` | GEMM (dense) |
| Hypergraph convolution | `D^{-1/2} @ B @ W @ B^T @ D^{-1/2} @ H` | SpMM chain |

**The insight:** SpMM is to hypergraphs what GEMM is to dense neural networks — the single kernel that everything reduces to.

---

## 7. Fundamental Operations (6 GPU Kernels)

### Kernel 1: Scatter (Vertex → Hyperedge)

```flow
fn scatter(vertex_features: Tensor, B: SparseMatrix) -> Tensor {
    // Each hyperedge aggregates features from its member vertices
    // Parallel: one warp per hyperedge
    // Output: [num_hyperedges, feature_dim]
    return spmm_transpose(B, vertex_features)
}
```

**GPU implementation:** cuSPARSE `cusparseSpmm` with `B^T`.

### Kernel 2: Gather (Hyperedge → Vertex)

```flow
fn gather(edge_messages: Tensor, B: SparseMatrix) -> Tensor {
    // Each vertex aggregates messages from incident hyperedges
    // Parallel: one warp per vertex
    // Output: [num_vertices, feature_dim]
    return spmm(B, edge_messages)
}
```

**GPU implementation:** cuSPARSE `cusparseSpmm` with `B`.

### Kernel 3: Message Passing (Gather → Transform → Scatter)

```flow
fn message_pass(
    features: Tensor,
    B: SparseMatrix,
    transform: fn(Tensor) -> Tensor
) -> Tensor {
    let messages = scatter(features, B)      // vertex → edge
    let transformed = transform(messages)    // arbitrary function
    return gather(transformed, B)            // edge → vertex
}
```

This is the **universal pattern** for GNNs, hypergraph convolutions, and graph queries.

### Kernel 4: Pattern Matching (BFS Traversal)

```flow
fn match_pattern(
    graph: HyperGraph,
    start_vertices: [VertexID],
    pattern: HyperedgePattern
) -> [VertexID] {
    let mut frontier = start_vertices
    let mut visited = set()

    while len(frontier) > 0
        // Expand frontier via hyperedges matching pattern
        let edge_frontier = scatter(frontier, graph.B)
        edge_frontier = filter_by_type(edge_frontier, pattern.type)
        frontier = gather(edge_frontier, graph.B)
        frontier = difference(frontier, visited)
        visited = union(visited, frontier)
    end

    return visited
}
```

**GPU implementation:** Level-synchronous BFS with frontier queues in device memory.

### Kernel 5: Aggregation (Reduce)

```flow
fn aggregate(vertex_features: Tensor, op: ReduceOp) -> f32 {
    // Reduce all vertex features to a scalar
    // ops: sum, mean, max, min
    // Parallel: tree reduction on GPU
    return reduce(vertex_features, op)
}
```

**GPU implementation:** cuBLAS reduction or custom CUDA kernel with shared memory.

### Kernel 6: Dense Transform (GEMM)

```flow
fn dense_transform(features: Tensor, weights: Tensor) -> Tensor {
    // Standard dense matrix multiply
    // Used for: learned projections in GNNs, attention mechanisms
    return gemm(features, weights)
}
```

**GPU implementation:** cuBLAS `cublasSgemm` (FP32) or Tensor Cores (FP16/INT8).

---

## 8. Database Queries as Hypergraph Traversal

### TypeQL-Inspired Pattern Matching

```flow
// Find all papers co-authored by people at Stanford and MIT
let query = pattern(
    (?paper, ?author1, ?author2, AUTHORED_BY),
    (?author1, "Stanford", AFFILIATED_WITH),
    (?author2, "MIT", AFFILIATED_WITH)
)

let results = match(graph, query)
```

**Execution plan:**
1. Find vertices with `AFFILIATED_WITH "Stanford"` → frontier1
2. Find vertices with `AFFILIATED_WITH "MIT"` → frontier2
3. Find hyperedges of type `AUTHORED_BY` connecting frontier1 ∪ frontier2
4. Return member vertices of type `?paper`

**GPU kernels:** Pattern matching = series of scatter/gather operations filtered by type.

### Recursive Queries (Transitive Closure)

```flow
// Find all nodes reachable from root via "INFLUENCES" edges
fn transitive_closure(graph: HyperGraph, root: VertexID, edge_type: TypeID) -> [VertexID] {
    let mut frontier = [root]
    let mut visited = set()

    while len(frontier) > 0
        let messages = scatter(frontier, graph.B)
        messages = filter_by_type(messages, edge_type)
        frontier = gather(messages, graph.B)
        frontier = difference(frontier, visited)
        visited = union(visited, frontier)
    end

    return visited
}
```

This is **exactly the GNN forward pass pattern** — layer-by-layer propagation through the graph.

---

## 9. Neural Networks as Hypergraph Operations

### Graph Convolutional Network (GCN) Layer

```flow
fn gcn_layer(
    features: Tensor,        // [num_vertices, feature_dim]
    graph: HyperGraph,
    weights: Tensor          // [feature_dim, output_dim]
) -> Tensor {
    // GCN: H' = σ(D^{-1/2} A D^{-1/2} H W)
    // Simplified hypergraph version:

    let messages = scatter(features, graph.B)      // B^T @ H
    let aggregated = gather(messages, graph.B)     // B @ messages
    let transformed = gemm(aggregated, weights)    // @ W
    return relu(transformed)                       // σ(·)
}
```

**This is 4 lines.** The entire GCN architecture is scatter → gather → GEMM → activation.

### Hypergraph Attention (Transformer-Style)

```flow
fn hypergraph_attention(
    features: Tensor,        // [num_vertices, d_model]
    graph: HyperGraph,
    W_Q: Tensor, W_K: Tensor, W_V: Tensor
) -> Tensor {
    // Compute Q, K, V projections
    let Q = gemm(features, W_Q)
    let K = gemm(features, W_K)
    let V = gemm(features, W_V)

    // Scatter Q to hyperedges, gather K from vertices
    let Q_edges = scatter(Q, graph.B)
    let K_edges = scatter(K, graph.B)
    let V_edges = scatter(V, graph.B)

    // Attention within each hyperedge (dense)
    let attention_weights = softmax(Q_edges @ K_edges.T / sqrt(d_model))
    let attended = attention_weights @ V_edges

    // Gather back to vertices
    return gather(attended, graph.B)
}
```

**Transformers are hypergraph operations.** Each attention head is a hyperedge connecting query to all keys.

### Full Neural Network (End-to-End)

```flow
fn neural_network(
    input_features: Tensor,
    graph: HyperGraph,
    layers: [Layer]
) -> Tensor {
    let mut h = input_features

    for layer in layers
        match layer.type
            "gcn" =>
                h = gcn_layer(h, graph, layer.weights)
            "attention" =>
                h = hypergraph_attention(h, graph, layer.W_Q, layer.W_K, layer.W_V)
            "dense" =>
                h = relu(gemm(h, layer.weights))
        end
    end

    return h
}
```

Training is the same pattern — forward pass (scatter/gather), compute loss, backward pass (transpose of scatter/gather).

---

## 10. The Type System (Polymorphic Entity-Relation)

### TypeDB's PERA Model (Adapted for GPU)

```flow
// Type hierarchy
type Person sub Entity
type Author sub Person
type Researcher sub Person

type Paper sub Entity
type Conference sub Entity

type AuthoredBy sub Relation
    relates author: Person
    relates paper: Paper

type PublishedAt sub Relation
    relates paper: Paper
    relates venue: Conference
    relates year: u32
```

### Storage: Type IDs on GPU

```flow
struct TypeInfo {
    vertex_types: [u32; num_vertices],     // type ID per vertex
    edge_types: [u32; num_hyperedges],     // type ID per hyperedge
    type_hierarchy: SparseMatrix,          // subtype relationships (DAG)
}

// Type filtering during query/training
fn filter_by_type(vertices: [VertexID], type_id: u32, types: TypeInfo) -> [VertexID] {
    // Parallel: one thread per vertex
    return filter(vertices, fn(v) types.vertex_types[v] == type_id end)
}
```

This enables **polymorphic queries** — match against a supertype, retrieve all subtypes.

---

## 11. Schema Definition in .flow

### Example Schema

```flow
// Define types
type Person {
    name: string,
    age: f32
}

type Paper {
    title: string,
    year: u32
}

// Define hyperedge types
type AuthoredBy {
    relates author: Person,
    relates paper: Paper,
    year: u32
}

// Create instances
let alice = Person { name: "Alice", age: 35.0 }
let bob = Person { name: "Bob", age: 40.0 }
let paper1 = Paper { title: "GPU Hypergraphs", year: 2026 }

// Create hyperedge
let authorship = AuthoredBy {
    author: [alice, bob],
    paper: [paper1],
    year: 2026
}

// The incidence matrix B is built automatically from these declarations
```

---

## 12. Example: Knowledge Graph

### Building a Knowledge Graph

```flow
// Define entities
let stanford = Institution { name: "Stanford", type: "University" }
let alice = Person { name: "Alice", affiliation: stanford }
let bob = Person { name: "Bob", affiliation: stanford }
let paper = Paper { title: "Hypergraphs for AI", year: 2026 }

// Define relationships (hyperedges)
let authorship = AuthoredBy { authors: [alice, bob], paper: paper }
let citation = Cites { citing: paper, cited: some_other_paper }

// Query: find all papers by Stanford authors
let stanford_authors = match(
    (?person, stanford, AFFILIATED_WITH)
)
let stanford_papers = match(
    (?paper, ?author, AUTHORED_BY)
    where ?author in stanford_authors
)
```

### Neural Knowledge Graph Embeddings

```flow
// Train embeddings via GNN on the knowledge graph
fn train_embeddings(graph: HyperGraph, num_epochs: u32) -> Tensor {
    let mut embeddings = random_init(graph.num_vertices, embedding_dim)

    for epoch in range(num_epochs)
        // Forward pass: 2-layer GCN
        let h1 = gcn_layer(embeddings, graph, W1)
        let h2 = gcn_layer(h1, graph, W2)

        // Loss: link prediction
        let loss = link_prediction_loss(h2, graph)

        // Backward pass (automatic via gradient tape)
        let grads = backward(loss)

        // Update weights
        W1 = W1 - learning_rate * grads.W1
        W2 = W2 - learning_rate * grads.W2
    end

    return embeddings
}
```

The knowledge graph IS the training data. The hyperedges define which embeddings interact.

---

## 13. Example: GNN Training

### Node Classification

```flow
// Train GNN for node classification (e.g., paper topic classification)
fn train_gnn(
    graph: HyperGraph,
    labels: [u32; num_vertices],
    train_mask: [bool; num_vertices]
) -> Model {
    let model = init_model(input_dim, hidden_dim, num_classes)

    for epoch in range(num_epochs)
        // Forward pass
        let logits = model.forward(graph.vertex_features, graph)

        // Loss (only on training nodes)
        let loss = cross_entropy(logits[train_mask], labels[train_mask])

        // Backward pass
        let grads = backward(loss)

        // Update
        model.update(grads, learning_rate)

        // Validation
        let val_acc = accuracy(logits[val_mask], labels[val_mask])
        print("Epoch {epoch}, Loss: {loss:.4}, Val Acc: {val_acc:.2}")
    end

    return model
}
```

---

## 14. Example: Transformer Attention

### Self-Attention as Hypergraph

```flow
// Each attention operation is a hyperedge connecting query to all keys
fn transformer_layer(
    tokens: Tensor,          // [seq_len, d_model]
    graph: HyperGraph        // hypergraph where edges = attention heads
) -> Tensor {
    // Multi-head attention
    let Q = tokens @ W_Q
    let K = tokens @ W_K
    let V = tokens @ W_V

    // Attention = hypergraph message passing
    let attended = hypergraph_attention(tokens, graph, W_Q, W_K, W_V)

    // Feedforward
    let ff = relu(attended @ W_ff1) @ W_ff2

    return layer_norm(attended + ff)
}
```

The hypergraph representation makes attention's structure explicit — not "magic," just scatter/gather on a specific topology.

---

## 15. Threading Model: Hyperedges as Units of Work

### Why Hyperedges Are Natural Thread Boundaries

Each hyperedge is:
- An independent unit of computation
- Fully parallel with other hyperedges
- Contains all context needed (member vertices)

```flow
// Parallel execution model
parallel_for edge in graph.hyperedges {
    // Each thread processes one hyperedge
    let members = get_members(edge, graph.B)
    let features = gather_features(members, vertex_features)
    let aggregated = aggregate(features, edge.aggregation_fn)
    write_result(edge, aggregated, edge_messages)
}

// Then scatter results back to vertices
parallel_for vertex in graph.vertices {
    let incident_edges = get_incident_edges(vertex, graph.B)
    let messages = gather_messages(incident_edges, edge_messages)
    let updated = update_fn(vertex_features[vertex], messages)
    write_result(vertex, updated, new_vertex_features)
}
```

This maps directly to:
- **CUDA:** one thread block per hyperedge
- **Vulkan compute:** one workgroup per hyperedge
- **CPU threading:** one task per hyperedge batch

---

## 16. Comparison to Existing Systems

| System | Storage | Query Language | ML Integration | GPU Native | Unified? |
|--------|---------|---------------|----------------|------------|----------|
| **Neo4j** | Property graph (disk) | Cypher | External (Python) | No | No |
| **TypeDB** | Hypergraph (RocksDB) | TypeQL | No | No | No |
| **DuckDB** | Columnar | SQL | External (Python) | No | No |
| **PyTorch** | N/A (tensors only) | Python API | Yes | Yes | No (no graph DB) |
| **DGL + PyTorch** | Separate graph structure | Python API | Yes | Partial | **Almost** (2 layers) |
| **cuGraph** | CSR on GPU | Python API | Via DGL/PyG | Yes | No (analytics only) |
| **OctoFlow HyperGraphDB** | **Sparse matrix (GPU)** | **.flow pattern matching** | **Native (same kernels)** | **Yes** | **YES** |

The key differentiator: **One data structure (incidence matrix B), one set of kernels (SpMM), one language (.flow).**

---

## 17. Implementation Roadmap

### Phase 47-48: Foundation (Matrix Ops + Type System)

**Phase 47: Sparse Matrix Primitives (~400 lines, 15 tests)**
- Sparse matrix type (CSR format)
- SpMM kernel (via cuSPARSE or custom CUDA)
- Transpose, addition, multiplication
- CSR ↔ dense conversion
- Storage: device memory allocation

**Phase 48: Type System (~300 lines, 12 tests)**
- Type declarations in .flow
- Type hierarchy (subtype relationships)
- Type checking at compile time
- Type filtering during query execution

### Phase 49-50: Hypergraph Core (~600 lines, 25 tests)

**Phase 49: Incidence Matrix + Basic Ops (~300 lines, 12 tests)**
- HyperGraph data structure
- Add vertex, add hyperedge
- Build incidence matrix B from edge definitions
- Scatter and gather kernels

**Phase 50: Queries + Message Passing (~300 lines, 13 tests)**
- Pattern matching (BFS traversal)
- Transitive closure
- Message passing primitive
- GCN layer implementation

### Phase 51: Neural Network Integration (~500 lines, 20 tests)

- Attention mechanism
- Multi-layer GNN forward/backward
- Gradient computation via autograd
- Training loop helpers
- Integration with existing array/tensor ops

### Phase 52: Optimization + Multi-GPU (~400 lines, 15 tests)

- Kernel fusion (SpMM + GEMM in one kernel)
- WholeGraph-style multi-GPU sharding
- Memory pooling
- Query optimizer
- **PUBLIC RELEASE READINESS**

---

## 18. Integration with OctoFlow Ecosystem

### The Two-Database Model

**OctoDB (Annex R)** and **HyperGraphDB (Annex L)** serve different purposes:

```
OctoDB:
  - Simple append-only columnar database
  - Persistent storage for analytics
  - Fast scans, time-travel queries
  - No relationships, no ML
  - Use case: "SQLite for OctoFlow"
  - Storage: .octo files on disk

HyperGraphDB:
  - Hypergraph database with neural network integration
  - Persistent graph topology + in-memory training
  - Relationships, embeddings, GNN training
  - Use case: "PyTorch + Neo4j for OctoFlow"
  - Storage: .hypergraph files (topology) + GPU memory (training)
```

### Persistent vs Hot Memory

**Persistent (disk, durable):**
- Graph topology (incidence matrix B structure)
- Vertex/edge types and attributes
- Embedding snapshots (saved after training)
- Model checkpoints

**Hot (GPU memory, ephemeral during training):**
- Current feature tensors
- Model weights being trained
- Gradients
- Optimizer state (Adam moments)
- Gradient tape

**The workflow:**
```flow
// 1. Load graph topology from disk (persistent)
let graph = hypergraph.load("citations.hypergraph")
// Disk → GPU: topology loaded, ready for training

// 2. Train neural network (hot, in-memory)
let mut model = init_gnn()
for epoch in range(200)
    let embeddings = train_epoch(graph, model)  // all GPU memory
end
// Training happens entirely in GPU memory (fast)

// 3. Save results back to disk (persist)
hypergraph.save_embeddings(graph, embeddings, version="v1")
hypergraph.save_checkpoint(model, "model.ckpt")
// GPU → Disk: persist learned embeddings

// 4. Later: query using saved embeddings (load from disk to GPU)
let graph2 = hypergraph.load("citations.hypergraph", embeddings="v1")
let similar_papers = query_similar(graph2, paper_id, top_k=10)
// Disk → GPU → Query → Result
```

### With OctoDB (Annex R)

```flow
// OctoDB: fast analytics on tabular data
let sales = octodb.read_table("sales.octo")
let total = sum(sales["revenue"])

// HyperGraphDB: build knowledge graph from tabular data, train embeddings
let graph = hypergraph.from_table(sales, schema)
let embeddings = train_gnn(graph, labels)
hypergraph.save("sales_graph.hypergraph", embeddings)
```

**Use case:** Start with OctoDB for OLAP queries. When you need relationships and ML, build a HyperGraphDB from the same data.

### With OctoMedia (Annex X)

```flow
// Build image similarity graph
let images = list_dir("photos/")
let features = extract_features(images)  // via neural network
let graph = build_similarity_graph(features, threshold=0.8)

// Query: find all images similar to query_image
let similar = match_pattern(graph, query_image, SIMILAR_TO)
```

**Use case:** Computer vision tasks where relationships between images matter.

### With OctoEngine (Annex Q)

```flow
// Game AI: build behavior graph for NPCs
let behavior_graph = hypergraph {
    states: [IDLE, PATROL, CHASE, ATTACK],
    transitions: [
        (IDLE, SEES_PLAYER, CHASE),
        (CHASE, CLOSE_ENOUGH, ATTACK),
        (ATTACK, PLAYER_DEAD, IDLE)
    ]
}

// Neural network learns when to transition
let policy = train_gnn(behavior_graph, gameplay_data)
```

**Use case:** Game AI that learns from hypergraph-structured behaviors.

---

## Summary

**HyperGraphDB is OctoFlow's PyTorch equivalent:**
- TypeDB-inspired polymorphic type system
- GPU-native sparse matrix storage (CSR format)
- Incidence matrix B as the universal primitive
- 6 fundamental kernels (scatter, gather, SpMM, GEMM, reduce, filter)
- Database queries = hypergraph traversal (BFS via scatter/gather)
- Neural networks = message passing (GNN via scatter/gather)
- Threading model = hyperedges as units of parallel work

**Implementation:** Phases 47-52 (matrix ops → types → hypergraph → queries → neural networks → optimization)

**Public release:** Phase 52 — when someone can build a knowledge graph, query it, and train a GNN on it — all in OctoFlow, all on GPU, all with the same primitives.

---

*"The hypergraph IS the neural network. The query IS the forward pass. The incidence matrix IS the gradient routing. One structure. One language. One GPU."*
