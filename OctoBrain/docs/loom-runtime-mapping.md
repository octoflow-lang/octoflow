# OctoBrain → Loom Runtime: GPU Mapping & Adaptive Dimensions

> **Purpose**: Detailed technical mapping of OctopoidTrader/ProcessMarkov algorithms to OctoFlow's Loom Engine GPU patterns
> **Date**: 2026-02-26

---

## 1. Loom Pattern Inventory

OctoFlow's Loom Engine provides these GPU patterns:

| Pattern | Operation | SPIR-V |
|---------|-----------|--------|
| `map` | Element-wise transform (19 MapOps) | Per-invocation arithmetic |
| `reduce` | Tree reduction (sum, min, max) | Shared memory tree fold |
| `scan` | Prefix sum (Hillis-Steele) | Branchless propagate |
| `temporal` | EMA / decay (loop + OpPhi) | SPIR-V loop with carry |
| `fused` | Normalize / scale_shift (multi-op) | Single kernel compound |

---

## 2. Algorithm → Kernel Mapping

### 2.1 Cosine Similarity (Foundation Op)

Used everywhere: Hopfield routing, prototype matching, code merging, completion.

```
cosine(a, b) = dot(a, b) / (||a|| × ||b||)

Loom decomposition:
  1. map.multiply(a, b)           → element-wise products
  2. reduce.sum(products)         → dot product
  3. map.multiply(a, a)           → a² elements
  4. reduce.sum(a_squared)        → ||a||²
  5. map.sqrt(norm_sq)            → ||a||
  6. map.divide(dot, norm_a × norm_b)  → cosine similarity

Batched (M queries × N targets):
  For prototype matching (1 query vs 500 prototypes):
    - Broadcast query [D] to [500 × D]
    - map.multiply(broadcast_query, proto_matrix)  → [500 × D]
    - reduce.sum(axis=1)                           → [500] dot products
    - → All 500 similarities in one GPU dispatch
```

### 2.2 Hopfield Routing

```
Input:  features [24]
Params: W_proj [24 × 64], patterns [16 × 64], temperature = 0.1

Step 1: Project (matmul as batched map+reduce)
  for each output_dim j in [0..64):
    map.multiply(features, W_proj[:, j])    → [24] products
    reduce.sum(products)                     → projected[j]
  → Or: fused.matmul_24x64(features, W_proj) as single kernel

Step 2: Normalize
  fused.normalize(projected)                 → x_norm [64]

Step 3: Pattern similarity (batched cosine)
  map.multiply(x_norm_broadcast, patterns)   → [16 × 64]
  reduce.sum(axis=1)                         → sim [16]

Step 4: Temperature-scaled softmax
  map.divide(sim, 0.1)                       → scaled [16]
  map.exp(scaled - max(scaled))              → exp_vals [16]
  reduce.sum(exp_vals)                       → denom
  map.divide(exp_vals, denom)                → weights [16]

Step 5: Winner
  reduce.max(weights)                        → attractor_id

Total: ~6 GPU dispatches (or 1 fused kernel)
```

### 2.3 Codebook Selection + CircuitGrower

```
Input:  attractor_weights [16]
Params: templates [16 × 4 × 64], W_score [64 × 7], b_score [7]

Step 1: Threshold + top-k (CPU — branching logic, small data)
  active_codes = [i for i, w in enumerate(weights) if w > 1e-6][:8]

Step 2: Template scaling (GPU — parallel over all active nodes)
  for each active code i:
    map.multiply(templates[i], weights[i])    → scaled_nodes [4 × 64]
  Concatenate → all_nodes [N × 64]           (N = sum of active × 4)

Step 3: Mean pool (GPU)
  reduce.sum(all_nodes, axis=0)              → sum [64]
  map.divide(sum, N)                         → pooled [64]

Step 4: Linear projection (GPU — matmul)
  for each strategy j in [0..7):
    map.multiply(pooled, W_score[:, j])      → [64] products
    reduce.sum(products)                     → raw_score[j]
  map.add(raw_scores, b_score)               → topology_scores [7]
```

### 2.4 3-Way Fusion

```
Input:  topology [7], direct [7], bypass [7]
Params: gate_logits [7 × 3]

Per strategy (parallel over 7):
  1. Softmax over 3 gates:
     map.exp(gate_logits[s] - max)           → [3]
     reduce.sum(exp_vals)                    → denom
     map.divide(exp_vals, denom)             → [α, β, γ]

  2. Weighted sum:
     fused[s] = α × topology[s] + β × direct[s] + γ × bypass[s]
     → map.multiply + reduce.sum

Total: 7 strategies × 3 gates = 21 multiply-adds (trivially parallel)
```

### 2.5 Hebbian Learning (Oja's Rule)

```
Input:  node_embeddings [N × 32], current permanence
Output: updated permanence

Step 1: Pairwise correlation
  for each pair (i, j) where i < j:
    map.multiply(emb[i], emb[j])             → [32] products
    reduce.sum(products)                     → correlation[i,j]
  → N(N-1)/2 pairs, parallelizable

Step 2: Mean embedding
  reduce.sum(embeddings, axis=0)             → sum [32]
  map.divide(sum, N)                         → mean_emb [32]

Step 3: Mean squared
  map.multiply(mean_emb, mean_emb)           → [32]
  reduce.sum(sq_products)                    → mean_sq

Step 4: Oja's delta
  avg_corr = mean(all correlations)
  Δ = lr × weight × (avg_corr - mean_sq × permanence) + 0.02 × weight
  permanence = clamp(permanence + Δ, 0.0, 1.0)
```

### 2.6 Prototype Matching (Vectorized)

```
Input:  query [13], prototypes [P × 13]
Output: best_proto_id, similarity

  map.multiply(query_broadcast, prototypes)  → [P × 13]
  reduce.sum(axis=1)                         → dot_products [P]
  reduce.max(dot_products)                   → best_sim, best_id

With P=500: All comparisons in ONE dispatch
With P=10000 (GPU-enabled): Same wall-clock time!
```

### 2.7 Temporal Features (EMA + ATR)

```
Input:  close_prices [N], high [N], low [N]

EMA (Loom temporal pattern):
  temporal.ema(close_prices, alpha=0.2)      → ema_fast [N]
  temporal.ema(close_prices, alpha=0.05)     → ema_slow [N]

ATR:
  map.subtract(high, low)                    → ranges [N]
  temporal.ema(ranges, alpha=0.07)           → atr [N]

Phase detection:
  map.subtract(close[1:], close[:-1])        → velocity [N-1]
  map.subtract(velocity[1:], velocity[:-1])  → acceleration [N-2]
  temporal.ema(acceleration, alpha=0.3)      → smoothed_accel
```

### 2.8 Message Passing (Propagate)

```
Input:  seed_nodes [S], all_edges, iterations=3

Per iteration, per node:
  1. Gather: Find all edges containing this node
  2. For each edge: mean(OTHER node embeddings) × permanence
  3. Weighted sum of all edge signals / sum(permanences)
  4. Damped update: 0.6 × old + 0.4 × new

Loom mapping:
  Step 2: map.multiply + reduce.sum (per edge)
  Step 3: reduce.sum (weighted accumulation)
  Step 4: fused.scale_shift (damping blend)

Challenge: Irregular graph structure → requires sparse dispatch
Solution: Pre-compute adjacency list, pad to fixed max-neighbors
  → map over fixed-width neighbor array with masking
```

---

## 3. Adaptive Dimensions

### 3.1 Current Fixed Dimensions

```
Feature dims:    13 (ProcessMarkov) or 24 (StrategySelector)
Embedding dim:   32 (DHGNN) or 64 (Shell)
Prototypes:      max 500 (LRU eviction)
Codes:           16-128 (homeostasis)
Patterns:        16 (Hopfield attractors)
Strategies:      7 active
Process window:  20 entries
```

### 3.2 GPU-Enabled Scaling

| Dimension | CPU Limit | GPU Target | Reason |
|-----------|-----------|------------|--------|
| Feature dim | 13-24 | 32-64 | More instrument features, order flow |
| Embedding dim | 32-64 | 128-256 | Richer regime representation |
| Prototypes | 500 | 5,000-50,000 | Finer market discrimination |
| Codes | 128 | 512-2,048 | More expressive topology |
| Patterns | 16 | 64-256 | More Hopfield attractors |
| Strategies | 7 | 20-50 | More strategy variants |
| Process window | 20 | 100-500 | Deeper history |

### 3.3 Dynamic Dimension Selection

**Principle**: Let the data determine dimensions, not hardcoded constants.

```flow
# Adaptive embedding dimension
let complexity = measure_regime_diversity(prototypes)
let embed_dim = if complexity > 0.8 then 256
                elif complexity > 0.5 then 128
                elif complexity > 0.3 then 64
                else 32
                end

# Adaptive prototype count
let data_density = total_observations / unique_prototypes
let max_protos = if data_density > 100 then 10000
                 elif data_density > 50 then 5000
                 elif data_density > 20 then 2000
                 else 500
                 end
```

### 3.4 Loom Dispatch Strategy

```
For small arrays (< 256 elements):
  → CPU fallback (overhead of GPU dispatch exceeds benefit)

For medium arrays (256 - 10K elements):
  → Single Loom dispatch (one workgroup)

For large arrays (10K+ elements):
  → Multi-workgroup dispatch with shared memory reduction

OctoBrain typical sizes:
  Prototype matching:  500-50K elements  → GPU
  Hopfield routing:    16-256 elements   → GPU if batched, CPU if single
  Codebook selection:  16-2048 elements  → GPU
  Feature extraction:  100-1000 bars     → GPU (temporal pattern)
  Hebbian learning:    3-10 node pairs   → CPU (too small for GPU)
  Message passing:     100-10K edges     → GPU (if batched per iteration)
```

---

## 4. OctoFlow Implementation Sketch

### 4.1 Core Types

```flow
# OctoBrain types in OctoFlow

struct Embedding(dim)           # N-dim normalized vector
struct Prototype(id, embedding, last_match, match_count)
struct Hyperedge(nodes, relation, permanence, activations, weight)
struct AttractorPattern(embedding, importance, activation_count)
struct CodeEntry(embedding, templates, utilization)
struct MarketFeatures(
    momentum_char, phase, vol_regime, range_pos,     # config [0-3]
    session, htf_trend, hour_sin, hour_cos,          # context [4-7]
    prev_momentum, momentum_delta, phase_delta,      # process [8-10]
    bar_range_trend, transition_velocity              # process [11-12]
)
```

### 4.2 Feature Extraction Module

```flow
# features.flow — 13-dim process-relational feature extraction

use "loom/temporal" as temporal
use "loom/map" as map

fn extract_features(bars, session_id)
    # Configuration (WHAT KIND)
    let ema_fast = temporal.ema(bars.close, 0.2)
    let ema_slow = temporal.ema(bars.close, 0.05)
    let separation = map.subtract(ema_fast, ema_slow)
    let momentum_char = classify_momentum(separation)
    let phase = classify_phase(bars.close)
    let vol_regime = classify_volatility(bars)
    let range_pos = compute_range_position(bars.close)

    # Context (WHERE)
    let htf_trend = compute_htf_trend(bars.close)
    let hour_sin = sin(2.0 * 3.14159 * hour / 24.0)
    let hour_cos = cos(2.0 * 3.14159 * hour / 24.0)

    # Process (HOW — trajectory)
    let prev_momentum = classify_momentum(separation_10_ago)
    let momentum_delta = momentum_char - prev_momentum
    let phase_delta = phase - prev_phase
    let bar_range_trend = avg_range_recent / avg_range_prev
    let transition_velocity = delta_separation / atr

    return MarketFeatures(
        momentum_char, phase, vol_regime, range_pos,
        session_id, htf_trend, hour_sin, hour_cos,
        prev_momentum, momentum_delta, phase_delta,
        bar_range_trend, transition_velocity
    )
end
```

### 4.3 Hopfield Router Module

```flow
# hopfield.flow — attractor pattern routing on GPU

use "loom/map" as map
use "loom/reduce" as reduce

fn hopfield_route(features, W_proj, patterns, temperature)
    # Project: 24 → 64
    let projected = matmul(features, W_proj)

    # Normalize
    let norm = sqrt(reduce.sum(map.multiply(projected, projected)))
    let x_norm = map.divide(projected, norm)

    # Cosine similarity vs all patterns
    let similarities = matmul_batch(x_norm, patterns)  # [num_patterns]

    # Temperature-scaled softmax
    let max_sim = reduce.max(similarities)
    let scaled = map.divide(map.subtract(similarities, max_sim), temperature)
    let exp_vals = map.exp(scaled)
    let denom = reduce.sum(exp_vals)
    let weights = map.divide(exp_vals, denom)

    let attractor_id = argmax(weights)
    let energy = -1.0 * similarities[attractor_id]

    return attractor_id, weights, energy
end
```

### 4.4 Plasticity Module

```flow
# plasticity.flow — Hebbian pattern drift

fn update_pattern(pattern, observed, importance, lr, elastic_decay)
    # Effective learning rate decreases with importance
    let effective_lr = lr / (1.0 + importance)

    # Drift pattern toward observation
    let drift = map.multiply(
        map.subtract(observed, pattern),
        effective_lr
    )
    let new_pattern = map.add(pattern, drift)

    # Update importance (elastic decay)
    let new_importance = importance * elastic_decay + 1.0

    return new_pattern, new_importance
end
```

---

## 5. Performance Estimates

### 5.1 Current CPU Performance (NumPy)

| Operation | CPU Time | Frequency |
|-----------|----------|-----------|
| Feature extraction | ~0.5ms | Every 5s |
| Prototype matching (500) | ~0.2ms | Every 5s |
| Hopfield routing | ~0.1ms | Per transition |
| Codebook + circuit | ~0.3ms | Per transition |
| Fusion | ~0.05ms | Per transition |
| Hebbian learning | ~0.1ms | Per teach |
| Message passing (3 iter) | ~2ms | Every 3 teaches |

Total per decision: ~3ms (CPU, well within 200ms tick budget)

### 5.2 Projected GPU Performance (Loom)

At current dimensions, GPU overhead exceeds CPU benefit. **GPU wins when scaling up**:

| Operation | CPU @ 500 protos | GPU @ 500 protos | GPU @ 50K protos |
|-----------|-----------------|-------------------|------------------|
| Prototype match | 0.2ms | 0.5ms (overhead) | 0.5ms (same!) |
| Hopfield route | 0.1ms | 0.3ms | 0.3ms |
| Full forward | 3ms | 2ms | 2.5ms |

**GPU breakeven**: ~2,000 prototypes. Above that, GPU is faster and scales O(1).

### 5.3 Batch Processing Advantage

For backtesting / shadow evaluation, GPU excels:

```
Shadow evaluation: 7 strategies × 1 market state = 7 forward passes
  CPU: 7 × 3ms = 21ms
  GPU: 1 dispatch (all 7 parallel) = ~1ms

Backtesting: 100K bars × feature extraction
  CPU: 100K × 0.5ms = 50 seconds
  GPU: 100K bars as single temporal dispatch = ~200ms
```

---

## 6. Open Questions

1. **OctoFlow matmul**: Current Loom patterns don't include explicit matrix multiply. Need fused map+reduce pattern or new `loom.matmul` primitive?

2. **Sparse graph dispatch**: Message passing over irregular hypergraph topology. Need padded adjacency + masking, or variable-length workgroup support?

3. **Dynamic memory**: Codebook homeostasis creates/destroys codes. GPU buffer resizing strategy? Pre-allocate max and use active mask?

4. **Mixed precision**: Hebbian permanence needs float32 precision. Feature matching could use float16 for throughput. Loom mixed-precision support?

5. **Persistence**: DHGNN state lives in SQLite (HypergraphDB). OctoFlow equivalent? .octo binary format for node embeddings + edge lists?

6. **Process window**: Rolling buffer of (proto_id, timestamp, session). Sequential by nature. Keep on CPU or implement as ring buffer in GPU shared memory?
