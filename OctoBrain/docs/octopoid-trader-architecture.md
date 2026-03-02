# OctopoidTrader Architecture: Shell, Topology, Plasticity & Brain

> **Source**: `G:\trading_brain_v2\OctopoidTrader` + `G:\trading_brain_v2\ProcessMarkov`
> **Purpose**: Complete architectural documentation for porting to OctoFlow / Loom Runtime
> **Date**: 2026-02-26

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [The Brain: ProcessMarkov + DHGNN](#2-the-brain)
3. [The Shell: Hopfield-Based Topology](#3-the-shell)
4. [Topology: Network Structure & Connections](#4-topology)
5. [Plasticity: Adaptive Mechanisms](#5-plasticity)
6. [Data Structures & Flow](#6-data-structures)
7. [OctoFlow / Loom Runtime Mapping](#7-octoflow-loom-mapping)

---

## 1. System Overview

OctopoidTrader is a live XAUUSD trader where **the DHGNN IS the trader**. Unlike rule-based systems, entry timing emerges from prototype transitions in a Dynamic Hebbian Graph Neural Network. Strategies are "hands" the brain uses to act — selected by hyperedge energy landscape recall.

### Process-Relational Foundation (Octopoid Ontology)

The system embodies three core shifts:

| Traditional Algo | OctopoidTrader |
|-----------------|----------------|
| Fixed indicator rules | Configuration-based prototype matching |
| Context as filter ("if London, use params X") | Context constitutes meaning (session IS PART of the relation type) |
| Snapshot signals ("RSI > 70 = sell") | Process trajectory (character + tempo + consistency of becoming) |

### Three-Process Architecture

```
[Engine]  tick loop → brain.perceive() → should_act() → select_strategy() → execute → teach()
    ↓ writes
[engine_state.json + events.jsonl]  (file-based IPC, atomic writes)
    ↑ reads
[API Server :8888]  +  [Discord Service]
```

### Decision Flow

```
tick → MarketState → brain.perceive()
  → DHGNN prototype transition detected?
    → trajectory quality gate (character ≥ 0.40, tempo 0.15-1.05)
      → strategy EV ranking via hyperedge landscape
        → executor.execute(strategy, direction, sl, tp)
          → strategy.manage() per tick
            → on close: brain.teach(outcome)
              → DHGNN learns (config × strategy × outcome)
```

---

## 2. The Brain: ProcessMarkov + DHGNN

### 2.1 ProcessMarkov Engine

The brain's core is `ProcessMarkovEngine` — an orchestrator integrating:

- **DHGNN**: N-ary Hebbian pattern learner (in-memory compute)
- **MarkovChain**: 1st + 2nd order state transitions
- **PrototypeStore**: Continuous feature clustering (cosine matching + EMA drift)
- **OHLCStore**: Multi-timeframe bar aggregation (M1→M5→M15→H1)
- **BarFeatures**: 13-dimensional process-relational feature extraction

### 2.2 DHGNN (Dynamic Hebbian Graph Neural Network)

The DHGNN is a **hypergraph** where edges connect N nodes simultaneously (not just pairs).

#### Core Data Structure: Hyperedge

```
Hyperedge:
  edge_id:        MD5 hash of (sorted nodes, relation)
  nodes:          FrozenSet of node IDs (N-ary, ≥2 nodes)
  relation:       typed string — "co_occur", "process_seq:london", "produces_win:overlap"
  permanence:     Hebbian strength [0.0, 1.0]
  activations:    fire count
  total_weight:   cumulative learning weight (pip-proportional)
  last_activated: timestamp
```

#### NodeStore: Deterministic Embeddings

```
NodeStore:
  dim = 32                          # embedding dimensionality
  embeddings: Dict[node_id → 32-dim normalized float32 vector]
  names: Dict[node_id → human label]

  hash_name(name) → int:           MD5-based deterministic ID from string
  get(id, name) → embedding:       deterministic init from seed = id % 2^31
  nearest(query, k) → top-k:       cosine similarity search
```

Key property: **same name always produces same ID and same initial embedding**. Reproducible across restarts.

#### PrototypeStore: Feature Clustering

```
PrototypeStore:
  max_prototypes = 500              # hard cap (LRU eviction)
  match_threshold = 0.85            # cosine similarity threshold
  ema_alpha = 0.1                   # drift rate

  match(features) → (proto_id, is_new, similarity):
    1. Normalize input features
    2. Vectorized cosine similarity against all prototypes
    3. If best_sim ≥ 0.85: match (return existing ID)
    4. If best_sim < 0.85: create new prototype
    5. EMA drift matched prototypes: new = (1-α)·old + α·query
    6. LRU eviction when at capacity
```

#### Hebbian Learning (Oja's Rule for Hyperedges)

```python
def learn(activated_nodes, relation, weight):
    embeddings = [node_store.get(n) for n in nodes]
    mean_emb = mean(embeddings)
    mean_sq = dot(mean_emb, mean_emb)

    # Pairwise correlation across all node pairs
    pairwise_corr = avg(dot(e_i, e_j) for all i < j)

    # Oja's update
    Δ = lr × weight × (pairwise_corr - mean_sq × permanence)

    # Unconditional bonus: "fire together → wire together"
    permanence = clip(permanence + Δ + 0.02 × weight, [0, 1])
```

#### Message Passing (Belief Propagation)

```python
def propagate(seed_nodes, iterations=3):
    for each iteration:
        for each node:
            # Gather from hyperedge neighborhoods
            neighbor_signals = []
            for each hyperedge containing this node:
                others = [embeddings of OTHER nodes in edge]
                signal = mean(others) × edge.permanence
                neighbor_signals.append(signal)

            # Weighted mean of neighbor signals
            new_signal = sum(signals × permanences) / sum(permanences)

            # Damped update
            embedding = 0.6 × old + 0.4 × new_signal
            normalize(embedding)
```

#### Hopfield Completion (Pattern Recall)

```python
def complete(known_nodes, k=5):
    # Strategy 1: Hyperedge-based scoring
    for each edge containing any known node:
        for each unknown node in that edge:
            score += permanence × (overlap_fraction)

    # Strategy 2: Embedding-based (complement)
    query = mean(embeddings of known nodes)
    refined = propagate(query, 2 iterations via edges)
    nn_results = cosine_nearest(refined, k)
    blend(hyperedge_scores, nn_scores, weight=0.5)

    return top_k
```

### 2.3 Feature Extraction: 13-Dimensional Process-Relational Vector

This is where octopoid ontology becomes code. Features are **qualitative categories** ("what KIND?"), not raw values ("how MUCH?").

```
CONFIGURATION (dims 0-3) — WHAT kind of market moment
  [0] momentum_char:    choppy(0.0) / drifting(0.33) / steady(0.67) / driven(1.0)
  [1] phase:            exhausted(0.0) / decel(0.25) / brewing(0.5) / accel(0.75) / peak(1.0)
  [2] volatility_regime: low(0.0) / normal(0.33) / high(0.67) / extreme(1.0)
  [3] range_position:    continuous [0.0, 1.0] — where in 50-bar range

CONTEXT (dims 4-7) — WHERE this is happening
  [4] session:          asian(0.0) / london(0.33) / overlap(0.67) / newyork(1.0)
  [5] htf_trend:        bearish(-1.0) / flat(0.0) / bullish(1.0)
  [6] hour_sin:         sin(2π × hour/24)  — cyclical encoding
  [7] hour_cos:         cos(2π × hour/24)  — cyclical encoding

PROCESS (dims 8-12) — HOW we got here (trajectory)
  [8]  prev_momentum:       momentum from 10 bars ago
  [9]  momentum_delta:      current - previous (strengthening/weakening)
  [10] phase_delta:         current - previous (accelerating/decelerating)
  [11] bar_range_trend:     range expansion/contraction ratio [0, 2]
  [12] transition_velocity: rate of EMA separation change, normalized by ATR [-1, +1]
```

### 2.4 MarkovChain: State Transitions

Lightweight 1st + 2nd order Markov over prototype sequences:

```
State key: "P:{proto_id}:{session}"

observe(state):   update _first[current][next] and _second[prev][current][next]
predict():        try 2nd-order (if ≥3 samples), fallback to 1st-order
                  return normalized probability distribution
```

### 2.5 Process Window & Learning

```python
def observe_process(features, session, timestamp):
    proto_id = prototype_store.match(features)
    process_window.append((proto_id, timestamp, session))  # rolling 20

    if len(window) >= 3:
        # Learn sequential pattern
        nodes = last_5_unique_proto_ids
        relation = f"process_seq:{session}"

        # Density weighting: turbulence signal
        density_weight = min(2.0, 60.0 / avg_interval_seconds)
        dhgnn.learn(nodes, relation, weight=density_weight)

def teach_process_outcome(outcome, session, pnl_pips):
    nodes = last_5_proto_ids + outcome_node
    weight = min(3.0, max(0.5, |pnl_pips| / 5.0))
    relation = f"produces_{outcome}:{session}"
    dhgnn.learn(nodes, relation, weight)
```

### 2.6 OctopoidBrain: The Decision Layer

```python
class OctopoidBrain:
    def perceive(market):
        # Feed to DHGNN, detect prototype transitions
        pm.observe(m1_bars, probes, ema, session, bid)
        if proto_id != last_proto:
            transition_detected = True
        # Update market with DHGNN features
        # Run shell plasticity + homeostasis if enabled

    def should_act(market) -> bool:
        # Transition + trajectory quality gate
        # Idle fallback after 60s without transition
        return (transition_detected OR idle_forced) AND trajectory_ok

    def select_strategy(market) -> str:
        # Query DHGNN energy landscape for per-strategy EV
        # Shell: route → topology scores → fuse with direct scores
        # Epsilon-greedy with anti-freeze
        return best_ev_strategy

    def teach(strategy, outcome, market):
        # Two learning channels:
        # 1. Strategy selection: (config × strategy × outcome) hyperedges
        # 2. Market behavior: regime + process learning
```

### 2.7 Strategy Selector: Dual-DHGNN

```
StrategySelector:
  _dhgnn:          main DHGNN (real trades only, weight=1.0)
  _explorer_dhgnn: shadow/explorer DHGNN (0.5-1.0x weight, adaptive)

  select(pm, market) → strategy_name:
    1. Extract 24-dim features from market
    2. Shell forward (if enabled): Hopfield → codebook → circuit → topology scores
    3. Direct DHGNN scores from hyperedge EV query
    4. Fuse via 3-way gates → final scores
    5. Epsilon-greedy selection (decays 0.20 → 0.03)

  teach_outcome(pm, market, strategy, outcome):
    Create hyperedges: (config nodes, strategy node) --strategy_win/loss--> (outcome node)
    Permanence = cumulative pip weight
    Decay every 50 teaches: × 0.995
    Propagate every 3 teaches: message passing
```

---

## 3. The Shell: Hopfield-Based Topology

The **shell** wraps the DHGNN with a topological pattern completion layer.

### 3.1 Architecture: Three Scoring Paths

```
                     ┌─ Path 1: Topology (Hopfield → Codebook → CircuitGrower)
Features (24-dim) ───┤─ Path 2: Direct (existing DHGNN scores)
                     └─ Path 3: Bypass (raw feature linear projection)
                            ↓
                     NumpyFusion (learnable per-strategy gates)
                            ↓
                     Fused Strategy Scores
```

### 3.2 NumpyHopfieldRouter

Routes 24-dim features to nearest learned attractor pattern via Hopfield dynamics.

```
Parameters:
  input_dim  = 24   (market features)
  dim        = 64   (embedding space)
  num_patterns = 16 (attractor count)
  temperature = 0.1 (softmax sharpness)

route(features) → (attractor_id, weights, energy):
  1. Project:     x = features @ W_proj          # 24→64 dim
  2. Normalize:   x_norm = x / ||x||
  3. Similarity:  sim = x_norm @ patterns_norm.T  # vs 16 attractors
  4. Softmax:     weights = softmax(sim / T)
  5. Winner:      attractor_id = argmax(weights)
  6. Energy:      E = -sim[attractor_id]          # negative cosine
```

**Intuition**: Each attractor pattern captures a market regime. The router finds which regime the current market most resembles. Low energy = strong match = confident routing.

### 3.3 NumpyCodebook & NumpyCircuitGrower

**Codebook**: Selects active codes from attractor routing weights.

```
State:
  code_embeddings:  [16 × 64]   — code centroids
  node_templates:   [16 × 4 × 64] — 4 templates per code
  utilization:      [16]          — activation counters

select(attractor_weights) → (nodes, active_codes):
  1. Find codes above threshold (1e-6)
  2. Select top-k=8 by weight
  3. Scale templates: nodes = templates × weight
  4. Concatenate all active nodes → [N × 64]
  5. Track utilization
```

**CircuitGrower**: Scores strategies from pooled codebook nodes.

```
score_strategies(nodes) → [num_strategies]:
  1. Mean-pool: pooled = mean(nodes, axis=0)     # → 64-dim
  2. Project:   scores = pooled @ W_score + bias  # → 7 strategies
```

### 3.4 NumpyFusion: 3-Way Gate

Combines three scoring paths via learnable per-strategy gates.

```
Gate initialization: (0.0, 2.0, 0.0) → softmax → [0.11, 0.78, 0.11]
  Path 1 (topology): 11%  — starts low, earns influence
  Path 2 (direct):   78%  — DHGNN dominates by default
  Path 3 (bypass):   11%  — raw feature baseline

fuse(topology_scores, direct_scores, bypass_scores) → fused_scores:
  for each strategy:
    α, β, γ = softmax(gate_logits[strategy])
    fused[s] = α × topology[s] + β × direct[s] + γ × bypass[s]
```

### 3.5 Shell Persistence

```
shell.save('shell_weights.npz'):
  - router:   W_proj [24×64], patterns [16×64]
  - codebook: embeddings [16×64], templates [16×4×64], utilization [16]
  - grower:   W_score [64×7], b_score [7]
  - fusion:   gate_weights [7×3]
  - bypass:   W_bypass [24×7], b_bypass [7]
  - metadata: dims, counts, strategy_names
```

---

## 4. Topology: Network Structure & Connections

### 4.1 Three-Layer Network

```
LAYER 1: HOPFIELD PATTERN SPACE (Attractor Dynamics)
  ├── 16 attractor patterns (64-dim vectors)
  ├── Dense all-to-all cosine connections
  ├── Input → projection → similarity → softmax → routing
  └── Patterns drift via plasticity

LAYER 2: CODEBOOK NODE SPACE (Template Instantiation)
  ├── Up to 128 code embeddings (64-dim each)
  ├── 4 node templates per code = up to 512 nodes
  ├── Weighted activation from attractor routing
  └── Self-organizing via homeostasis (split/merge)

LAYER 3: STRATEGY SCORE SPACE (Circuit Grower)
  ├── 7 strategies (output nodes)
  ├── Mean-pooled codebook nodes → linear projection
  └── Static weights (learned during training)
```

### 4.2 Full Data Flow

```
Features (24-dim)
    ↓
[W_proj: 24×64] → projected features (64-dim)
    ↓
[patterns: 16×64] ← cosine similarity (Hopfield routing)
    ↓ softmax(sim / 0.1)
[attractor_weights: 16] → top-8 selection
    ↓
[code_embeddings: 16×64] & [node_templates: 16×4×64]
    ↓ instantiate & concatenate
[active_nodes: N×64]
    ↓ mean-pool
[pooled: 64] → [W_score: 64×7] + bias
    ↓
[topology_scores: 7]
    ↓
[NumpyFusion: 3-way gate per strategy]
    ↓
[fused_scores: 7] → argmax → selected strategy
```

### 4.3 DHGNN Hypergraph Topology

Separate from the shell, the DHGNN maintains its own topology:

```
NODES: Deterministic embeddings (32-dim)
  ├── Prototype nodes: "proto:{id}"
  ├── Strategy nodes: "strategy:{name}"
  ├── Session nodes: "session:{london|overlap|...}"
  ├── Regime nodes: "regime:{trending|ranging|volatile}"
  └── Outcome nodes: "outcome:win", "outcome:loss"

HYPEREDGES: N-ary typed connections
  ├── process_seq:{session}     — sequential prototype patterns
  ├── produces_win:{session}    — prototype combos that won
  ├── produces_loss:{session}   — prototype combos that lost
  ├── strategy_win              — config → strategy → win
  ├── strategy_loss             — config → strategy → loss
  └── co_occur                  — general co-activation

INDICES:
  ├── node_index: node_id → Set[edge_ids]
  └── relation_index: relation → Set[edge_ids]
```

---

## 5. Plasticity: Adaptive Mechanisms

Four adaptation mechanisms operate at different timescales:

### 5.1 Hopfield Pattern Plasticity (per perceive, ~5s)

```python
# Attractor patterns drift toward observed market features
effective_lr = 0.03 / (1.0 + importance[pattern_id])
drift = effective_lr × (observed_features - pattern_vector)
pattern_vector += drift
importance[pattern_id] = importance[pattern_id] × 0.99 + 1.0
```

**Properties**:
- Fresh patterns (low importance): adapt quickly (~3% per observation)
- Established patterns (high importance): resist change (~0.3% per observation)
- Elastic decay: importance fades if pattern isn't activated

### 5.2 Codebook Homeostasis (every 500 observations, ~40 min)

```
SPLIT (overloaded code):
  If utilization[i] / total > 0.20 AND num_codes < 128:
    Clone code with Gaussian noise (σ=0.05)
    Halve original's utilization
    → Prevents information bottleneck

MERGE (redundant codes):
  If cosine_similarity(code_i, code_j) > 0.95:
    Average embeddings and templates
    Sum utilization, delete duplicate
    → Frees capacity for specialization
```

### 5.3 DHGNN Edge Decay (every 50 teaches, ~2-3 min)

```
All permanences × 0.995 per cycle
Half-life ≈ 138 cycles ≈ ~7 hours
Effect: Old market conditions lose influence
        System adapts to regime changes within ~30 minutes
```

### 5.4 Prototype EMA Drift (per observation)

```
On prototype match:
  proto_embedding = (1 - 0.1) × old + 0.1 × new_observation
  → Prototypes slowly track market evolution
  → No explicit regime change detection needed
```

### 5.5 Strategy Learning Rates (Adaptive over lifetime)

```
Shadow weight:   max(0.50,  2.0 - real_trades × 1.5/8000)
Explorer weight: max(0.30,  1.2 - real_trades × 0.9/10000)
Explore rate:    max(0.03,  0.20 - 0.002 × trades)

Early (0 trades):     shadows 2.0×, explorer 1.2×, explore 20%
Mid (4500 trades):    shadows 1.15×, explorer 0.78×, explore ~11%
Late (8000+ trades):  shadows 0.5×, explorer 0.3×, explore 3%
```

---

## 6. Data Structures & Flow

### 6.1 MarketState

```
MarketState (dataclass):
  bid, ask, spread_pips                                   # Price
  trend ('long'/'short'/None), momentum, phase            # Trend
  atr_pips, atr_baseline                                  # Volatility
  session ('asian'/'london'/'overlap'/'newyork')           # Session
  proto_id, prev_proto_id                                  # DHGNN
  regime ('trending'/'ranging'/'volatile')                 # Regime
  regime_direction ('bull'/'bear'/'neutral')               # Direction
  regime_confidence (0.0-1.0)                              # Confidence
  trajectory_char (0=exhausting, 1=building)               # Quality
  transition_tempo (0=stalled, 1=frantic)                  # Speed
  trajectory_consistency (0=chaotic, 1=locked)             # Stability
  pre_mfe_pips                                             # MFE
  tick_count                                               # Counter
```

### 6.2 Strategy Priors (from live performance)

```
trend_follow:  +0.10  ($203.83, 81.4% WR, 264 trades)
htf_trend:     +0.08  ($123.36, 83.3% WR, 245 trades)
htf_dbr:       +0.07  ($91.67,  79.7% WR, 182 trades)
htf_revert:    +0.05  ($78.60,  58.6% WR, 181 trades)
dbr:           +0.03  ($47.71,  66.9% WR, 169 trades)
mean_revert:   +0.02  ($9.97,   64.9% WR,  57 trades)
wave_ride:     +0.01  ($12.68,  45.2% WR,  84 trades)
```

### 6.3 Key Constants

```
SYMBOL = "XAUUSD.s"          PIP_SIZE = 0.1
SL: 7-40 pips (ATR-scaled)   TP: 7-100 pips (ATR-scaled)
MIN_RR_RATIO = 0.8           SPREAD_BUFFER = 2.0 pips
MAX_CONCURRENT = 4           RISK_PCT = 0.5%
PM_WARMUP = 30 observations  MAX_IDLE = 60s
```

---

## 7. OctoFlow / Loom Runtime Mapping

### 7.1 Why OctoFlow + Loom?

The OctopoidTrader currently runs on CPU (pure NumPy). The computationally intensive operations — cosine similarity, matrix projections, softmax, message passing — are inherently parallel and ideal for GPU acceleration via the Loom Engine (OctoFlow's GPU VM).

### 7.2 Component → Loom Mapping

| OctopoidTrader Component | Current | Loom Equivalent | GPU Benefit |
|-------------------------|---------|-----------------|-------------|
| **Hopfield routing** | `features @ W_proj` then `x @ patterns.T` then softmax | `loom.map(multiply)` + `loom.reduce(sum)` + `loom.fused(normalize)` | 16 attractor comparisons in parallel |
| **Codebook selection** | Top-k from weights, template scaling | `loom.map(multiply)` per active code | N×64 template instantiation parallel |
| **CircuitGrower** | Mean-pool + linear projection | `loom.reduce(sum)` + `loom.map(multiply+add)` | 64→7 projection in one kernel |
| **Fusion gates** | Per-strategy softmax + weighted sum | `loom.fused(scale_shift)` | 7 strategies × 3 paths parallel |
| **Cosine similarity** | `dot(a, b) / (||a|| × ||b||)` | `loom.map(multiply)` + `loom.reduce(sum)` + `loom.fused(normalize)` | Core of ALL matching ops |
| **Hebbian learning** | Pairwise correlation + Oja's update | `loom.map(multiply)` + `loom.reduce(sum)` for correlations | N² pair correlations parallel |
| **Message passing** | Neighbor gather + weighted mean | `loom.scan(prefix_sum)` for accumulation | E edges × N iterations |
| **Prototype matching** | All-prototype cosine comparison | `loom.map(multiply)` + `loom.reduce(sum)` over 500 protos | 500× parallel comparison |
| **Feature extraction** | EMA, ATR, phase detection | `loom.temporal(ema)` + `loom.map(abs/sqrt)` | Bar-level parallel compute |

### 7.3 Adaptive Dimensions

**Key insight**: The system's dimensions are not fixed — they adapt to what the data provides.

```
CURRENT FIXED DIMS:
  Features:     24-dim (strategy selector) or 13-dim (ProcessMarkov)
  Embeddings:   32-dim (DHGNN nodes) or 64-dim (shell)
  Prototypes:   up to 500 (LRU eviction)
  Codes:        16-128 (homeostasis split/merge)
  Patterns:     16 (Hopfield attractors)
  Strategies:   7 active

ADAPTIVE IN LOOM:
  Feature dim:     Could grow as new instruments/data sources added
  Embedding dim:   Could scale with complexity (more regimes → higher dim)
  Prototype count: GPU handles 10K+ prototypes at same speed as 500
  Code count:      Homeostasis split/merge bounded only by GPU memory
  Pattern count:   Could scale to 64/128 attractors with GPU cosine
```

**Loom's advantage**: With GPU parallel compute, the matching operations that currently scale O(P×D) on CPU become O(1) wall-clock on GPU (all prototypes compared simultaneously). This enables:

1. **More prototypes** = finer-grained market regime detection
2. **Higher embedding dims** = richer representation capacity
3. **More attractor patterns** = more nuanced Hopfield routing
4. **Larger codebooks** = more expressive topology

### 7.4 Proposed OctoFlow Architecture

```flow
# OctoBrain in OctoFlow (conceptual)

# --- Feature Extraction ---
use "loom/temporal" as temporal
use "loom/map" as map

let bars = tap("m1_bars.octo")
let ema_fast = temporal.ema(bars, 0.2)       # GPU EMA
let ema_slow = temporal.ema(bars, 0.05)
let atr = map.abs(bars.high - bars.low)      # GPU ATR

# --- Prototype Matching ---
use "loom/fused" as fused

let features = build_features(ema_fast, ema_slow, atr, session)
let projected = map.multiply(features, W_proj)       # 24→64
let normalized = fused.normalize(projected)
let similarities = map.multiply(normalized, patterns) # vs all attractors
let weights = fused.softmax(similarities, temperature=0.1)

# --- Codebook + Circuit ---
let active_nodes = codebook_select(weights)
let pooled = reduce.mean(active_nodes)
let topology_scores = map.multiply(pooled, W_score)

# --- Fusion ---
let fused = fusion_gate(topology_scores, direct_scores, bypass_scores)
let strategy = argmax(fused)
```

### 7.5 Loom Kernel Opportunities

**Kernel 1: Hopfield Route** (fused)
```
Input:  features [24]
Output: attractor_id, weights [16], energy
Ops:    matmul(24×64) → normalize → matmul(64×16) → softmax → argmax
        → Single GPU kernel, no CPU roundtrip
```

**Kernel 2: Codebook Score** (fused)
```
Input:  attractor_weights [16]
Output: strategy_scores [7]
Ops:    threshold → top-k → template scale → concatenate → mean-pool → matmul(64×7)
        → Single kernel from routing to scoring
```

**Kernel 3: Hebbian Learn** (map + reduce)
```
Input:  node_embeddings [N × 32], permanence
Output: updated permanence, updated embeddings
Ops:    pairwise_dot → mean → oja_delta → clamp
        → Parallel over all node pairs
```

**Kernel 4: Prototype Match** (map + reduce)
```
Input:  query [13], prototypes [P × 13]
Output: best_id, similarity
Ops:    broadcast_multiply → reduce_sum → argmax
        → All P prototypes compared simultaneously
```

### 7.6 What Stays on CPU

Not everything should move to GPU:

- **MT5 communication**: Tick reception, order execution (I/O bound)
- **Markov chain**: Small state (< 1000 entries), sequential transitions
- **Decision logic**: `should_act()`, `select_strategy()` are branching logic
- **File I/O**: engine_state.json, events.jsonl writing
- **Strategy management**: Per-tick position management (sequential)

### 7.7 Migration Path

```
Phase 1: Feature Extraction → Loom
  Move EMA, ATR, phase detection to GPU temporal/map patterns
  Keep everything else on CPU
  Validation: feature vectors match NumPy output (1e-6 tolerance)

Phase 2: Prototype Matching → Loom
  Move cosine similarity + argmax to GPU
  Enables scaling to 10K+ prototypes
  Validation: same proto_id assignments

Phase 3: Shell Forward Pass → Loom
  Move Hopfield routing + codebook + circuit to single fused kernel
  Three scoring paths computed in one GPU dispatch
  Validation: same strategy selections

Phase 4: Hebbian Learning → Loom
  Move Oja's rule + correlation computation to GPU
  Enables larger hyperedges (10+ nodes)
  Validation: permanence convergence matches

Phase 5: Full Brain on Loom
  All compute on GPU, only I/O on CPU
  Adaptive dimensions: scale embeddings/prototypes/codes dynamically
  Profiling: measure latency improvement per kernel
```

---

## Appendix A: Octopoid Ontology Mapping

The system's architecture directly implements process-relational ontology:

| Octopoid Concept | Implementation |
|-----------------|----------------|
| **Configuration** (mutually constituted whole) | 13-dim feature vector: momentum × phase × vol × range (not independent variables) |
| **Context constitutes** (not filters) | Session is PART OF relation type: `"process_seq:london"` ≠ `"process_seq:asian"` |
| **Process primary** (trajectory, not snapshots) | Dims 8-12: deltas, acceleration, transition velocity (how becoming) |
| **What KIND not how MUCH** | Ordinal categories: choppy/drifting/steady/driven (not RSI=73.2) |
| **No skeleton** | No fixed rules. Strategy emerges from Hebbian energy landscape |
| **Limiting case** | Shell starts at 78% DHGNN weight, topology earns influence through performance |
| **Driven vs drifting** | Trajectory character (0=exhausting, 1=building) gates entry |

## Appendix B: File Locations

```
G:\trading_brain_v2\
├── OctopoidTrader/
│   ├── brain.py              # OctopoidBrain
│   ├── strategy_selector.py  # StrategySelector + dual-DHGNN
│   ├── executor.py           # StrategyExecutor
│   ├── market.py             # MarketState dataclass
│   ├── config.py             # All constants
│   ├── shadow.py             # ShadowEvaluator
│   ├── shell/
│   │   ├── shell_np.py       # NumpyShell orchestrator
│   │   ├── hopfield_np.py    # NumpyHopfieldRouter
│   │   ├── codebook_np.py    # NumpyCodebook + NumpyCircuitGrower
│   │   ├── fusion_np.py      # NumpyFusion (3-way gate)
│   │   ├── plasticity_np.py  # HebbianPlasticity
│   │   └── homeostasis_np.py # HomeostaticRegulator
│   └── strategies/           # 14 strategy implementations
│
├── ProcessMarkov/
│   ├── dhgnn.py              # DHGNN + NodeStore + PrototypeStore
│   ├── hypergraph.py         # HypergraphDB (SQLite persistence)
│   ├── engine.py             # ProcessMarkovEngine + MarkovChain
│   ├── features.py           # 13-dim feature extraction
│   ├── signals.py            # SetupScanner + SignalGenerator
│   └── ohlc_store.py         # Multi-timeframe OHLC store
│
└── RebateFarmer/             # Shared: MT5Ops, TickEMA, PMBridge
```
