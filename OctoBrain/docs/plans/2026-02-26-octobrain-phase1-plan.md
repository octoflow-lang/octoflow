# OctoBrain Phase 1: CPU Foundation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the complete OctoBrain adaptive brain on CPU using OctoFlow scalar/array ops, proving correctness before GPU acceleration.

**Architecture:** Skeleton-free brain with adaptive dimensions. Functions return updated state maps (OctoFlow maps are value-typed). 2D data stored as flattened 1D arrays with stride access (`arr[row * cols + col]`). No hardcoded dimensions — all sizes discovered at runtime.

**Tech Stack:** OctoFlow (.flow files), flowgpu-cli runtime, .octo binary format for persistence.

**Run command** (used throughout — define once):
```bash
FLOWRUN='powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\\FlowGPU\\run_test.ps1" run --bin flowgpu-cli --'
# Usage: $FLOWRUN run path/to/file.flow
```

---

### Task 1: Vector Math Utilities

Foundation: cosine similarity, vector normalization, dot product — used by every other module.

**Files:**
- Create: `lib/vecmath.flow`
- Create: `tests/test_vecmath.flow`

**Step 1: Write the test**

Create `tests/test_vecmath.flow`:

```flow
use "../lib/vecmath"

# --- Test dot product ---
let a = [1.0, 0.0, 0.0]
let b = [0.0, 1.0, 0.0]
let d1 = vecmath.dot_product(a, b, 3.0)
if d1 == 0.0
    print("PASS dot_product orthogonal = 0")
else
    print("FAIL dot_product orthogonal: got {d1}")
end

let c = [1.0, 2.0, 3.0]
let d = [4.0, 5.0, 6.0]
let d2 = vecmath.dot_product(c, d, 3.0)
# 1*4 + 2*5 + 3*6 = 32
if d2 == 32.0
    print("PASS dot_product = 32")
else
    print("FAIL dot_product: got {d2}")
end

# --- Test vec_norm ---
let e = [3.0, 4.0]
let n1 = vecmath.vec_norm(e, 2.0)
if n1 == 5.0
    print("PASS vec_norm = 5")
else
    print("FAIL vec_norm: got {n1}")
end

# --- Test cosine similarity ---
# Identical vectors → 1.0
let f = [1.0, 2.0, 3.0]
let sim1 = vecmath.cosine_sim(f, f, 3.0)
if sim1 > 0.999
    print("PASS cosine_sim identical ~ 1.0")
else
    print("FAIL cosine_sim identical: got {sim1}")
end

# Orthogonal vectors → 0.0
let sim2 = vecmath.cosine_sim(a, b, 3.0)
if abs(sim2) < 0.001
    print("PASS cosine_sim orthogonal ~ 0.0")
else
    print("FAIL cosine_sim orthogonal: got {sim2}")
end

# --- Test normalize ---
let g = [3.0, 4.0]
let g_norm = vecmath.normalize(g, 2.0)
let check_norm = vecmath.vec_norm(g_norm, 2.0)
if abs(check_norm - 1.0) < 0.001
    print("PASS normalize produces unit vector")
else
    print("FAIL normalize: norm = {check_norm}")
end

# --- Test vec_subtract ---
let h = [5.0, 3.0]
let j = [2.0, 1.0]
let diff = vecmath.vec_subtract(h, j, 2.0)
if diff[0] == 3.0 && diff[1] == 2.0
    print("PASS vec_subtract")
else
    print("FAIL vec_subtract: got [{diff[0]}, {diff[1]}]")
end

# --- Test vec_scale ---
let k = [2.0, 4.0]
let scaled = vecmath.vec_scale(k, 0.5, 2.0)
if scaled[0] == 1.0 && scaled[1] == 2.0
    print("PASS vec_scale")
else
    print("FAIL vec_scale")
end

# --- Test vec_add ---
let added = vecmath.vec_add(h, j, 2.0)
if added[0] == 7.0 && added[1] == 4.0
    print("PASS vec_add")
else
    print("FAIL vec_add")
end

print("--- vecmath tests complete ---")
```

**Step 2: Run test to verify it fails**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_vecmath.flow"
```
Expected: FAIL — module `vecmath` not found or functions undefined.

**Step 3: Implement vecmath.flow**

Create `lib/vecmath.flow`:

```flow
# vecmath.flow — Vector math utilities for OctoBrain
# All vectors are flat 1D arrays. Dimension passed explicitly.

fn dot_product(a, b, dim)
    let mut result = 0.0
    let mut i = 0.0
    while i < dim
        result = result + a[i] * b[i]
        i = i + 1.0
    end
    return result
end

fn vec_norm(v, dim)
    let mut sum_sq = 0.0
    let mut i = 0.0
    while i < dim
        sum_sq = sum_sq + v[i] * v[i]
        i = i + 1.0
    end
    return sqrt(sum_sq)
end

fn cosine_sim(a, b, dim)
    let dot = dot_product(a, b, dim)
    let na = vec_norm(a, dim)
    let nb = vec_norm(b, dim)
    if na < 0.000001 || nb < 0.000001
        return 0.0
    end
    return dot / (na * nb)
end

fn normalize(v, dim)
    let n = vec_norm(v, dim)
    if n < 0.000001
        return v
    end
    let mut result = []
    let mut i = 0.0
    while i < dim
        push(result, v[i] / n)
        i = i + 1.0
    end
    return result
end

fn vec_subtract(a, b, dim)
    let mut result = []
    let mut i = 0.0
    while i < dim
        push(result, a[i] - b[i])
        i = i + 1.0
    end
    return result
end

fn vec_add(a, b, dim)
    let mut result = []
    let mut i = 0.0
    while i < dim
        push(result, a[i] + b[i])
        i = i + 1.0
    end
    return result
end

fn vec_scale(v, scalar, dim)
    let mut result = []
    let mut i = 0.0
    while i < dim
        push(result, v[i] * scalar)
        i = i + 1.0
    end
    return result
end

# Extract a sub-vector from a flat 2D array
# flat_array[row * cols + 0 .. row * cols + cols-1]
fn vec_extract(flat_array, row, cols)
    let mut result = []
    let offset = row * cols
    let mut i = 0.0
    while i < cols
        push(result, flat_array[offset + i])
        i = i + 1.0
    end
    return result
end

# Write a sub-vector into a flat 2D array (returns new array)
fn vec_insert(flat_array, row, cols, values)
    let mut result = []
    let total = len(flat_array)
    let offset = row * cols
    let mut i = 0.0
    while i < total
        if i >= offset && i < offset + cols
            push(result, values[i - offset])
        else
            push(result, flat_array[i])
        end
        i = i + 1.0
    end
    return result
end

# Create a zero vector of given dimension
fn vec_zeros(dim)
    let mut result = []
    let mut i = 0.0
    while i < dim
        push(result, 0.0)
        i = i + 1.0
    end
    return result
end
```

**Step 4: Run test to verify it passes**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_vecmath.flow"
```
Expected: All PASS lines printed.

**Step 5: Commit**

```bash
git add lib/vecmath.flow tests/test_vecmath.flow
git commit -m "feat(octobrain): vector math utilities — dot, cosine, normalize, extract/insert"
```

---

### Task 2: Prototype Store

Match observations to learned prototypes. Prototypes grow unbounded when new data doesn't match.

**Files:**
- Create: `lib/proto.flow`
- Create: `tests/test_proto.flow`

**Step 1: Write the test**

Create `tests/test_proto.flow`:

```flow
use "../lib/vecmath"
use "../lib/proto"

# --- Test: first observation creates first prototype ---
let mut state = proto.proto_new()
state = proto.proto_observe(state, [1.0, 0.0, 0.0], 3.0)
let pc = map_get(state, "proto_count")
if pc == 1.0
    print("PASS first observation creates prototype")
else
    print("FAIL proto_count: got {pc}")
end

# --- Test: similar observation matches existing ---
state = proto.proto_observe(state, [0.99, 0.01, 0.0], 3.0)
let pc2 = map_get(state, "proto_count")
if pc2 == 1.0
    print("PASS similar observation matches (still 1 proto)")
else
    print("FAIL proto_count after similar: got {pc2}")
end

# --- Test: different observation creates new prototype ---
state = proto.proto_observe(state, [0.0, 1.0, 0.0], 3.0)
let pc3 = map_get(state, "proto_count")
if pc3 == 2.0
    print("PASS different observation creates new proto (2 total)")
else
    print("FAIL proto_count after different: got {pc3}")
end

# --- Test: match returns correct proto ID ---
let mid = map_get(state, "last_match_id")
if mid == 1.0
    print("PASS last_match_id = 1 (second proto)")
else
    print("FAIL last_match_id: got {mid}")
end

# --- Test: re-observing first cluster matches proto 0 ---
state = proto.proto_observe(state, [0.98, 0.02, 0.01], 3.0)
let mid2 = map_get(state, "last_match_id")
if mid2 == 0.0
    print("PASS re-match first cluster = proto 0")
else
    print("FAIL re-match: got {mid2}")
end

# --- Test: transition detection ---
# Previous was proto 0, before that proto 1 → transition happened
let td = map_get(state, "transition_detected")
if td == 1.0
    print("PASS transition detected (1 → 0)")
else
    print("FAIL transition_detected: got {td}")
end

# --- Test: EMA drift ---
# After matching, prototype should have drifted toward the observation
let protos = map_get(state, "embeddings")
let embed_dim = map_get(state, "embed_dim")
let p0 = vecmath.vec_extract(protos, 0.0, embed_dim)
# Original was [1,0,0], matched [0.98,0.02,0.01], should have drifted slightly
if p0[0] < 1.0 && p0[0] > 0.95
    print("PASS prototype drifted toward observation")
else
    print("FAIL drift: p0[0] = {p0[0]}")
end

print("--- proto tests complete ---")
```

**Step 2: Run test to verify it fails**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_proto.flow"
```
Expected: FAIL — proto module not found.

**Step 3: Implement proto.flow**

Create `lib/proto.flow`:

```flow
# proto.flow — Prototype store with adaptive growth
# Prototypes stored as flat array: [proto_count × embed_dim]

use "../lib/vecmath"

let MATCH_THRESHOLD = 0.85
let EMA_ALPHA = 0.1

fn proto_new()
    let mut state = map_new()
    map_set(state, "proto_count", 0.0)
    map_set(state, "embed_dim", 0.0)
    map_set(state, "embeddings", [])
    map_set(state, "match_counts", [])
    map_set(state, "last_match_id", -1.0)
    map_set(state, "last_match_sim", 0.0)
    map_set(state, "prev_match_id", -1.0)
    map_set(state, "transition_detected", 0.0)
    map_set(state, "transition_count", 0.0)
    return state
end

fn proto_observe(state, embedding, dim)
    let proto_count = map_get(state, "proto_count")
    let embed_dim = map_get(state, "embed_dim")
    let mut embeddings = map_get(state, "embeddings")
    let mut match_counts = map_get(state, "match_counts")
    let prev_id = map_get(state, "last_match_id")

    # Normalize the input
    let normed = vecmath.normalize(embedding, dim)

    # First observation: set embed_dim, create first prototype
    if proto_count == 0.0
        map_set(state, "embed_dim", dim)
        # Append normalized embedding to flat array
        let mut i = 0.0
        while i < dim
            push(embeddings, normed[i])
            i = i + 1.0
        end
        push(match_counts, 1.0)
        map_set(state, "embeddings", embeddings)
        map_set(state, "match_counts", match_counts)
        map_set(state, "proto_count", 1.0)
        map_set(state, "last_match_id", 0.0)
        map_set(state, "last_match_sim", 1.0)
        map_set(state, "prev_match_id", prev_id)
        map_set(state, "transition_detected", 0.0)
        return state
    end

    # Find best matching prototype
    let mut best_sim = -1.0
    let mut best_id = -1.0
    let mut p = 0.0
    while p < proto_count
        let proto_emb = vecmath.vec_extract(embeddings, p, dim)
        let sim = vecmath.cosine_sim(normed, proto_emb, dim)
        if sim > best_sim
            best_sim = sim
            best_id = p
        end
        p = p + 1.0
    end

    let mut new_proto_count = proto_count

    if best_sim >= MATCH_THRESHOLD
        # Match: EMA drift prototype toward observation
        let proto_emb = vecmath.vec_extract(embeddings, best_id, dim)
        let mut drifted = []
        let mut d = 0.0
        while d < dim
            let v = (1.0 - EMA_ALPHA) * proto_emb[d] + EMA_ALPHA * normed[d]
            push(drifted, v)
            d = d + 1.0
        end
        # Re-normalize drifted vector
        let drifted_norm = vecmath.normalize(drifted, dim)
        embeddings = vecmath.vec_insert(embeddings, best_id, dim, drifted_norm)
        match_counts[best_id] = match_counts[best_id] + 1.0
    else
        # No match: create new prototype
        let mut i = 0.0
        while i < dim
            push(embeddings, normed[i])
            i = i + 1.0
        end
        push(match_counts, 1.0)
        best_id = proto_count
        best_sim = 1.0
        new_proto_count = proto_count + 1.0
    end

    # Transition detection
    let mut transition = 0.0
    let mut t_count = map_get(state, "transition_count")
    if prev_id >= 0.0 && best_id != prev_id
        transition = 1.0
        t_count = t_count + 1.0
    end

    map_set(state, "embeddings", embeddings)
    map_set(state, "match_counts", match_counts)
    map_set(state, "proto_count", new_proto_count)
    map_set(state, "last_match_id", best_id)
    map_set(state, "last_match_sim", best_sim)
    map_set(state, "prev_match_id", prev_id)
    map_set(state, "transition_detected", transition)
    map_set(state, "transition_count", t_count)
    return state
end
```

**Step 4: Run test to verify it passes**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_proto.flow"
```
Expected: All PASS lines.

**Step 5: Commit**

```bash
git add lib/proto.flow tests/test_proto.flow
git commit -m "feat(octobrain): prototype store — match, grow, drift, transition detection"
```

---

### Task 3: Adaptive Embedding

Discover embedding dimension from data variance. Project raw input to learned space.

**Files:**
- Create: `lib/embed.flow`
- Create: `tests/test_embed.flow`

**Step 1: Write the test**

Create `tests/test_embed.flow`:

```flow
use "../lib/embed"

# --- Test: variance computation ---
# 3 observations, 2 dimensions
# dim 0: [1.0, 3.0, 5.0] → mean=3, var=8/3≈2.67
# dim 1: [2.0, 2.0, 2.0] → mean=2, var=0
let obs = [1.0, 2.0, 3.0, 2.0, 5.0, 2.0]
let vars = embed.compute_variance(obs, 3.0, 2.0)
if vars[0] > 2.0
    print("PASS dim 0 has high variance")
else
    print("FAIL dim 0 variance: got {vars[0]}")
end
if vars[1] < 0.001
    print("PASS dim 1 has zero variance")
else
    print("FAIL dim 1 variance: got {vars[1]}")
end

# --- Test: dimension discovery ---
# Only dim 0 is significant (> 1% of max)
let discovered = embed.discover_dim(vars, 2.0)
if discovered == 1.0
    print("PASS discovered 1 significant dimension")
else
    print("FAIL discovered dim: got {discovered}")
end

# Mixed significance
let vars2 = [10.0, 5.0, 0.05, 0.001]
let disc2 = embed.discover_dim(vars2, 4.0)
if disc2 == 3.0
    print("PASS discovered 3 significant dims (10, 5, 0.05 > 0.1)")
else
    print("FAIL discovered dim: got {disc2}")
end

# --- Test: identity projection (embed_dim == input_dim) ---
let mut state = embed.embed_new()
state = embed.embed_set_dims(state, 3.0, 3.0)
let projected = embed.embed_project(state, [2.0, 4.0, 6.0], 3.0)
# Identity projection should preserve values (after normalization)
if len(projected) == 3.0
    print("PASS projection produces correct dim")
else
    print("FAIL projection dim: got {len(projected)}")
end

print("--- embed tests complete ---")
```

**Step 2: Run test to verify it fails**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_embed.flow"
```

**Step 3: Implement embed.flow**

Create `lib/embed.flow`:

```flow
# embed.flow — Adaptive embedding with variance-based dimension discovery

use "../lib/vecmath"

let VARIANCE_THRESHOLD_RATIO = 0.01
let DISCOVERY_INTERVAL = 50.0

fn embed_new()
    let mut state = map_new()
    map_set(state, "input_dim", 0.0)
    map_set(state, "embed_dim", 0.0)
    map_set(state, "W_embed", [])
    map_set(state, "obs_count", 0.0)
    map_set(state, "obs_buffer", [])
    map_set(state, "buffer_size", 0.0)
    map_set(state, "max_buffer", 100.0)
    return state
end

fn embed_set_dims(state, input_dim, embed_dim)
    map_set(state, "input_dim", input_dim)
    map_set(state, "embed_dim", embed_dim)
    # Initialize W_embed as identity-like [input_dim × embed_dim]
    let mut W = []
    let mut r = 0.0
    while r < input_dim
        let mut c = 0.0
        while c < embed_dim
            if r == c
                push(W, 1.0)
            else
                push(W, 0.0)
            end
            c = c + 1.0
        end
        r = r + 1.0
    end
    map_set(state, "W_embed", W)
    return state
end

fn compute_variance(obs_flat, num_obs, input_dim)
    # obs_flat: [num_obs × input_dim] flattened
    # Returns: [input_dim] variances

    # Step 1: compute means
    let mut means = vecmath.vec_zeros(input_dim)
    let mut r = 0.0
    while r < num_obs
        let mut d = 0.0
        while d < input_dim
            means[d] = means[d] + obs_flat[r * input_dim + d]
            d = d + 1.0
        end
        r = r + 1.0
    end
    let mut d = 0.0
    while d < input_dim
        means[d] = means[d] / num_obs
        d = d + 1.0
    end

    # Step 2: compute variances
    let mut vars = vecmath.vec_zeros(input_dim)
    r = 0.0
    while r < num_obs
        d = 0.0
        while d < input_dim
            let diff = obs_flat[r * input_dim + d] - means[d]
            vars[d] = vars[d] + diff * diff
            d = d + 1.0
        end
        r = r + 1.0
    end
    d = 0.0
    while d < input_dim
        vars[d] = vars[d] / num_obs
        d = d + 1.0
    end

    return vars
end

fn discover_dim(variances, input_dim)
    # Count dimensions with variance > threshold_ratio * max_variance
    let mut max_var = 0.0
    let mut d = 0.0
    while d < input_dim
        if variances[d] > max_var
            max_var = variances[d]
        end
        d = d + 1.0
    end

    if max_var < 0.000001
        return 1.0
    end

    let threshold = max_var * VARIANCE_THRESHOLD_RATIO
    let mut count = 0.0
    d = 0.0
    while d < input_dim
        if variances[d] > threshold
            count = count + 1.0
        end
        d = d + 1.0
    end

    if count < 1.0
        return 1.0
    end
    return count
end

fn embed_project(state, raw_data, input_dim)
    let embed_dim = map_get(state, "embed_dim")
    let W = map_get(state, "W_embed")

    # Matrix multiply: raw [input_dim] × W [input_dim × embed_dim] → [embed_dim]
    let mut result = []
    let mut c = 0.0
    while c < embed_dim
        let mut val = 0.0
        let mut r = 0.0
        while r < input_dim
            val = val + raw_data[r] * W[r * embed_dim + c]
            r = r + 1.0
        end
        push(result, val)
        c = c + 1.0
    end

    # Normalize the projection
    return vecmath.normalize(result, embed_dim)
end

fn embed_buffer_obs(state, data, input_dim)
    let mut buffer = map_get(state, "obs_buffer")
    let mut buf_size = map_get(state, "buffer_size")
    let max_buf = map_get(state, "max_buffer")

    # Append observation to buffer (flat)
    let mut i = 0.0
    while i < input_dim
        push(buffer, data[i])
        i = i + 1.0
    end
    buf_size = buf_size + 1.0

    # Evict oldest if over max
    if buf_size > max_buf
        # Remove first input_dim elements
        let mut new_buffer = []
        i = input_dim
        while i < len(buffer)
            push(new_buffer, buffer[i])
            i = i + 1.0
        end
        buffer = new_buffer
        buf_size = buf_size - 1.0
    end

    map_set(state, "obs_buffer", buffer)
    map_set(state, "buffer_size", buf_size)
    map_set(state, "obs_count", map_get(state, "obs_count") + 1.0)
    return state
end
```

**Step 4: Run test to verify it passes**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_embed.flow"
```

**Step 5: Commit**

```bash
git add lib/embed.flow tests/test_embed.flow
git commit -m "feat(octobrain): adaptive embedding — variance discovery, projection, buffering"
```

---

### Task 4: Hyperedge Store

N-ary connections between prototype nodes with Hebbian permanence.

**Files:**
- Create: `lib/edges.flow`
- Create: `tests/test_edges.flow`

**Step 1: Write the test**

Create `tests/test_edges.flow`:

```flow
use "../lib/edges"

# --- Test: create edge store ---
let mut store = edges.edges_new()
let ec = map_get(store, "edge_count")
if ec == 0.0
    print("PASS empty edge store")
else
    print("FAIL edge_count: {ec}")
end

# --- Test: add a hyperedge (nodes 0,1,2) ---
store = edges.edges_add(store, [0.0, 1.0, 2.0], 1.0)
let ec2 = map_get(store, "edge_count")
if ec2 == 1.0
    print("PASS added first edge")
else
    print("FAIL edge_count after add: {ec2}")
end

# --- Test: adding same nodes strengthens existing edge ---
store = edges.edges_add(store, [0.0, 1.0, 2.0], 0.5)
let ec3 = map_get(store, "edge_count")
if ec3 == 1.0
    print("PASS duplicate nodes strengthen, not duplicate")
else
    print("FAIL edge_count after duplicate: {ec3}")
end

# Check permanence increased
let perms = map_get(store, "permanences")
if perms[0] > 0.3
    print("PASS permanence strengthened: {perms[0]:.3}")
else
    print("FAIL permanence: {perms[0]:.3}")
end

# --- Test: add different edge ---
store = edges.edges_add(store, [1.0, 3.0], 1.0)
let ec4 = map_get(store, "edge_count")
if ec4 == 2.0
    print("PASS added second edge (different nodes)")
else
    print("FAIL edge_count: {ec4}")
end

# --- Test: query by context overlap ---
let matches = edges.edges_query(store, [0.0, 1.0])
if len(matches) >= 1.0
    print("PASS query found matching edges")
else
    print("FAIL query returned {len(matches)} matches")
end

# --- Test: decay ---
store = edges.edges_decay(store, 0.5)
let perms2 = map_get(store, "permanences")
if perms2[0] < perms[0]
    print("PASS decay reduced permanence")
else
    print("FAIL decay did not reduce permanence")
end

print("--- edges tests complete ---")
```

**Step 2: Run test to verify it fails**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_edges.flow"
```

**Step 3: Implement edges.flow**

Create `lib/edges.flow`:

```flow
# edges.flow — Hyperedge store for N-ary prototype connections
# Edges stored as parallel arrays: nodes (flat), arities, permanences, weights

let INITIAL_PERMANENCE = 0.3
let PERMANENCE_BONUS = 0.02

fn edges_new()
    let mut store = map_new()
    map_set(store, "edge_count", 0.0)
    map_set(store, "nodes", [])
    map_set(store, "arities", [])
    map_set(store, "offsets", [])
    map_set(store, "permanences", [])
    map_set(store, "weights", [])
    map_set(store, "activations", [])
    return store
end

fn _edge_key(node_ids)
    # Sort node IDs and create a string key for dedup
    let sorted = sort_array(node_ids)
    let mut key = ""
    let mut i = 0.0
    while i < len(sorted)
        if i > 0.0
            key = key + ","
        end
        key = key + str(int(sorted[i]))
        i = i + 1.0
    end
    return key
end

fn _find_edge(store, key)
    # Linear scan for matching edge key
    let edge_count = map_get(store, "edge_count")
    let nodes = map_get(store, "nodes")
    let arities = map_get(store, "arities")
    let offsets = map_get(store, "offsets")

    let mut e = 0.0
    while e < edge_count
        let arity = arities[e]
        let offset = offsets[e]
        # Reconstruct this edge's node list
        let mut edge_nodes = []
        let mut n = 0.0
        while n < arity
            push(edge_nodes, nodes[offset + n])
            n = n + 1.0
        end
        let ekey = _edge_key(edge_nodes)
        if ekey == key
            return e
        end
        e = e + 1.0
    end
    return -1.0
end

fn edges_add(store, node_ids, weight)
    let key = _edge_key(node_ids)
    let existing = _find_edge(store, key)

    if existing >= 0.0
        # Strengthen existing edge
        let mut perms = map_get(store, "permanences")
        let mut weights = map_get(store, "weights")
        let mut acts = map_get(store, "activations")

        let old_perm = perms[existing]
        let bonus = PERMANENCE_BONUS * weight
        let mut new_perm = old_perm + bonus
        if new_perm > 1.0
            new_perm = 1.0
        end
        perms[existing] = new_perm
        weights[existing] = weights[existing] + weight
        acts[existing] = acts[existing] + 1.0

        map_set(store, "permanences", perms)
        map_set(store, "weights", weights)
        map_set(store, "activations", acts)
        return store
    end

    # New edge
    let mut nodes = map_get(store, "nodes")
    let mut arities = map_get(store, "arities")
    let mut offsets = map_get(store, "offsets")
    let mut perms = map_get(store, "permanences")
    let mut weights_arr = map_get(store, "weights")
    let mut acts = map_get(store, "activations")

    let offset = len(nodes)
    let arity = len(node_ids)
    let sorted = sort_array(node_ids)

    let mut i = 0.0
    while i < arity
        push(nodes, sorted[i])
        i = i + 1.0
    end

    push(arities, arity)
    push(offsets, offset)
    push(perms, INITIAL_PERMANENCE)
    push(weights_arr, weight)
    push(acts, 1.0)

    map_set(store, "nodes", nodes)
    map_set(store, "arities", arities)
    map_set(store, "offsets", offsets)
    map_set(store, "permanences", perms)
    map_set(store, "weights", weights_arr)
    map_set(store, "activations", acts)
    map_set(store, "edge_count", map_get(store, "edge_count") + 1.0)
    return store
end

fn edges_query(store, context_nodes)
    # Find edges that overlap with context_nodes
    # Returns array of [edge_index, overlap_fraction, permanence]
    let edge_count = map_get(store, "edge_count")
    let nodes = map_get(store, "nodes")
    let arities = map_get(store, "arities")
    let offsets_arr = map_get(store, "offsets")
    let perms = map_get(store, "permanences")

    let mut results = []
    let ctx_len = len(context_nodes)

    let mut e = 0.0
    while e < edge_count
        let arity = arities[e]
        let offset = offsets_arr[e]

        # Count overlap
        let mut overlap = 0.0
        let mut c = 0.0
        while c < ctx_len
            let ctx_node = context_nodes[c]
            let mut n = 0.0
            while n < arity
                if nodes[offset + n] == ctx_node
                    overlap = overlap + 1.0
                end
                n = n + 1.0
            end
            c = c + 1.0
        end

        if overlap > 0.0
            # Store as flat triple: [edge_idx, overlap/arity, permanence]
            push(results, e)
            push(results, overlap / arity)
            push(results, perms[e])
        end
        e = e + 1.0
    end

    return results
end

fn edges_decay(store, factor)
    let mut perms = map_get(store, "permanences")
    let edge_count = map_get(store, "edge_count")
    let mut e = 0.0
    while e < edge_count
        perms[e] = perms[e] * factor
        e = e + 1.0
    end
    map_set(store, "permanences", perms)
    return store
end

fn edges_get_nodes(store, edge_index)
    # Get node IDs for a specific edge
    let nodes = map_get(store, "nodes")
    let arities = map_get(store, "arities")
    let offsets_arr = map_get(store, "offsets")

    let arity = arities[edge_index]
    let offset = offsets_arr[edge_index]

    let mut result = []
    let mut n = 0.0
    while n < arity
        push(result, nodes[offset + n])
        n = n + 1.0
    end
    return result
end
```

**Step 4: Run test to verify it passes**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_edges.flow"
```

**Step 5: Commit**

```bash
git add lib/edges.flow tests/test_edges.flow
git commit -m "feat(octobrain): hyperedge store — N-ary edges, overlap query, decay"
```

---

### Task 5: Hebbian Learning (Oja's Rule)

Strengthen hyperedges between co-occurring prototypes using pairwise correlation.

**Files:**
- Create: `lib/hebbian.flow`
- Create: `tests/test_hebbian.flow`

**Step 1: Write the test**

Create `tests/test_hebbian.flow`:

```flow
use "../lib/vecmath"
use "../lib/hebbian"

# --- Test: pairwise correlation of aligned vectors ---
# Two identical unit vectors → correlation = 1.0
let embs = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
let corr = hebbian.avg_pairwise_corr(embs, 2.0, 3.0)
if abs(corr - 1.0) < 0.001
    print("PASS identical vectors correlation = 1.0")
else
    print("FAIL correlation: got {corr}")
end

# Two orthogonal vectors → correlation = 0.0
let embs2 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
let corr2 = hebbian.avg_pairwise_corr(embs2, 2.0, 3.0)
if abs(corr2) < 0.001
    print("PASS orthogonal vectors correlation = 0.0")
else
    print("FAIL correlation: got {corr2}")
end

# --- Test: Oja's permanence update ---
# High correlation + low permanence → increase
let p1 = hebbian.oja_update(0.3, 0.8, 0.1, 0.01, 1.0)
if p1 > 0.3
    print("PASS permanence increased from {p1:.4}")
else
    print("FAIL permanence: got {p1}")
end

# Low correlation + high permanence → decrease
let p2 = hebbian.oja_update(0.9, 0.1, 0.5, 0.01, 1.0)
if p2 < 0.9
    print("PASS permanence decreased from {p2:.4}")
else
    print("FAIL permanence: got {p2}")
end

# Permanence stays clamped [0, 1]
let p3 = hebbian.oja_update(0.99, 1.0, 0.0, 1.0, 5.0)
if p3 <= 1.0
    print("PASS permanence clamped to 1.0")
else
    print("FAIL permanence over 1: got {p3}")
end

print("--- hebbian tests complete ---")
```

**Step 2: Run test to verify it fails**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_hebbian.flow"
```

**Step 3: Implement hebbian.flow**

Create `lib/hebbian.flow`:

```flow
# hebbian.flow — Oja's rule for hyperedge permanence learning

use "../lib/vecmath"

let HEBBIAN_LR = 0.01
let HEBBIAN_BONUS = 0.02

fn avg_pairwise_corr(embeddings_flat, num_nodes, embed_dim)
    # Compute average pairwise dot product (cosine) between all node pairs
    # embeddings_flat: [num_nodes × embed_dim] flattened
    let mut total_corr = 0.0
    let mut pair_count = 0.0

    let mut i = 0.0
    while i < num_nodes
        let ei = vecmath.vec_extract(embeddings_flat, i, embed_dim)
        let mut j = i + 1.0
        while j < num_nodes
            let ej = vecmath.vec_extract(embeddings_flat, j, embed_dim)
            let corr = vecmath.cosine_sim(ei, ej, embed_dim)
            total_corr = total_corr + corr
            pair_count = pair_count + 1.0
            j = j + 1.0
        end
        i = i + 1.0
    end

    if pair_count < 1.0
        return 0.0
    end
    return total_corr / pair_count
end

fn mean_embedding(embeddings_flat, num_nodes, embed_dim)
    let mut mean = vecmath.vec_zeros(embed_dim)
    let mut i = 0.0
    while i < num_nodes
        let mut d = 0.0
        while d < embed_dim
            mean[d] = mean[d] + embeddings_flat[i * embed_dim + d]
            d = d + 1.0
        end
        i = i + 1.0
    end
    let mut d = 0.0
    while d < embed_dim
        mean[d] = mean[d] / num_nodes
        d = d + 1.0
    end
    return mean
end

fn oja_update(permanence, avg_corr, mean_sq, lr, weight)
    # Oja's rule adapted for hyperedges
    # delta = lr * weight * (avg_correlation - mean_sq * permanence) + bonus
    let delta = lr * weight * (avg_corr - mean_sq * permanence) + HEBBIAN_BONUS * weight
    let mut new_perm = permanence + delta
    if new_perm > 1.0
        new_perm = 1.0
    end
    if new_perm < 0.0
        new_perm = 0.0
    end
    return new_perm
end

fn learn_edge(edge_store, proto_store, node_ids, weight)
    # Full Hebbian learning for one hyperedge
    # 1. Gather embeddings for all nodes
    # 2. Compute pairwise correlation
    # 3. Compute mean squared
    # 4. Apply Oja's update to permanence

    let embed_dim = map_get(proto_store, "embed_dim")
    let protos = map_get(proto_store, "embeddings")
    let num_nodes = len(node_ids)

    # Gather embeddings
    let mut gathered = []
    let mut i = 0.0
    while i < num_nodes
        let nid = node_ids[i]
        let emb = vecmath.vec_extract(protos, nid, embed_dim)
        let mut d = 0.0
        while d < embed_dim
            push(gathered, emb[d])
            d = d + 1.0
        end
        i = i + 1.0
    end

    # Pairwise correlation
    let corr = avg_pairwise_corr(gathered, num_nodes, embed_dim)

    # Mean embedding and its squared norm
    let mean = mean_embedding(gathered, num_nodes, embed_dim)
    let mean_sq = vecmath.dot_product(mean, mean, embed_dim)

    # Add/strengthen edge with Oja-computed permanence
    # First, add the edge (or find existing)
    let edges = require("../lib/edges")
    edge_store = edges.edges_add(edge_store, node_ids, weight)

    # Now update permanence with Oja's rule
    let edge_count = map_get(edge_store, "edge_count")
    let mut perms = map_get(edge_store, "permanences")

    # The edge we just added/strengthened is the one matching node_ids
    # Find it by checking the last-modified edge
    let idx = edge_count - 1.0
    let old_perm = perms[idx]
    let new_perm = oja_update(old_perm, corr, mean_sq, HEBBIAN_LR, weight)
    perms[idx] = new_perm
    map_set(edge_store, "permanences", perms)

    return edge_store
end
```

**Step 4: Run test to verify it passes**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_hebbian.flow"
```

**Step 5: Commit**

```bash
git add lib/hebbian.flow tests/test_hebbian.flow
git commit -m "feat(octobrain): Hebbian learning — Oja's rule, pairwise correlation"
```

---

### Task 6: Hopfield Recall

Given context prototypes, score all prototypes and project to action space.

**Files:**
- Create: `lib/recall.flow`
- Create: `tests/test_recall.flow`

**Step 1: Write the test**

Create `tests/test_recall.flow`:

```flow
use "../lib/vecmath"
use "../lib/proto"
use "../lib/edges"
use "../lib/recall"

# Setup: create a brain with known prototypes and edges
let mut ps = proto.proto_new()
# Proto 0: [1,0,0]
ps = proto.proto_observe(ps, [1.0, 0.0, 0.0], 3.0)
# Proto 1: [0,1,0]
ps = proto.proto_observe(ps, [0.0, 1.0, 0.0], 3.0)
# Proto 2: [0,0,1]
ps = proto.proto_observe(ps, [0.0, 0.0, 1.0], 3.0)

let mut es = edges.edges_new()
# Edge: proto 0 and 1 co-occur (high permanence)
es = edges.edges_add(es, [0.0, 1.0], 5.0)
# Edge: proto 1 and 2 co-occur (low permanence)
es = edges.edges_add(es, [1.0, 2.0], 1.0)

# --- Test: context mean ---
let window = [0.0, 1.0]
let embed_dim = map_get(ps, "embed_dim")
let protos = map_get(ps, "embeddings")
let ctx_mean = recall.context_mean(protos, window, embed_dim)
# Mean of [1,0,0] and [0,1,0] = [0.5, 0.5, 0]
if abs(ctx_mean[0] - 0.5) < 0.01 && abs(ctx_mean[1] - 0.5) < 0.01
    print("PASS context_mean correct")
else
    print("FAIL context_mean: [{ctx_mean[0]:.2}, {ctx_mean[1]:.2}, {ctx_mean[2]:.2}]")
end

# --- Test: score protos ---
let scores = recall.score_protos(protos, ctx_mean, 3.0, 3.0)
# Protos 0 and 1 should score higher than proto 2 (closer to context mean)
if scores[0] > scores[2] && scores[1] > scores[2]
    print("PASS protos 0,1 score higher than proto 2")
else
    print("FAIL scores: [{scores[0]:.3}, {scores[1]:.3}, {scores[2]:.3}]")
end

# --- Test: action projection ---
# Simple W_score: [embed_dim × action_count] = [3 × 2]
# Action 0 = proto dim 0, Action 1 = proto dim 1
let W_score = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
let actions = recall.project_actions(ctx_mean, W_score, 3.0, 2.0)
if len(actions) == 2.0
    print("PASS action projection has correct dim")
else
    print("FAIL action dim: {len(actions)}")
end

print("--- recall tests complete ---")
```

**Step 2: Run test to verify it fails**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_recall.flow"
```

**Step 3: Implement recall.flow**

Create `lib/recall.flow`:

```flow
# recall.flow — Hopfield completion: context → action scores

use "../lib/vecmath"

fn context_mean(protos_flat, window, embed_dim)
    # Compute mean embedding of prototype IDs in window
    let num = len(window)
    let mut mean = vecmath.vec_zeros(embed_dim)

    let mut i = 0.0
    while i < num
        let pid = window[i]
        let emb = vecmath.vec_extract(protos_flat, pid, embed_dim)
        let mut d = 0.0
        while d < embed_dim
            mean[d] = mean[d] + emb[d]
            d = d + 1.0
        end
        i = i + 1.0
    end

    let mut d = 0.0
    while d < embed_dim
        mean[d] = mean[d] / num
        d = d + 1.0
    end
    return mean
end

fn score_protos(protos_flat, query, embed_dim, proto_count)
    # Score all prototypes by cosine similarity to query
    let mut scores = []
    let mut p = 0.0
    while p < proto_count
        let emb = vecmath.vec_extract(protos_flat, p, embed_dim)
        let sim = vecmath.cosine_sim(query, emb, embed_dim)
        push(scores, sim)
        p = p + 1.0
    end
    return scores
end

fn weighted_score(protos_flat, query, embed_dim, proto_count, edge_store)
    # Score protos by cosine similarity, weighted by hyperedge permanence
    let base_scores = score_protos(protos_flat, query, embed_dim, proto_count)

    # Collect edge weights per proto
    let edge_count = map_get(edge_store, "edge_count")
    let nodes = map_get(edge_store, "nodes")
    let arities = map_get(edge_store, "arities")
    let offsets = map_get(edge_store, "offsets")
    let perms = map_get(edge_store, "permanences")

    let mut edge_weight = vecmath.vec_zeros(proto_count)

    let mut e = 0.0
    while e < edge_count
        let arity = arities[e]
        let offset = offsets[e]
        let perm = perms[e]
        let mut n = 0.0
        while n < arity
            let nid = nodes[offset + n]
            if nid < proto_count
                edge_weight[nid] = edge_weight[nid] + perm
            end
            n = n + 1.0
        end
        e = e + 1.0
    end

    # Combine: cosine × (1 + edge_weight)
    let mut final_scores = []
    let mut p = 0.0
    while p < proto_count
        let combined = base_scores[p] * (1.0 + edge_weight[p])
        push(final_scores, combined)
        p = p + 1.0
    end
    return final_scores
end

fn project_actions(embedding, W_score, embed_dim, action_count)
    # Linear projection: embedding [embed_dim] × W_score [embed_dim × action_count]
    let mut actions = []
    let mut a = 0.0
    while a < action_count
        let mut val = 0.0
        let mut d = 0.0
        while d < embed_dim
            val = val + embedding[d] * W_score[d * action_count + a]
            d = d + 1.0
        end
        push(actions, val)
        a = a + 1.0
    end
    return actions
end

fn recall(proto_store, edge_store, window, action_count, W_score)
    # Full recall pipeline: context → scores → actions
    let embed_dim = map_get(proto_store, "embed_dim")
    let proto_count = map_get(proto_store, "proto_count")
    let protos = map_get(proto_store, "embeddings")

    if proto_count < 1.0 || len(window) < 1.0
        return vecmath.vec_zeros(action_count)
    end

    # 1. Context mean embedding
    let ctx = context_mean(protos, window, embed_dim)

    # 2. Score protos weighted by edges
    let scores = weighted_score(protos, ctx, embed_dim, proto_count, edge_store)

    # 3. Weighted mean of top-scoring proto embeddings
    # Use scores as weights for a soft attention over proto embeddings
    let mut weighted_emb = vecmath.vec_zeros(embed_dim)
    let mut total_weight = 0.0
    let mut p = 0.0
    while p < proto_count
        if scores[p] > 0.0
            let emb = vecmath.vec_extract(protos, p, embed_dim)
            let mut d = 0.0
            while d < embed_dim
                weighted_emb[d] = weighted_emb[d] + emb[d] * scores[p]
                d = d + 1.0
            end
            total_weight = total_weight + scores[p]
        end
        p = p + 1.0
    end
    if total_weight > 0.0
        let mut d = 0.0
        while d < embed_dim
            weighted_emb[d] = weighted_emb[d] / total_weight
            d = d + 1.0
        end
    end

    # 4. Project to action space
    return project_actions(weighted_emb, W_score, embed_dim, action_count)
end
```

**Step 4: Run test to verify it passes**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_recall.flow"
```

**Step 5: Commit**

```bash
git add lib/recall.flow tests/test_recall.flow
git commit -m "feat(octobrain): Hopfield recall — context mean, weighted scoring, action projection"
```

---

### Task 7: Plasticity

Pattern drift and edge decay for continuous adaptation.

**Files:**
- Create: `lib/plasticity.flow`
- Create: `tests/test_plasticity.flow`

**Step 1: Write the test**

Create `tests/test_plasticity.flow`:

```flow
use "../lib/vecmath"
use "../lib/proto"
use "../lib/edges"
use "../lib/plasticity"

# Setup
let mut ps = proto.proto_new()
ps = proto.proto_observe(ps, [1.0, 0.0, 0.0], 3.0)
ps = proto.proto_observe(ps, [0.0, 1.0, 0.0], 3.0)

# --- Test: drift proto toward observation ---
let observed = [0.7, 0.3, 0.0]
ps = plasticity.drift_proto(ps, 0.0, observed, 0.1, 1.0)
let protos = map_get(ps, "embeddings")
let p0_0 = protos[0]
# Proto 0 was [1,0,0], observed [0.7,0.3,0], should drift slightly
if p0_0 < 1.0 && p0_0 > 0.8
    print("PASS drift moved proto toward observation")
else
    print("FAIL drift: p0[0] = {p0_0}")
end

# --- Test: importance slows drift ---
# Second drift with higher importance → less movement
let p0_before = protos[0]
ps = plasticity.drift_proto(ps, 0.0, observed, 0.1, 10.0)
let protos2 = map_get(ps, "embeddings")
let p0_after = protos2[0]
let delta1 = abs(1.0 - p0_before)
let delta2 = abs(p0_before - p0_after)
if delta2 < delta1
    print("PASS higher importance → smaller drift")
else
    print("FAIL importance damping: delta1={delta1:.4}, delta2={delta2:.4}")
end

# --- Test: edge decay ---
let mut es = edges.edges_new()
es = edges.edges_add(es, [0.0, 1.0], 1.0)
let perm_before = map_get(es, "permanences")
es = plasticity.decay_all_edges(es, 0.995)
let perm_after = map_get(es, "permanences")
if perm_after[0] < perm_before[0]
    print("PASS edge decay reduced permanence")
else
    print("FAIL edge decay")
end

print("--- plasticity tests complete ---")
```

**Step 2: Run test, then implement, then verify**

Create `lib/plasticity.flow`:

```flow
# plasticity.flow — Continuous adaptation: pattern drift + edge decay

use "../lib/vecmath"
use "../lib/edges"

fn drift_proto(proto_store, proto_id, observed, lr, importance)
    # Drift prototype toward observed embedding
    # effective_lr = lr / (1 + importance)
    let embed_dim = map_get(proto_store, "embed_dim")
    let mut protos = map_get(proto_store, "embeddings")

    let effective_lr = lr / (1.0 + importance)
    let proto_emb = vecmath.vec_extract(protos, proto_id, embed_dim)
    let normed_obs = vecmath.normalize(observed, embed_dim)

    # Drift: proto += lr * (observed - proto)
    let diff = vecmath.vec_subtract(normed_obs, proto_emb, embed_dim)
    let delta = vecmath.vec_scale(diff, effective_lr, embed_dim)
    let drifted = vecmath.vec_add(proto_emb, delta, embed_dim)
    let drifted_norm = vecmath.normalize(drifted, embed_dim)

    protos = vecmath.vec_insert(protos, proto_id, embed_dim, drifted_norm)
    map_set(proto_store, "embeddings", protos)
    return proto_store
end

fn decay_all_edges(edge_store, factor)
    return edges.edges_decay(edge_store, factor)
end

fn homeostasis_check(proto_store)
    # Check for overloaded or redundant prototypes
    # Returns: map with "splits" and "merges" arrays
    let proto_count = map_get(proto_store, "proto_count")
    let counts = map_get(proto_store, "match_counts")
    let embed_dim = map_get(proto_store, "embed_dim")
    let protos = map_get(proto_store, "embeddings")

    # Total utilization
    let mut total = 0.0
    let mut p = 0.0
    while p < proto_count
        total = total + counts[p]
        p = p + 1.0
    end

    let mut result = map_new()
    let mut splits = []
    let mut merges = []

    if total < 1.0
        map_set(result, "splits", splits)
        map_set(result, "merges", merges)
        return result
    end

    # Check for overloaded (> 20% of total)
    p = 0.0
    while p < proto_count
        if counts[p] / total > 0.20
            push(splits, p)
        end
        p = p + 1.0
    end

    # Check for redundant (cosine > 0.95)
    let mut i = 0.0
    while i < proto_count
        let ei = vecmath.vec_extract(protos, i, embed_dim)
        let mut j = i + 1.0
        while j < proto_count
            let ej = vecmath.vec_extract(protos, j, embed_dim)
            let sim = vecmath.cosine_sim(ei, ej, embed_dim)
            if sim > 0.95
                push(merges, i)
                push(merges, j)
            end
            j = j + 1.0
        end
        i = i + 1.0
    end

    map_set(result, "splits", splits)
    map_set(result, "merges", merges)
    return result
end
```

**Step 3: Run test, commit**

```bash
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_plasticity.flow"
git add lib/plasticity.flow tests/test_plasticity.flow
git commit -m "feat(octobrain): plasticity — pattern drift, edge decay, homeostasis check"
```

---

### Task 8: Public API + Integration Test + Sine Wave Demo

Wire everything together into the public `octobrain` module and prove it works end-to-end.

**Files:**
- Create: `lib/octobrain.flow`
- Create: `tests/test_brain.flow`
- Create: `examples/sine_wave.flow`

**Step 1: Write the integration test**

Create `tests/test_brain.flow`:

```flow
use "../lib/octobrain"

# --- Test: create brain ---
let mut brain = octobrain.octobrain_new(2.0)
let ac = map_get(brain, "action_count")
if ac == 2.0
    print("PASS brain created with 2 actions")
else
    print("FAIL action_count: {ac}")
end

# --- Test: observe data ---
brain = octobrain.octobrain_observe(brain, [1.0, 0.0])
brain = octobrain.octobrain_observe(brain, [0.9, 0.1])
brain = octobrain.octobrain_observe(brain, [0.95, 0.05])
let pc = map_get(map_get(brain, "proto_store"), "proto_count")
if pc >= 1.0
    print("PASS prototypes created: {pc}")
else
    print("FAIL no prototypes")
end

# --- Test: different cluster ---
brain = octobrain.octobrain_observe(brain, [0.0, 1.0])
brain = octobrain.octobrain_observe(brain, [0.1, 0.9])
let pc2 = map_get(map_get(brain, "proto_store"), "proto_count")
if pc2 >= 2.0
    print("PASS second cluster: {pc2} protos")
else
    print("FAIL only {pc2} protos")
end

# --- Test: transition detection ---
let td = octobrain.octobrain_transition_detected(brain)
print("Transition detected: {td}")

# --- Test: recall ---
let scores = octobrain.octobrain_recall(brain)
if len(scores) == 2.0
    print("PASS recall returns 2 action scores: [{scores[0]:.3}, {scores[1]:.3}]")
else
    print("FAIL recall dim: {len(scores)}")
end

# --- Test: teach ---
brain = octobrain.octobrain_teach(brain, 0.0, 1.0, 1.0)
let ec = map_get(map_get(brain, "edge_store"), "edge_count")
print("Edges after teach: {ec}")

# --- Test: stats ---
let stats = octobrain.octobrain_stats(brain)
print("Stats: embed_dim={map_get(stats, \"embed_dim\")}, protos={map_get(stats, \"proto_count\")}, edges={map_get(stats, \"edge_count\")}")

print("--- brain integration tests complete ---")
```

**Step 2: Implement octobrain.flow**

Create `lib/octobrain.flow`:

```flow
# octobrain.flow — Public API for the skeleton-free adaptive brain

use "../lib/vecmath"
use "../lib/embed"
use "../lib/proto"
use "../lib/edges"
use "../lib/hebbian"
use "../lib/recall"
use "../lib/plasticity"

let PROCESS_WINDOW_SIZE = 10.0
let DECAY_INTERVAL = 50.0
let PLASTICITY_LR = 0.03
let DECAY_FACTOR = 0.995

fn octobrain_new(action_count)
    let mut brain = map_new()
    map_set(brain, "action_count", action_count)
    map_set(brain, "observation_count", 0.0)
    map_set(brain, "embed_store", embed.embed_new())
    map_set(brain, "proto_store", proto.proto_new())
    map_set(brain, "edge_store", edges.edges_new())
    map_set(brain, "process_window", [])
    map_set(brain, "W_score", [])
    map_set(brain, "initialized", 0.0)
    return brain
end

fn _init_W_score(embed_dim, action_count)
    # Initialize action projection weights (small random-like values)
    let mut W = []
    let total = embed_dim * action_count
    let mut i = 0.0
    while i < total
        # Deterministic pseudo-random: use position-based pattern
        let val = sin(i * 1.618 + 0.5) * 0.1
        push(W, val)
        i = i + 1.0
    end
    return W
end

fn octobrain_observe(brain, data)
    let input_dim = len(data)
    let mut obs_count = map_get(brain, "observation_count")
    let action_count = map_get(brain, "action_count")
    let mut es = map_get(brain, "embed_store")
    let mut ps = map_get(brain, "proto_store")
    let mut edge_store = map_get(brain, "edge_store")
    let mut window = map_get(brain, "process_window")

    # First observation: initialize dimensions
    if map_get(brain, "initialized") == 0.0
        es = embed.embed_set_dims(es, input_dim, input_dim)
        map_set(brain, "W_score", _init_W_score(input_dim, action_count))
        map_set(brain, "initialized", 1.0)
    end

    # Buffer observation for variance analysis
    es = embed.embed_buffer_obs(es, data, input_dim)

    # Periodic dimension discovery
    let buf_size = map_get(es, "buffer_size")
    if buf_size >= 10.0 && int(obs_count) % 50 == 0
        let buffer = map_get(es, "obs_buffer")
        let vars = embed.compute_variance(buffer, buf_size, input_dim)
        let new_dim = embed.discover_dim(vars, input_dim)
        let old_dim = map_get(es, "embed_dim")
        if new_dim != old_dim
            es = embed.embed_set_dims(es, input_dim, new_dim)
            map_set(brain, "W_score", _init_W_score(new_dim, action_count))
        end
    end

    # Embed: project to discovered space
    let embed_dim = map_get(es, "embed_dim")
    let embedding = embed.embed_project(es, data, input_dim)

    # Match: find nearest prototype
    ps = proto.proto_observe(ps, embedding, embed_dim)
    let matched_id = map_get(ps, "last_match_id")
    let transition = map_get(ps, "transition_detected")

    # Update process window
    push(window, matched_id)
    if len(window) > PROCESS_WINDOW_SIZE
        let mut new_window = []
        let mut i = 1.0
        while i < len(window)
            push(new_window, window[i])
            i = i + 1.0
        end
        window = new_window
    end

    # On transition: learn hyperedge between recent prototypes
    if transition == 1.0 && len(window) >= 2.0
        # Learn co-occurrence of last few protos
        let mut unique_nodes = []
        let mut seen = map_new()
        let mut i = 0.0
        while i < len(window)
            let nid = str(int(window[i]))
            if map_has(seen, nid) == 0.0
                push(unique_nodes, window[i])
                map_set(seen, nid, 1.0)
            end
            i = i + 1.0
        end
        if len(unique_nodes) >= 2.0
            edge_store = edges.edges_add(edge_store, unique_nodes, 1.0)
        end
    end

    # Plasticity: drift matched prototype
    if matched_id >= 0.0
        let importance = map_get(ps, "match_counts")
        let imp = importance[matched_id]
        ps = plasticity.drift_proto(ps, matched_id, embedding, PLASTICITY_LR, imp)
    end

    # Periodic edge decay
    obs_count = obs_count + 1.0
    if int(obs_count) % int(DECAY_INTERVAL) == 0
        edge_store = plasticity.decay_all_edges(edge_store, DECAY_FACTOR)
    end

    map_set(brain, "observation_count", obs_count)
    map_set(brain, "embed_store", es)
    map_set(brain, "proto_store", ps)
    map_set(brain, "edge_store", edge_store)
    map_set(brain, "process_window", window)
    return brain
end

fn octobrain_recall(brain)
    let ps = map_get(brain, "proto_store")
    let es = map_get(brain, "edge_store")
    let window = map_get(brain, "process_window")
    let action_count = map_get(brain, "action_count")
    let W_score = map_get(brain, "W_score")
    let embed_dim = map_get(map_get(brain, "embed_store"), "embed_dim")

    return recall.recall(ps, es, window, action_count, W_score)
end

fn octobrain_teach(brain, action_id, outcome, weight)
    let mut edge_store = map_get(brain, "edge_store")
    let window = map_get(brain, "process_window")

    # Create outcome-linked hyperedge: window protos + action → outcome
    if len(window) >= 1.0
        let mut teach_nodes = []
        let mut i = 0.0
        while i < len(window)
            push(teach_nodes, window[i])
            i = i + 1.0
        end
        # Encode action and outcome as high node IDs (offset to avoid collision)
        let action_node = 10000.0 + action_id
        let outcome_node = 20000.0 + outcome
        push(teach_nodes, action_node)
        push(teach_nodes, outcome_node)

        let teach_weight = weight * abs(outcome)
        if teach_weight < 0.1
            teach_weight = 0.1
        end
        edge_store = edges.edges_add(edge_store, teach_nodes, teach_weight)
    end

    map_set(brain, "edge_store", edge_store)
    return brain
end

fn octobrain_transition_detected(brain)
    let ps = map_get(brain, "proto_store")
    return map_get(ps, "transition_detected")
end

fn octobrain_stats(brain)
    let mut stats = map_new()
    let es = map_get(brain, "embed_store")
    let ps = map_get(brain, "proto_store")
    let edge_store = map_get(brain, "edge_store")

    map_set(stats, "embed_dim", map_get(es, "embed_dim"))
    map_set(stats, "input_dim", map_get(es, "input_dim"))
    map_set(stats, "proto_count", map_get(ps, "proto_count"))
    map_set(stats, "edge_count", map_get(edge_store, "edge_count"))
    map_set(stats, "observation_count", map_get(brain, "observation_count"))
    map_set(stats, "transition_count", map_get(ps, "transition_count"))
    return stats
end
```

**Step 3: Write the sine wave demo**

Create `examples/sine_wave.flow`:

```flow
# sine_wave.flow — OctoBrain discovers periodicity in sin(t)
# Expected: brain discovers 2-4 prototypes corresponding to
# phases of the sine wave, transitions at inflection points.

use "../lib/octobrain"

# 2 actions: "going up" vs "going down"
let mut brain = octobrain.octobrain_new(2.0)

let mut t = 0.0
let mut transitions = 0.0
let mut last_print = 0.0

print("=== OctoBrain Sine Wave Discovery ===")
print("")

while t < 500.0
    # Generate sine observation: [value, derivative]
    let val = sin(t * 0.1)
    let deriv = cos(t * 0.1) * 0.1

    brain = octobrain.octobrain_observe(brain, [val, deriv])

    if octobrain.octobrain_transition_detected(brain) == 1.0
        transitions = transitions + 1.0
    end

    # Print status every 100 observations
    if t - last_print >= 100.0
        let stats = octobrain.octobrain_stats(brain)
        let protos = map_get(stats, "proto_count")
        let edgec = map_get(stats, "edge_count")
        let scores = octobrain.octobrain_recall(brain)

        print("t={t:.0}: protos={protos:.0}, edges={edgec:.0}, transitions={transitions:.0}")
        print("  scores: up={scores[0]:.4}, down={scores[1]:.4}")
        print("  val={val:.3}, deriv={deriv:.3}")
        last_print = t
    end

    t = t + 1.0
end

let final_stats = octobrain.octobrain_stats(brain)
print("")
print("=== Final Stats ===")
print("Embed dim:    {map_get(final_stats, \"embed_dim\"):.0}")
print("Prototypes:   {map_get(final_stats, \"proto_count\"):.0}")
print("Edges:        {map_get(final_stats, \"edge_count\"):.0}")
print("Observations: {map_get(final_stats, \"observation_count\"):.0}")
print("Transitions:  {map_get(final_stats, \"transition_count\"):.0}")
```

**Step 4: Run tests and demo**

```bash
# Integration test
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\tests\\test_brain.flow"

# Sine wave demo
$FLOWRUN run "C:\\OctoFlow\\OctoBrain\\examples\\sine_wave.flow"
```

Expected from sine demo:
- Discovers 2-4 prototypes (peak, trough, ascending, descending phases)
- Transitions correlate with sine zero-crossings and extrema
- Action scores differentiate between up and down phases

**Step 5: Commit**

```bash
git add lib/octobrain.flow tests/test_brain.flow examples/sine_wave.flow
git commit -m "feat(octobrain): public API + integration test + sine wave demo

Phase 1 complete: skeleton-free adaptive brain on CPU.
Brain discovers prototypes, detects transitions, learns
via Hebbian edges, recalls via Hopfield completion."
```

---

## Validation Gate

Phase 1 is complete when:

1. All 6 test files pass (vecmath, proto, embed, edges, hebbian, recall, plasticity, brain)
2. Sine wave demo runs and shows:
   - 2+ discovered prototypes
   - Transitions at phase changes
   - Non-zero action scores that differ between up/down phases
3. No hardcoded dimensions — all discovered from data
4. All state stored in maps — ready for .octo persistence in Phase 6

## Next Phase

After Phase 1 passes validation, Phase 2 moves prototype matching to GPU via Loom JIT kernels. The CPU implementation from Phase 1 becomes the reference for correctness validation.
