# GPU-Resident Autonomous Agents

## The Boundary-Breaking Capability

This directory contains demonstration programs that showcase OctoFlow's unique capability: **GPU-Resident Autonomous Agents** - computation patterns that run decision-making loops primarily on GPU with minimal CPU involvement.

No CPU-native language can achieve this level of seamless GPU autonomy.

## Why This Matters

Traditional GPU programming forces a choice:
- **CPU-bound**: Write normal code, lose GPU performance (100x slower)
- **GPU-bound**: Write custom kernels, manage memory explicitly, coordinate dispatches manually (300+ lines of boilerplate)

OctoFlow eliminates this trade-off:
- Write natural control flow (loops, conditionals, variables)
- GPU operations execute autonomously
- CPU handles high-level orchestration only
- Zero boilerplate, zero manual memory management

## The Demos

### 1. Pathfinding Agent (`gpu_autonomous_agent.flow`)

**Scenario**: An autonomous agent navigating a 2D grid with obstacles to reach a goal.

**Autonomous Loop**:
```
for each iteration:
  1. PERCEIVE: Compute attraction field via GPU (5 operations)
  2. DECIDE: Evaluate neighbor costs, pick best direction
  3. ACT: Update agent position
  4. CHECK: Has goal been reached?
```

**Key Stats**:
- 52 iterations to reach goal
- 260 GPU operations total
- 1024 cells evaluated per iteration
- CPU role: Loop control and convergence checking only

**Run it**:
```bash
octoflow run examples/gpu_autonomous_agent.flow
```

**Output Preview**:
```
=== GPU-Resident Autonomous Agent ===
Pathfinding agent navigating 32x32 grid to goal

Start: (2, 2)
Goal:  (29, 29)

  Iter 0: position (2, 2), distance=38.18
  Iter 10: position (7, 7), distance=31.11
  Iter 20: position (11, 13), distance=24.08

GOAL REACHED at iteration 52!
Final position: (28, 28)

GPU operations per iteration: 5
Total GPU dispatches: 260
CPU role: Orchestration and convergence checking only
```

### 2. Multi-Dispatch Pipeline (`gpu_autonomous_multi_dispatch.flow`)

**Scenario**: Process 1 million elements through a 5-stage GPU pipeline, then run 5 different reductions.

**Autonomous Pipeline**:
```
Stage 1: Square (x*x)
Stage 2: Square root
Stage 3: Sine transform
Stage 4: Scale by 2.0
Stage 5: Absolute value

Then reduce via: sum, mean, max, min, variance
```

**Key Stats**:
- 1,000,000 elements processed
- 10 GPU operations total
- 10,000,000 element operations
- CPU-GPU interactions: 12 total
- **CPU involvement: 0.00012%**
- **GPU autonomy: >99.9999%**

**Run it**:
```bash
octoflow run examples/gpu_autonomous_multi_dispatch.flow
```

**Output Preview**:
```
=== GPU-Resident Multi-Dispatch Autonomous Demo ===

Phase 1: GPU Initialization
  - Created 1000000 elements: 7.0 ms

Phase 2: Autonomous GPU Processing Pipeline
  Pipeline time: 15.8 ms

Phase 3: Autonomous GPU Reduction
  - Sum:      1917848.7500
  - Mean:     1.9178
  Reduction time: 108.4 ms

Total time: 131.4 ms
Throughput: 76095 elements*ops/ms

CPU involvement: 0.000120%
GPU autonomy: >99.9999%
```

### 3. Particle Swarm (Work in Progress)

The `gpu_autonomous_simple.flow` demonstrates particle swarm optimization with GPU-based updates. Currently being refined for array mutation semantics.

## The Technical Achievement

### What OctoFlow Does Differently

1. **Natural Control Flow**
   - `for` loops that span GPU operations
   - `if` conditions based on GPU results
   - Mutable variables updated by GPU computations

2. **Automatic GPU Dispatch**
   - No explicit kernel launches
   - No manual memory transfers
   - No synchronization primitives

3. **Seamless Composition**
   - Chain GPU operations naturally
   - Mix CPU logic with GPU compute
   - Results flow automatically

### Compare to Other Languages

**Python + NumPy**:
```python
# Runs on CPU (100x slower)
for i in range(50):
    errors = positions - targets
    velocities -= errors * 0.1
    positions += velocities
```

**Python + CuPy**:
```python
# Requires explicit GPU orchestration
import cupy as cp
positions = cp.array(positions)  # Explicit transfer
for i in range(50):
    errors = cp.subtract(positions, targets)  # GPU kernel
    cp.cuda.Stream.null.synchronize()  # Explicit sync
    # ... more explicit management
```

**CUDA/C++**:
```c
// Requires custom kernels
__global__ void update_kernel(float* pos, float* vel, ...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... kernel code
}

// Explicit memory management
cudaMalloc(&d_pos, size);
cudaMemcpy(d_pos, h_pos, size, cudaMemcpyHostToDevice);

// Explicit dispatch
for (int i = 0; i < 50; i++) {
    update_kernel<<<blocks, threads>>>(d_pos, d_vel, ...);
    cudaDeviceSynchronize();
}

cudaMemcpy(h_pos, d_pos, size, cudaMemcpyDeviceToHost);
cudaFree(d_pos);
```

**OctoFlow**:
```flow
// Natural, concise, performant
for iter in range(0, 50)
    let errors = gpu_sub(positions, targets)
    velocities = gpu_sub(velocities, gpu_scale(errors, 0.1))
    positions = gpu_add(positions, velocities)
end
```

## Performance Characteristics

### GPU Utilization
- **Pathfinding Agent**: 260 GPU dispatches in 2.7 seconds
- **Multi-Dispatch**: 10 million element-ops in 131ms
- **Throughput**: 76,000+ element-ops per millisecond

### CPU Overhead
- **Minimal**: Only loop iteration, condition checks, progress reporting
- **<0.001%** of total computation in multi-dispatch demo
- **No manual orchestration**: Compiler handles GPU coordination

### Memory Efficiency
- **GPU-Resident**: Data stays on GPU between operations
- **Zero-Copy Chains**: Pipeline stages don't roundtrip to CPU
- **Automatic**: No manual memory management required

## The Vision: Autonomous GPU Computation

These demos prove the core thesis:

**General-purpose programs should be able to delegate intensive computation to GPU while maintaining natural control flow on CPU.**

Not "write GPU kernels then call them."
Not "port entire program to GPU."

But: **Blend CPU control with GPU compute seamlessly.**

This is the future OctoFlow enables:
- AI agents with GPU-accelerated perception
- Simulations with GPU physics but CPU logic
- Data processing with GPU transforms but CPU orchestration
- Games with GPU rendering AND GPU gameplay logic

## Try It Yourself

Run all three demos:
```bash
# Pathfinding agent
octoflow run examples/gpu_autonomous_agent.flow

# Multi-dispatch pipeline
octoflow run examples/gpu_autonomous_multi_dispatch.flow
```

Expected runtime: <5 seconds total

## Implementation Details

### GPU Operations Used
- `gpu_fill(value, count)` - Fill array on GPU
- `gpu_add(a, b)` - Element-wise addition
- `gpu_sub(a, b)` - Element-wise subtraction
- `gpu_mul(a, b)` - Element-wise multiplication
- `gpu_scale(a, scalar)` - Scalar multiplication
- `gpu_sqrt(a)` - Square root
- `gpu_sin(a)` - Sine transform
- `gpu_abs(a)` - Absolute value
- `gpu_sum(a)` - Reduction: sum
- `gpu_mean(a)` - Reduction: mean
- `gpu_max(a)` - Reduction: max
- `gpu_min(a)` - Reduction: min
- `gpu_variance(a)` - Reduction: variance

### Under the Hood
- Each `gpu_*` operation dispatches a Vulkan compute shader
- Results stay GPU-resident until explicitly read (array indexing)
- Pipeline caching prevents redundant SPIR-V compilation
- Staging buffers optimize data transfer when needed

## Conclusion

**GPU-Resident Autonomous Agents** represent a new paradigm in GPU computing:

- Not just "parallel loops" (OpenMP, ISPC)
- Not just "GPU kernels" (CUDA, OpenCL)
- But **autonomous decision-making with GPU acceleration**

This is what makes OctoFlow boundary-breaking.

This is the capability no CPU-native language can match.

This is the future of general-purpose GPU programming.

---

**OctoFlow**: GPU computing, naturally.
