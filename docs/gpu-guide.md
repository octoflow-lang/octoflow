# GPU Computing in OctoFlow

## How It Works

OctoFlow compiles operations to SPIR-V compute shaders and dispatches
them on your GPU via Vulkan. This happens automatically — you don't
write shaders or manage GPU memory.

```
.flow source → Parser → SPIR-V Emitter → Vulkan Dispatch → GPU
```

## GPU Arrays

When you call a `gpu_` function, the result stays in GPU memory:

```
let a = gpu_fill(1.0, 10000000)    // 10M elements, lives on GPU
let b = gpu_add(a, a)              // computed on GPU, result stays on GPU
let c = gpu_scale(b, 0.5)          // still on GPU, zero data transfer
print(gpu_sum(c))                  // only NOW does data come to CPU (one number)
```

Three GPU operations. Zero CPU-GPU data transfer between them. Only the
final sum brings a single float back to CPU for printing.

## When Data Moves to CPU

Data leaves the GPU only when it must:

| Trigger | Why |
|---------|-----|
| `print()` | Need to display values |
| `write_file()` / `write_csv()` | Need to write to disk |
| `for item in arr` | Need to iterate elements |
| `arr[i]` | Need a single element |
| `gpu_sum()` / `gpu_min()` / `gpu_max()` | Reduction returns a scalar |

Everything else stays GPU-resident.

## Deferred Dispatch

Chained GPU operations are batched into a single Vulkan command buffer
submission. Instead of one fence-wait per operation, the entire chain
pays overhead once:

```
let a = gpu_fill(1.0, 10000000)
let b = gpu_add(a, a)           // deferred
let c = gpu_mul(b, b)           // deferred
let d = gpu_scale(c, 0.001)     // deferred
let e = gpu_sum(d)              // flush: all 3 dispatches in 1 submit
```

This is why per-operation timings in OctoFlow benchmarks show sub-millisecond
numbers for element-wise operations on warm runs.

## GPU Operations Reference

### Create Arrays

| Function | Description |
|----------|-------------|
| `gpu_fill(val, n)` | N elements, all set to val |
| `gpu_range(start, end, step)` | Arithmetic sequence |

### Element-wise (Binary)

| Function | Description |
|----------|-------------|
| `gpu_add(a, b)` | a + b |
| `gpu_sub(a, b)` | a - b |
| `gpu_mul(a, b)` | a * b |
| `gpu_div(a, b)` | a / b |

### Element-wise (Unary / Scalar)

| Function | Description |
|----------|-------------|
| `gpu_scale(a, s)` | a * s |
| `gpu_abs(a)` | |a| |
| `gpu_negate(a)` | -a |
| `gpu_sqrt(a)` | sqrt(a) |
| `gpu_exp(a)` | e^a |
| `gpu_log(a)` | ln(a) |
| `gpu_sin(a)` | sin(a) |
| `gpu_cos(a)` | cos(a) |
| `gpu_pow(a, n)` | a^n |
| `gpu_clamp(a, lo, hi)` | clamp each element |
| `gpu_reverse(a)` | reverse order |

### Reductions

| Function | Returns |
|----------|---------|
| `gpu_sum(a)` | Sum of all elements |
| `gpu_min(a)` | Minimum element |
| `gpu_max(a)` | Maximum element |
| `gpu_mean(a)` | Arithmetic mean |
| `gpu_product(a)` | Product of all elements |
| `gpu_variance(a)` | Variance |
| `gpu_stddev(a)` | Standard deviation |

### Conditional and Reshaping

| Function | Description |
|----------|-------------|
| `gpu_where(cond, a, b)` | Select a where cond!=0, else b |
| `gpu_concat(a, b)` | Concatenate two arrays |
| `gpu_gather(data, indices)` | Indexed lookup (out-of-bounds returns 0.0) |
| `gpu_scatter(values, indices, size)` | Scatter values to indexed positions |
| `gpu_ema(a, alpha)` | Exponential moving average |

### Matrix

| Function | Description |
|----------|-------------|
| `gpu_matmul(a, b, m, k, n)` | Matrix multiply (m x k) * (k x n) |

### Creation and I/O

| Function | Description |
|----------|-------------|
| `gpu_random(n)` | Array of n random floats [0.0, 1.0) |
| `gpu_load_csv(path)` | Load CSV to GPU array |
| `gpu_load_binary(path)` | Load binary f32 to GPU |
| `gpu_save_csv(arr, path)` | Save GPU array to CSV |
| `gpu_save_binary(arr, path)` | Save GPU array as raw f32 |

### Custom Compute

| Function | Description |
|----------|-------------|
| `gpu_compute(spv_path, arr)` | Dispatch a custom SPIR-V kernel |
| `gpu_run(spv_path, arrs..., scalars...)` | Universal dispatch with multiple inputs and push constants |

## GPU Virtual Machine

For workloads that need multiple dispatches without CPU round-trips, OctoFlow provides a GPU-native virtual machine. The CPU submits once; the GPU runs an entire dispatch chain autonomously.

### Architecture

The GPU VM has 5 memory regions (Vulkan SSBOs):

| Buffer | Purpose |
|--------|---------|
| **Registers** | Per-instance I/O (input/output data per VM instance) |
| **Globals** | Shared data visible to all instances |
| **Heap** | Large read-only data (weights, lookup tables) |
| **Metrics** | HOST_VISIBLE GPU→CPU status (polled at ~1us, zero-copy) |
| **Control** | HOST_VISIBLE CPU→GPU commands (live-writable without rebuild) |

### Basic Usage

```
// Boot VM, load a kernel, execute
let vm = vm_boot()
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
let _w = vm_write_register(vm, 0, 0, data)

// Build a dispatch chain — multiple kernels, one submit
let pc = [0.0, 3.0, 8.0]
let _d = vm_dispatch(vm, "stdlib/gpu/kernels/vm_scale.spv", pc, 1.0)
let prog = vm_build(vm)
let _e = vm_execute(prog)

let result = vm_read_register(vm, 0, 0)
// result = [3, 6, 9, 12, 15, 18, 21, 24]
vm_shutdown(vm)
```

### Key Capabilities

- **Single submit**: Chain N dispatches into one `vkQueueSubmit` — no CPU stalls between stages
- **CPU polling**: Read GPU status in ~1us via HOST_VISIBLE memory (zero-copy, no fence)
- **Indirect dispatch**: GPU self-programs workgroup counts — no CPU involvement
- **Live control**: CPU writes to Control buffer without rebuilding the command buffer
- **Dormant VMs**: Over-provisioned command buffers activate without rebuild
- **I/O streaming**: CPU streams data to Globals; GPU processes with reusable command buffers
- **Homeostasis**: GPU self-regulates via maxnorm + regulator kernels

### Use Cases

- **LLM inference**: Layer-by-layer transformer execution in a single dispatch chain
- **Database queries**: Decompress → WHERE → aggregate chains on GPU-resident columnar data
- **Multi-agent**: Independent VM instances communicate via register-based message passing
- **Real-time pipelines**: CPU feeds batches, GPU processes without restart

## Checking GPU Status

In the REPL:

```
> :gpu
GPU: NVIDIA GeForce GTX 1660 SUPER
  arrays: 3 total (2 gpu-resident, 1 cpu)

> :arrays
  a    10,000,000  38.1 MB  gpu
  b    10,000,000  38.1 MB  gpu
  c    10,000,000  38.1 MB  cpu
```

## Performance

Benchmarks on NVIDIA GeForce GTX 1660 SUPER (10M elements):

| Operation | Time |
|-----------|------|
| gpu_add | 0.46 ms |
| gpu_mul | 3.27 ms |
| gpu_scale | 0.48 ms |
| gpu_sum (reduction) | 0.42 ms |

Per-operation overhead drops as chains grow due to deferred dispatch batching.

## Supported GPUs

Any GPU with Vulkan 1.0+ support:

- **NVIDIA** — GeForce, Quadro, Tesla (all modern cards)
- **AMD** — Radeon, Instinct
- **Intel** — Arc, integrated (Gen 9+)
- **Apple** — via MoltenVK

No CUDA required. No vendor lock-in. No SDK to install.
