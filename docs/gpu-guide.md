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
| `gpu_random(n, lo, hi)` | N random values in [lo, hi) |

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
| `gpu_dot(a, b)` | Dot product |
| `gpu_count(a)` | Element count |
| `gpu_cumsum(a)` | Cumulative sum (prefix scan) |

### Transformations

| Function | Description |
|----------|-------------|
| `gpu_sort(a)` | Parallel radix sort |
| `gpu_concat(a, b)` | Concatenate two arrays |
| `gpu_gather(data, indices)` | Gather elements by index |
| `gpu_scatter(vals, indices, n)` | Scatter values by index into array of size n |
| `gpu_topk(arr, k)` | Top-K values (sorted descending) |
| `gpu_topk_indices(arr, k)` | Indices of top-K values |
| `gpu_ema(arr, alpha)` | Exponential moving average |

### Conditional

| Function | Description |
|----------|-------------|
| `gpu_where(cond, a, b)` | Select a where cond!=0, else b |

### Matrix

| Function | Description |
|----------|-------------|
| `gpu_matmul(a, b, m, n, k)` | Matrix multiply: A is m×k, B is k×n, result is m×n |

### GPU I/O

| Function | Description |
|----------|-------------|
| `gpu_load_csv(path)` | Load CSV file directly to GPU array |
| `gpu_load_binary(path)` | Load raw f32 binary to GPU array |
| `gpu_save_csv(arr, path)` | Save GPU array to CSV file |
| `gpu_save_binary(arr, path)` | Save GPU array to raw f32 binary |

### Timing

| Function | Description |
|----------|-------------|
| `gpu_timer_start()` | Start GPU timer |
| `gpu_timer_end()` | End GPU timer, returns microseconds |
| `gpu_info()` | GPU device information string |

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
