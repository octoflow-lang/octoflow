# GPU Compute Recipes

Complete, working .flow programs using `stdlib/gpu/patterns`.
One import. No setup. No cleanup. Just results.

---

## Quick Start

Every recipe starts with:

```
use "stdlib/gpu/patterns"
```

That's it. GPU initializes on first use. Falls back to CPU for small arrays.

---

## Recipe 1: Analyze a Dataset

Get a complete statistical profile of any numeric array.

```flow
use "stdlib/gpu/patterns"

let prices = [2340.5, 2355.2, 2318.9, 2367.1, 2390.0, 2345.8, 2412.3, 2378.6, 2405.1, 2350.0]

let report = gpu_analyze(prices)

print("Count:    {report[\"count\"]}")
print("Mean:     {report[\"mean\"]:.2}")
print("Std Dev:  {report[\"std\"]:.2}")
print("Min:      {report[\"min\"]:.2}")
print("Max:      {report[\"max\"]:.2}")
print("Median:   {report[\"median\"]:.2}")
print("Q25:      {report[\"q25\"]:.2}")
print("Q75:      {report[\"q75\"]:.2}")
print("Range:    {report[\"range\"]:.2}")
print("Outliers: {report[\"outlier_count\"]} ({report[\"outlier_pct\"]:.1}%)")
```

**What it returns:** A map with keys: `mean`, `std`, `min`, `max`, `median`, `q25`, `q75`, `count`, `range`, `outlier_count`, `outlier_pct`.

---

## Recipe 2: Compare Two Series

Measure how similar two arrays are.

```flow
use "stdlib/gpu/patterns"

let gold  = [2340, 2355, 2318, 2367, 2390, 2345, 2412, 2378, 2405, 2350]
let silver = [29.5, 29.8, 29.2, 30.1, 30.4, 29.6, 30.8, 30.2, 30.5, 29.7]

let cmp = gpu_compare(gold, silver)

print("Correlation:      {cmp[\"correlation\"]:.4}")
print("Mean difference:  {cmp[\"mean_diff\"]:.2}")
print("Trend alignment:  {cmp[\"aligned_pct\"]:.1}%")
```

**What it returns:** A map with keys: `correlation` (-1 to 1), `mean_diff`, `aligned_pct` (% of bars where both moved same direction).

---

## Recipe 3: Moving Averages and Bollinger Bands

Smooth a time series and compute trading bands.

```flow
use "stdlib/gpu/patterns"

let closes = [100, 102, 101, 103, 105, 104, 106, 108, 107, 110,
              109, 111, 113, 112, 114, 116, 115, 117, 119, 118]

// Simple moving average (5-period)
let sma = gpu_rolling_mean_auto(closes, 5)
print("SMA(5): {sma}")

// Exponential moving average (alpha = 2/(5+1) ≈ 0.333)
let ema = gpu_ema_auto(closes, 0.333)
print("EMA: {ema}")

// Bollinger Bands (5-period, 2 std deviations)
// Returns flat array: [mid_0..mid_k, upper_0..upper_k, lower_0..lower_k]
// where k = len(closes) - window + 1
let bb = gpu_bollinger(closes, 5, 2.0)
let k = len(closes) - 5.0 + 1.0
print("Middle: first = {bb[0]}, last = {bb[int(k - 1.0)]}")
print("Upper:  first = {bb[int(k)]}")
print("Lower:  first = {bb[int(k * 2.0)]}")
```

**Flat array layout:** `gpu_bollinger` returns `[mid..., upper..., lower...]` with `k` elements each. Extract with index math: mid at `[0..k-1]`, upper at `[k..2k-1]`, lower at `[2k..3k-1]`.

---

## Recipe 4: Matrix Multiply

Multiply two matrices on the GPU. Flat arrays, row-major.

```flow
use "stdlib/gpu/patterns"

// A: 2x3 matrix
let A = [1, 2, 3,
         4, 5, 6]

// B: 3x2 matrix
let B = [7,  8,
         9,  10,
         11, 12]

// C = A * B → 2x2 result
let C = gpu_matmul_auto(A, B, 2, 2, 3)
print("Result: {C}")
// Expected: [58, 64, 139, 154]
```

**Parameters:** `gpu_matmul_auto(A, B, m, n, k)` where A is m×k, B is k×n, result is m×n.

---

## Recipe 5: Filter and Top-K

Find the most interesting values.

```flow
use "stdlib/gpu/patterns"

let scores = [85, 92, 45, 78, 95, 67, 88, 91, 73, 99, 55, 82]

// Keep only scores above 80
let high_scores = gpu_filter_auto(scores, 80.0, "gt")
print("Above 80: {high_scores}")

// Top 3 scores
let top3 = gpu_top_k_auto(scores, 3)
print("Top 3: {top3}")

// Remove duplicates
let unique_scores = gpu_unique_auto(scores)
print("Unique: {unique_scores}")
```

**Filter ops:** `"gt"` (>), `"lt"` (<), `"eq"` (==), `"gte"` (>=), `"lte"` (<=).

---

## Recipe 6: Normalize Data

Scale arrays into standard ranges.

```flow
use "stdlib/gpu/patterns"

let raw = [10, 50, 30, 90, 70, 20, 80, 40, 60, 100]

// Min-max scale to [0, 1]
let normed = gpu_normalize_auto(raw)
print("Normalized: {normed}")

// Z-score: (x - mean) / std
let zscores = gpu_zscore_auto(raw)
print("Z-scores: {zscores}")

// Clamp to [25, 75]
let clamped = gpu_clamp_auto(raw, 25.0, 75.0)
print("Clamped: {clamped}")
```

---

## Recipe 7: Element-wise GPU Math

Apply math operations to every element via GPU kernels.

```flow
use "stdlib/gpu/patterns"

let data = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

// Square root of every element
let roots = gpu_map_auto(data, "sqrt")
print("Roots: {roots}")

// Other operations: "abs", "exp", "log", "sin", "cos",
//                   "floor", "ceil", "round", "negate"

// Add two arrays element-wise
let a = [1, 2, 3, 4, 5]
let b = [10, 20, 30, 40, 50]
let sum = gpu_add_auto(a, b)
print("Sum: {sum}")

// Multiply element-wise
let product = gpu_mul_auto(a, b)
print("Product: {product}")

// Conditional select: where condition > 0, pick from a, else from b
let cond = [1, 0, 1, 0, 1]
let picked = gpu_where_auto(cond, a, b)
print("Selected: {picked}")
// Expected: [1, 20, 3, 40, 5]
```

---

## Recipe 8: Financial Analysis

Complete financial toolkit on price data.

```flow
use "stdlib/gpu/patterns"

let closes = [100, 102, 101, 103, 105, 104, 106, 108, 107, 110,
              109, 111, 113, 112, 114, 116, 115, 117, 119, 118]

// Percentage change between bars
let pct = gpu_pct_change_auto(closes)
print("% Change: {pct}")

// Log returns
let lr = gpu_log_returns_auto(closes)
print("Log returns: {lr}")

// Rate of change (5-period)
let roc = gpu_roc_auto(closes, 5)
print("ROC(5): {roc}")

// Momentum (5-period)
let mom = gpu_momentum_auto(closes, 5)
print("Momentum(5): {mom}")
```

For ATR (requires high/low/close arrays):

```flow
use "stdlib/gpu/patterns"

let highs  = [102, 104, 103, 105, 107, 106, 108, 110, 109, 112]
let lows   = [98,  100, 99,  101, 103, 102, 104, 106, 105, 108]
let closes = [100, 102, 101, 103, 105, 104, 106, 108, 107, 110]

let atr = gpu_atr_auto(highs, lows, closes, 5)
print("ATR(5): {atr}")
```

---

## Recipe 9: Signal Processing

Smooth, filter, and analyze signals.

```flow
use "stdlib/gpu/patterns"

// Noisy signal
let signal = [0, 1, 0.5, 2, 1.5, 3, 2.5, 4, 3.5, 5,
              4.5, 3, 3.5, 2, 2.5, 1, 1.5, 0, 0.5, -1]

// Smooth with rolling mean (window = 3)
let smooth = gpu_rolling_mean_auto(signal, 3)
print("Smoothed: {smooth}")

// Exponential smoothing (alpha = 0.3)
let ema = gpu_ema_auto(signal, 0.3)
print("EMA: {ema}")

// Convolution with a custom kernel
let kernel = [0.25, 0.5, 0.25]  // simple blur
let convolved = gpu_convolve_auto(signal, kernel)
print("Convolved: {convolved}")

// Autocorrelation (check for periodicity)
let ac = gpu_autocorrelation_auto(signal, 10)
print("Autocorrelation: {ac}")
```

---

## Recipe 10: Dot Product and Similarity

Quick vector math.

```flow
use "stdlib/gpu/patterns"

let v1 = [1, 0, 1, 0, 1]
let v2 = [0, 1, 1, 0, 1]

// Dot product
let dp = gpu_dot_auto(v1, v2)
print("Dot product: {dp}")
// Expected: 2.0 (positions 2 and 4 match)

// Matrix-vector multiply
let matrix = [1, 2, 3,
              4, 5, 6]  // 2x3
let vec = [1, 1, 1]     // 3 elements
let result = gpu_matvec_auto(matrix, vec, 2, 3)
print("M*v: {result}")
// Expected: [6, 15]
```

---

## Function Reference

### Analysis
| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `gpu_analyze(arr)` | array | map | Complete statistical profile |
| `gpu_compare(a, b)` | two arrays | map | Correlation + trend alignment |

### Statistics
| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `gpu_mean_auto(arr)` | array | number | Arithmetic mean |
| `gpu_sum_auto(arr)` | array | number | Sum of elements |
| `gpu_std_auto(arr)` | array | number | Standard deviation |
| `gpu_min_auto(arr)` | array | number | Minimum value |
| `gpu_max_auto(arr)` | array | number | Maximum value |
| `gpu_median_auto(arr)` | array | number | Median value |

### Math
| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `gpu_matmul_auto(A, B, m, n, k)` | flat arrays + dims | array | Matrix multiply (m×k * k×n) |
| `gpu_matvec_auto(M, v, rows, cols)` | matrix + vector + dims | array | Matrix-vector multiply |
| `gpu_dot_auto(a, b)` | two arrays | number | Dot product |
| `gpu_transpose_auto(M, rows, cols)` | flat array + dims | array | Matrix transpose |

### Arrays
| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `gpu_sort_auto(arr)` | array | array | Sort ascending |
| `gpu_filter_auto(arr, threshold, op)` | array + number + op string | array | Keep matching elements |
| `gpu_unique_auto(arr)` | array | array | Remove duplicates |
| `gpu_top_k_auto(arr, k)` | array + count | array | K largest values |
| `gpu_normalize_auto(arr)` | array | array | Scale to [0, 1] |
| `gpu_clamp_auto(arr, lo, hi)` | array + bounds | array | Clip to range |
| `gpu_zscore_auto(arr)` | array | array | Z-score normalize |

### Signal
| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `gpu_rolling_mean_auto(arr, window)` | array + window size | array | Simple moving average |
| `gpu_ema_auto(arr, alpha)` | array + smoothing factor | array | Exponential moving average |
| `gpu_convolve_auto(signal, kernel)` | two arrays | array | 1D convolution |
| `gpu_autocorrelation_auto(arr, max_lag)` | array + max lag | array | Autocorrelation |

### Finance
| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `gpu_bollinger(arr, window, num_std)` | array + params | flat array | Bollinger Bands `[mid..., upper..., lower...]` |
| `gpu_atr_auto(H, L, C, period)` | 3 arrays + period | array | Average True Range |
| `gpu_roc_auto(arr, period)` | array + period | array | Rate of change (%) |
| `gpu_momentum_auto(arr, period)` | array + period | array | Momentum (absolute) |
| `gpu_log_returns_auto(arr)` | array | array | Log returns |
| `gpu_pct_change_auto(arr)` | array | array | Percentage change |

### Element-wise
| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `gpu_map_auto(arr, op)` | array + op string | array | Apply math (sqrt, exp, log, sin, ...) |
| `gpu_add_auto(a, b)` | two arrays | array | Add element-wise |
| `gpu_sub_auto(a, b)` | two arrays | array | Subtract element-wise |
| `gpu_mul_auto(a, b)` | two arrays | array | Multiply element-wise |
| `gpu_div_auto(a, b)` | two arrays | array | Divide element-wise |
| `gpu_where_auto(cond, a, b)` | three arrays | array | Conditional select |

---

## Under the Hood

`stdlib/gpu/patterns` imports 8 modules:

| Module | What it provides |
|--------|-----------------|
| `runtime` | Vulkan lifecycle (`rt_init`, buffers, dispatch chains) |
| `ops` | Pre-compiled SPIR-V kernels for element-wise math |
| `stats` | GPU reductions (sum, mean, std, percentile, etc.) |
| `linalg` | Matrix/vector operations |
| `array_ops` | Filter, sort, unique, compact, top-k |
| `signal` | Convolution, autocorrelation, EMA, FFT |
| `aggregate` | Rolling window operations |
| `composite` | High-level combinations (Bollinger, ATR, describe) |

Auto-initialization: The first function that needs the Vulkan runtime calls `rt_init()` once. Subsequent calls skip it.

CPU fallback: Arrays with fewer than 64 elements use CPU loops. GPU dispatch overhead is not worth it for tiny arrays.

---

## When to Use Advanced Mode

`stdlib/gpu/patterns` covers 90% of use cases. For the remaining 10%, import the individual modules directly:

```flow
use "stdlib/gpu/runtime"
use "stdlib/gpu/stats"

// Manual lifecycle for fine-grained control
rt_init()

// Use lower-level functions directly
let result = gpu_histogram_counts(data, 50)  // not in patterns.flow
let pct = gpu_percentile(data, 95.0)          // direct call

rt_cleanup()
```

Functions available in advanced mode but not in patterns:
- `gpu_histogram_counts(arr, nbins)` — frequency histogram
- `gpu_percentile(arr, p)` — arbitrary percentile
- `gpu_covariance_matrix(data, n_features, n_samples)` — covariance matrix
- `gpu_argsort(arr)` — sort indices
- `gpu_rank(arr)` — ordinal ranks
- `gpu_searchsorted(sorted, values)` — binary search
- `gpu_interpolate_linear(x_old, y_old, x_new)` — linear interpolation
- `gpu_resample(arr, new_length)` — resample to new length
- `gpu_fft_radix2(arr)` — FFT (power-of-2 length)
- `gpu_bandpass_filter(signal, low_freq, high_freq)` — frequency filter
- `gpu_cumsum(arr)` — cumulative sum
- `gpu_cumprod(arr)` — cumulative product
- `gpu_diff(arr)` — first differences
- `gpu_outer_product(v1, v2)` — outer product matrix

---

## Test Suite

60 tests across 4 files validate the GPU subsystem:

| File | Tests | What it covers |
|------|-------|---------------|
| `stdlib/gpu/tests/test_gpu_recipes.flow` | 33 | All 35 patterns.flow functions with known inputs/outputs |
| `stdlib/gpu/tests/test_gpu_edge_cases.flow` | 14 | Empty arrays, single elements, negative values, zero-variance, large values |
| `stdlib/gpu/tests/bench_gpu_performance.flow` | 7 | Element-wise, reduction, sort, matmul, sqrt, stats, analyze at 1K-100K scale |
| `stdlib/gpu/tests/test_gpu_stress.flow` | 6 | 4-VM simultaneous, 4096 floats, 10-op deep chain, lifecycle reuse, async+poll, cross-VM isolation |

Run all:

```bash
octoflow run stdlib/gpu/tests/test_gpu_recipes.flow --allow-read --allow-ffi
octoflow run stdlib/gpu/tests/test_gpu_edge_cases.flow --allow-read --allow-ffi
octoflow run stdlib/gpu/tests/bench_gpu_performance.flow --allow-read --allow-ffi
octoflow run stdlib/gpu/tests/test_gpu_stress.flow --allow-read --allow-ffi
```
