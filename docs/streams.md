# Streams & Pipelines

Streams are GPU-dispatched data pipelines. Data flows from a source
through a chain of operations, each executing on the GPU.

## Creating a Stream

```
stream prices = tap("input.csv")
```

`tap()` reads data from a file into a GPU-resident array. The path
must be a literal string.

## Pipe Chains

Chain operations with `|>`:

```
stream result = tap("data.csv") |> abs |> scale(100) |> clamp(0, 255)
emit(result, "output.csv")
```

Each `|>` stage transforms every element in the stream.

## Pipeline Functions

Define reusable transforms:

```
fn normalize: subtract(50) |> divide(100)
fn warm_filter: brightness(20) |> contrast(1.2) |> saturate(1.1)

stream out = tap("photo.jpg") |> warm_filter
emit(out, "output.png")
```

## Output

```
emit(stream, "output.csv")     // write to file
emit(stream, "output.png")     // write image
```

## All Pipe Operations

### Element-wise (Zero Parameters)

| Operation | Description |
|-----------|-------------|
| `abs` | Absolute value |
| `sqrt` | Square root |
| `exp` | Exponential (e^x) |
| `log` | Natural logarithm |
| `negate` | Negate all elements |
| `floor` | Floor to integer |
| `ceil` | Ceiling to integer |
| `round` | Round to nearest |
| `sin` | Sine |
| `cos` | Cosine |

### Element-wise (One Parameter)

| Operation | Description |
|-----------|-------------|
| `add(n)` | Add scalar to each element |
| `subtract(n)` | Subtract scalar |
| `multiply(n)` | Multiply by scalar |
| `divide(n)` | Divide by scalar |
| `pow(n)` | Raise to power |
| `mod(n)` | Modulo |
| `min(n)` | Element-wise minimum with scalar |
| `max(n)` | Element-wise maximum with scalar |
| `scale(n)` | Alias for multiply(n) |

### Element-wise (Two Parameters)

| Operation | Description |
|-----------|-------------|
| `clamp(lo, hi)` | Clamp each element to [lo, hi] |

### Temporal (Sequential)

| Operation | Description |
|-----------|-------------|
| `ema(alpha)` | Exponential moving average |
| `decay(factor)` | Exponential decay |

### Reductions

| Operation | Description |
|-----------|-------------|
| `sum` | Sum all elements |
| `min` | Find minimum |
| `max` | Find maximum |

### Scan (Prefix)

| Operation | Description |
|-----------|-------------|
| `prefix_sum` | Cumulative sum |

### Automatic Fusion

Consecutive map operations are fused into a single GPU kernel:

```
// These 3 operations execute as 1 GPU dispatch:
stream out = tap("data.csv") |> subtract(50) |> divide(100)
```

The compiler detects common patterns like normalize (`subtract |> divide`)
and scale-shift (`multiply |> add`) and fuses them automatically.

## Examples

### Normalize to 0-1

```
stream raw = tap("sensor.csv")
stream normalized = raw |> subtract(min_val) |> divide(max_val - min_val)
emit(normalized, "normalized.csv")
```

### Image Processing

```
use filters

stream photo = tap("input.jpg")
stream warm = photo |> filters.brightness(20.0) |> filters.contrast(1.2)
emit(warm, "output.png")
```

### Financial Data

```
stream prices = tap("prices.csv")
stream smoothed = prices |> ema(0.1)
stream returns = smoothed |> log |> subtract(0) |> multiply(100)
emit(returns, "returns.csv")
```
