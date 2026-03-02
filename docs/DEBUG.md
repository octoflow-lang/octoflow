# OctoFlow Debug Guide

Debug toolkit for GPU dispatch chains and general .flow development.

## Quick Start

```flow
use "debug"

let mut arr = [3.0, 1.0, 4.0, 1.0, 5.0]
debug_assert_range(arr, 0.0, 10.0, "bounds")    // ASSERT PASS
debug_assert_sorted(arr, "order")                 // ASSERT FAIL
debug_stats(arr, "my_data")                       // prints min/max/mean/sum
```

## Array-Based Functions (no GPU required)

### Assertions

| Function | Description |
|---|---|
| `debug_assert_range(arr, min, max, label)` | All elements in [min, max] |
| `debug_assert_sorted(arr, label)` | Non-decreasing order |
| `debug_assert_sum(arr, expected, tol, label)` | Sum within tolerance |
| `debug_assert_nonzero(arr, label)` | No zero elements |
| `debug_assert_all_equal(arr, expected, label)` | All elements equal value |
| `debug_assert_monotonic(arr, label)` | Strictly increasing |
| `debug_assert_no_nan(arr, label)` | No NaN values* |
| `debug_assert_pairwise_equal(a, b, tol, label)` | Arrays match within tolerance |

All return `1.0` (pass) or `0.0` (fail). Print details on failure.

*Note: NaN does not survive array storage in OctoFlow (KNOWN_ISSUES #11). `debug_assert_no_nan` is primarily useful for GPU buffer downloads where NaN may appear from GPU computation.

### Inspection

| Function | Description |
|---|---|
| `debug_inspect(arr, label)` | Print first 16 elements |
| `debug_stats(arr, label)` | Print count, min, max, mean, sum |

### Profiling

| Function | Description |
|---|---|
| `debug_timer_start()` | Returns `now_ms()` timestamp |
| `debug_timer_end(t0, label)` | Prints elapsed ms, returns seconds |
| `debug_timer_end_silent(t0)` | Returns elapsed seconds (no print) |

**Important**: Use `now_ms()` for timing, not `time()`. The `time()` builtin has 1-second resolution.

## Chain-Based Functions (GPU runtime required)

For use with `stdlib/gpu/runtime.flow` dispatch chains. Require `--allow-ffi --allow-read`.

| Function | Description |
|---|---|
| `chain_assert_range(buf, count, min, max, label)` | Buffer values in range |
| `chain_assert_sorted(buf, count, label)` | Buffer sorted |
| `chain_assert_sum(buf, count, expected, tol, label)` | Buffer sum check |
| `chain_assert_nonzero(buf, count, label)` | No zeros in buffer |
| `chain_assert_all_equal(buf, count, expected, label)` | All buffer values equal |
| `chain_debug_inspect(buf, count, label)` | Print first 16 buffer elements |
| `chain_debug_stats(buf, count, label)` | Print buffer statistics |

Chain functions call `rt_download()` internally to read GPU buffer to `rt_result[]`.

## SPIR-V Debug Tools

| Function | Description |
|---|---|
| `spv_validate(path)` | Run `spirv-val` on .spv file |
| `spv_disassemble(path)` | Run `spirv-dis` on .spv file |

Require `--allow-exec` flag.

## Debugging Patterns

### Validate GPU output against CPU reference

```flow
use "debug"

let mut gpu_result = gpu_cumsum(input)

// Build CPU reference
let n = len(input)
let mut cpu_ref = []
let mut running = 0.0
let mut i = 0.0
while i < n
  running = running + input[int(i)]
  push(cpu_ref, running)
  i = i + 1.0
end

let _ok = debug_assert_pairwise_equal(gpu_result, cpu_ref, 0.01, "scan_verify")
```

### Profile a GPU operation

```flow
use "debug"

let mut data = gpu_fill(1.0, 1000000.0)
let t0 = debug_timer_start()
let result = gpu_add(data, data)
let _e = debug_timer_end(t0, "gpu_add_1M")
// Output: TIMER [gpu_add_1M]: 0.15 ms
```

### Inspect intermediate results

```flow
use "debug"

let mut a = gpu_fill(2.0, 8.0)
let mut b = gpu_mul(a, a)
let _n = debug_inspect(b, "after_square")
// Output:
//   Debug inspect [after_square]:
//     [0] = 4
//     [1] = 4
//     ...
let _m = debug_stats(b, "squared")
// Output:
//   Debug stats [squared]:
//     count = 8
//     min   = 4
//     max   = 4
//     mean  = 4
//     sum   = 32
```
