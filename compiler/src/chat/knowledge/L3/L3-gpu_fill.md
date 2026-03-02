# gpu_fill (L3)

## Working Example
```flow
let ones = gpu_fill(1.0, 5)
let scaled = gpu_scale(ones, 7.0)
let total = gpu_sum(scaled)
print("filled array scaled sum: {total}")

let zeros = gpu_fill(0.0, 1000)
let z_sum = gpu_sum(zeros)
print("zero fill sum: {z_sum}")
```

## Expected Output
```
filled array scaled sum: 35.0
zero fill sum: 0.0
```

## Common Mistakes
- DON'T: `let s = gpu_sum(gpu_fill(1.0, 5))` → DO: `let a = gpu_fill(1.0, 5)` then `let s = gpu_sum(a)`
- DON'T: `gpu_fill(5, 10)` → DO: `gpu_fill(5.0, 10)` (use float literals)
- DON'T: `let arr = [1.0, 1.0, 1.0]` for GPU → DO: `let arr = gpu_fill(1.0, 3)`

## Edge Cases
- gpu_fill(0.0, n) is useful for accumulators in loops
- Second argument n is an integer count, not a float
- Result lives on GPU; use gpu_sum or gpu_mean to extract scalars
