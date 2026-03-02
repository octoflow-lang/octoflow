# gpu_mean (L3)

## Working Example
```flow
let data = gpu_random(20000)
let avg = gpu_mean(data)
let lo = gpu_min(data)
let hi = gpu_max(data)
print("mean: {avg}")
print("min: {lo}")
print("max: {hi}")

let uniform = gpu_fill(5.0, 100)
let u_avg = gpu_mean(uniform)
print("uniform mean: {u_avg}")
```

## Expected Output
```
mean: 0.5012
min: 0.00003
max: 0.99991
uniform mean: 5.0
```

## Common Mistakes
- DON'T: `let m = gpu_mean(gpu_random(100))` → DO: `let r = gpu_random(100)` then `let m = gpu_mean(r)`
- DON'T: `mean(data)` on GPU arrays → DO: `gpu_mean(data)` (use the gpu_ prefix for GPU data)
- DON'T: `gpu_mean(a, b)` → DO: gpu_mean takes a single array argument

## Edge Cases
- Returns a scalar float, not a GPU array
- For a filled array, mean equals the fill value
- Random arrays of large n converge to mean ~0.5
