# gpu_random (L3)

## Working Example
```flow
let data = gpu_random(10000)
let avg = gpu_mean(data)
let lo = gpu_min(data)
let hi = gpu_max(data)
print("mean of random array: {avg}")
print("min: {lo}")
print("max: {hi}")
```

## Expected Output
```
mean of random array: 0.4987
min: 0.0001
max: 0.9998
```

## Common Mistakes
- DON'T: `let avg = gpu_mean(gpu_random(10000))` → DO: separate into two let statements
- DON'T: `gpu_random(10000.0)` → DO: `gpu_random(10000)` (integer count)
- DON'T: `print(avg)` → DO: `print("mean: {avg}")`

## Edge Cases
- Values are in range [0, 1), never exactly 1.0
- Each call produces different values; results are non-deterministic
- For reproducible results, use gpu_fill with known values instead
