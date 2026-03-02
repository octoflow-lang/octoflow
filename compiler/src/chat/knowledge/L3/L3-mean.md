# mean / statistics builtins (L3)

## Working Example
```flow
let data = gpu_random(1000)

let avg = mean(data)
let med = median(data)
let sd = stddev(data)
let v = variance(data)
let lo = min_val(data)
let hi = max_val(data)

print("mean: {avg}")
print("median: {med}")
print("stddev: {sd}")
print("variance: {v}")
print("min: {lo}")
print("max: {hi}")
```

## Expected Output
```
mean: 0.4987
median: 0.5012
stddev: 0.2889
variance: 0.0835
min: 0.0004
max: 0.9993
```

## Common Mistakes
- DON'T: `data.mean()` → DO: `mean(data)` (free functions, not methods)
- DON'T: `std(data)` → DO: `stddev(data)` (full name required)
- DON'T: `max(data)` → DO: `max_val(data)` (max_val not max)

## Edge Cases
- These are built-in; no use/import needed
- All return scalar floats
- variance equals stddev squared
- For GPU-specific reduction use gpu_mean, gpu_min, gpu_max
