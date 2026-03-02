# pearson (L3)

## Working Example
```flow
use stats/correlation

let x = gpu_fill(1.0, 5)
let step = gpu_fill(1.0, 5)
let x = gpu_add(x, step)

let y = gpu_scale(x, 3.0)
let noise = gpu_fill(0.1, 5)
let y = gpu_add(y, noise)

let r = pearson(x, y)
print("pearson correlation: {r}")

let flat = gpu_fill(5.0, 5)
let r2 = pearson(x, flat)
print("correlation with constant: {r2}")
```

## Expected Output
```
pearson correlation: 0.9999
correlation with constant: 0.0
```

## Common Mistakes
- DON'T: `pearson(x)` → DO: `pearson(x, y)` (requires two arrays)
- DON'T: forget `use stats/correlation` → DO: always import
- DON'T: `correlation(x, y)` → DO: `pearson(x, y)` (exact function name)

## Edge Cases
- Returns a scalar float in range [-1.0, 1.0]
- Perfect positive linear relationship returns ~1.0
- Constant array yields 0.0 (no variance to correlate)
- Both arrays must have the same length
