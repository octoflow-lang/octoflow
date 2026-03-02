# normalize (L3)

## Working Example
```flow
use data/transform

let raw = gpu_fill(10.0, 5)
let offsets = gpu_random(5)
let offsets = gpu_scale(offsets, 90.0)
let raw = gpu_add(raw, offsets)

let normed = normalize(raw)
let n_min = gpu_min(normed)
let n_max = gpu_max(normed)
let n_mean = gpu_mean(normed)
print("normalized min: {n_min}")
print("normalized max: {n_max}")
print("normalized mean: {n_mean}")
```

## Expected Output
```
normalized min: 0.0
normalized max: 1.0
normalized mean: 0.4521
```

## Common Mistakes
- DON'T: `normalize(arr, 0, 1)` → DO: `normalize(arr)` (always scales to [0,1])
- DON'T: `data/transform.normalize(arr)` → DO: `use data/transform` then `normalize(arr)`
- DON'T: `norm(arr)` → DO: `normalize(arr)` (full function name)

## Edge Cases
- Output minimum is always 0.0, maximum is always 1.0
- If all values are identical, result may be all 0.0 (zero variance)
- Useful as preprocessing before ml functions like kmeans
- Returns a GPU array of the same length
