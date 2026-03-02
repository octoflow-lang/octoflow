# impute_missing (L3)

## Working Example
```flow
use ml/preprocess

let data = gpu_fill(5.0, 10)
let dirty = gpu_add(data, gpu_random(10))
let clean = impute_missing(dirty, "mean")
let clean_avg = gpu_mean(clean)
print("imputed mean: {clean_avg}")

let med_clean = impute_missing(dirty, "median")
let med_avg = gpu_mean(med_clean)
print("median imputed mean: {med_avg}")
```

## Expected Output
```
imputed mean: 5.4821
median imputed mean: 5.4736
```

## Common Mistakes
- DON'T: `impute_missing(data)` → DO: `impute_missing(data, "mean")` (strategy is required)
- DON'T: `impute_missing(data, mean)` → DO: `impute_missing(data, "mean")` (strategy is a string)
- DON'T: forget `use ml/preprocess` → DO: add it at the top

## Edge Cases
- Supported strategies: "mean", "median", "zero"
- Missing values (NaN) are replaced with the computed statistic
- If no values are missing, the array is returned unchanged
