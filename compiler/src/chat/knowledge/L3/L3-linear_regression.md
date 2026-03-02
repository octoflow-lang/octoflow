# linear_regression (L3)

## Working Example
```flow
use ml/regression

let x = gpu_fill(0.0, 5)
let x = gpu_add(x, gpu_fill(1.0, 5))
let x = gpu_add(x, gpu_fill(0.0, 5))

let y = gpu_scale(x, 2.0)
let bias = gpu_fill(3.0, 5)
let y = gpu_add(y, bias)

let model = linear_regression(x, y)
let slope = model.slope
let intercept = model.intercept
print("slope: {slope}")
print("intercept: {intercept}")
```

## Expected Output
```
slope: 2.0
intercept: 3.0
```

## Common Mistakes
- DON'T: `linear_regression(x, y, z)` → DO: `linear_regression(x, y)` (two arrays only)
- DON'T: `let s = linear_regression(x, y).slope` → DO: `let model = linear_regression(x, y)` then `let s = model.slope`
- DON'T: forget `use ml/regression` → DO: add it at the top of the file

## Edge Cases
- x and y must have the same length
- Returns a record with .slope and .intercept fields
- For prediction: compute slope * new_x + intercept manually
