# gpu_matmul (L3)

## Working Example
```flow
let a = gpu_fill(1.0, 6)
let b = gpu_fill(2.0, 6)
let c = gpu_matmul(a, b, 2, 3, 3)
let total = gpu_sum(c)
print("matmul result sum: {total}")

let identity = gpu_fill(0.0, 4)
let vals = gpu_fill(3.0, 4)
let result = gpu_matmul(vals, identity, 2, 2, 2)
let r_sum = gpu_sum(result)
print("matmul with zeros: {r_sum}")
```

## Expected Output
```
matmul result sum: 36.0
matmul with zeros: 0.0
```

## Common Mistakes
- DON'T: `gpu_matmul(a, b)` → DO: `gpu_matmul(a, b, m, n, k)` (all 5 params required)
- DON'T: `let c = gpu_sum(gpu_matmul(a, b, 2, 3, 3))` → DO: separate into two let statements
- DON'T: confuse dimension order → DO: A is m*k elements, B is k*n elements, result is m*n

## Edge Cases
- Matrices are flat arrays; A has m*k elements, B has k*n elements
- Result array has m*n elements
- gpu_matmul(a, b, 2, 3, 3) means A is 2x3, B is 3x3, result is 2x3
