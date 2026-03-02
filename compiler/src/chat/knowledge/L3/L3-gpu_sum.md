# gpu_sum (L3)

## Working Example
```flow
let vals = gpu_fill(2.5, 100)
let total = gpu_sum(vals)
print("sum of 100 elements of 2.5: {total}")

let r = gpu_random(5000)
let r_sum = gpu_sum(r)
print("sum of 5000 random values: {r_sum}")
```

## Expected Output
```
sum of 100 elements of 2.5: 250.0
sum of 5000 random values: 2503.47
```

## Common Mistakes
- DON'T: `let s = gpu_sum(gpu_fill(1.0, 10))` → DO: `let a = gpu_fill(1.0, 10)` then `let s = gpu_sum(a)`
- DON'T: `let s = gpu_sum(a, b)` → DO: `let c = gpu_add(a, b)` then `let s = gpu_sum(c)` (gpu_sum takes one array)
- DON'T: `print(total)` → DO: `print("total: {total}")`

## Edge Cases
- Returns a scalar float, not a GPU array
- For mean instead of sum, use gpu_mean
- On very large arrays, floating-point precision may vary slightly
