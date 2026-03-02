# gpu_add (L3)

## Working Example
```flow
let a = gpu_fill(3.0, 4)
let b = gpu_fill(7.0, 4)
let c = gpu_add(a, b)
let total = gpu_sum(c)
let avg = gpu_mean(c)
print("element-wise sum total: {total}")
print("each element should be 10: {avg}")
```

## Expected Output
```
element-wise sum total: 40.0
each element should be 10: 10.0
```

## Common Mistakes
- DON'T: `let total = gpu_sum(gpu_add(a, b))` → DO: `let c = gpu_add(a, b)` then `let total = gpu_sum(c)`
- DON'T: `let c = a + b` for GPU arrays → DO: `let c = gpu_add(a, b)`
- DON'T: `gpu_add(a, 5.0)` → DO: `gpu_scale(a, 1.0)` and `gpu_fill(5.0, n)` then gpu_add

## Edge Cases
- Both arrays must have the same length
- To add a scalar to every element, create a filled array first with gpu_fill
- Result is a new GPU array; originals are unchanged
