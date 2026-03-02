# gpu_scale (L3)

## Working Example
```flow
let data = gpu_fill(4.0, 5)
let doubled = gpu_scale(data, 2.0)
let d_sum = gpu_sum(doubled)
print("doubled sum: {d_sum}")

let noise = gpu_random(1000)
let amplified = gpu_scale(noise, 100.0)
let amp_mean = gpu_mean(amplified)
print("amplified mean: {amp_mean}")

let negated = gpu_scale(data, -1.0)
let neg_sum = gpu_sum(negated)
print("negated sum: {neg_sum}")
```

## Expected Output
```
doubled sum: 40.0
amplified mean: 49.83
negated sum: -20.0
```

## Common Mistakes
- DON'T: `let r = gpu_sum(gpu_scale(a, 2.0))` → DO: `let b = gpu_scale(a, 2.0)` then `let r = gpu_sum(b)`
- DON'T: `gpu_scale(a, b)` where b is an array → DO: use gpu_scale with a scalar float only
- DON'T: `a * 2.0` on GPU arrays → DO: `gpu_scale(a, 2.0)`

## Edge Cases
- Scaling by 0.0 produces an all-zero array
- Scaling by -1.0 negates every element
- The scalar argument must be a float, not an array
