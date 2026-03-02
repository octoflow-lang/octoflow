# normal_pdf (L3)

## Working Example
```flow
use stats/distribution

let mu = 0.0
let sigma = 1.0

let peak = normal_pdf(0.0, mu, sigma)
print("pdf at mean: {peak}")

let at_one = normal_pdf(1.0, mu, sigma)
print("pdf at x=1: {at_one}")

let at_neg = normal_pdf(-1.0, mu, sigma)
print("pdf at x=-1: {at_neg}")

let mut xs = []
let mut densities = []
for i in range(0, 21)
  let x = -3.0 + i * 0.3
  let d = normal_pdf(x, mu, sigma)
  push(xs, x)
  push(densities, d)
end

let d_at_0 = get(densities, 10)
let d_at_tail = get(densities, 0)
print("density at center: {d_at_0}")
print("density at tail: {d_at_tail}")

let custom = normal_pdf(100.0, 100.0, 15.0)
print("IQ pdf at 100 (mu=100, s=15): {custom}")
```

## Expected Output
```
pdf at mean: 0.3989
pdf at x=1: 0.2420
pdf at x=-1: 0.2420
density at center: 0.3989
density at tail: 0.0044
IQ pdf at 100 (mu=100, s=15): 0.0266
```

## Common Mistakes
- DON'T: `normal_pdf(x)` → DO: `normal_pdf(x, mu, sigma)` (all three args required)
- DON'T: `normal_pdf(x, sigma, mu)` → DO: `normal_pdf(x, mu, sigma)` (order matters)
- DON'T: `sigma = 0.0` → DO: sigma must be positive

## Edge Cases
- Returns density, not probability (can exceed 1.0 for small sigma)
- Symmetric: normal_pdf(mu + d, mu, s) == normal_pdf(mu - d, mu, s)
- Very large |x - mu| / sigma returns values approaching 0.0
