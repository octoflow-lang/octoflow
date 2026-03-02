# t_test_one_sample (L3)

## Working Example
```flow
use stats/hypothesis

let mut data = [5.1, 4.9, 5.3, 5.0, 4.8, 5.2, 5.1, 4.7, 5.4, 5.0]

let mu = 5.0
let result = t_test_one_sample(data, mu)

let t_stat = result.t
let p_val = result.p
let df = result.df
print("H0: mean = {mu}")
print("t-statistic: {t_stat}")
print("p-value: {p_val}")
print("degrees of freedom: {df}")

if p_val < 0.05
  print("reject H0 at alpha=0.05")
end

if p_val >= 0.05
  print("fail to reject H0 at alpha=0.05")
end

let mut shifted = [6.1, 5.9, 6.3, 6.0, 5.8, 6.2, 6.1, 5.7, 6.4, 6.0]
let result2 = t_test_one_sample(shifted, mu)
let p2 = result2.p
print("shifted data p-value: {p2}")

if p2 < 0.05
  print("shifted data: reject H0")
end
```

## Expected Output
```
H0: mean = 5.0
t-statistic: 0.5271
p-value: 0.6107
degrees of freedom: 9.0
fail to reject H0 at alpha=0.05
shifted data p-value: 0.0001
shifted data: reject H0
```

## Common Mistakes
- DON'T: `t_test_one_sample(data)` → DO: `t_test_one_sample(data, mu)` (mu required)
- DON'T: `if p_val < 0.05 { ... }` → DO: use `if ... end` blocks
- DON'T: `result["p"]` → DO: `result.p` (struct field access)

## Edge Cases
- Data list must have at least 2 elements
- Returns two-tailed p-value by default
- t_stat can be negative if sample mean is below mu
