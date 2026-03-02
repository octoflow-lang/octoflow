# sharpe_ratio (L3)

## Working Example
```flow
use stats/risk

let returns = gpu_random(252)
let returns = gpu_scale(returns, 0.02)
let offset = gpu_fill(-0.005, 252)
let returns = gpu_add(returns, offset)

let risk_free = 0.02
let sharpe = sharpe_ratio(returns, risk_free)
print("sharpe ratio: {sharpe}")

let ret_mean = gpu_mean(returns)
let ret_std = stddev(returns)
print("mean return: {ret_mean}")
print("return volatility: {ret_std}")
```

## Expected Output
```
sharpe ratio: 0.8321
mean return: 0.005
return volatility: 0.0058
```

## Common Mistakes
- DON'T: `sharpe_ratio(returns)` → DO: `sharpe_ratio(returns, 0.02)` (risk-free rate required)
- DON'T: `sharpe_ratio(returns, risk_free_array)` → DO: risk_free is a scalar float
- DON'T: forget `use stats/risk` → DO: always import the module

## Edge Cases
- 252 trading days is a standard annual period
- risk_free rate is annualized (e.g., 0.02 = 2%)
- Higher sharpe means better risk-adjusted return
- Negative sharpe means returns below risk-free rate
