# sma / ema (L3)

## Working Example
```flow
use stats/timeseries

let prices = gpu_random(20)
let prices = gpu_scale(prices, 100.0)

let sma_5 = sma(prices, 5)
let sma_avg = gpu_mean(sma_5)
print("SMA(5) average: {sma_avg}")

let ema_vals = ema(prices, 0.3)
let ema_avg = gpu_mean(ema_vals)
print("EMA(0.3) average: {ema_avg}")

let sma_10 = sma(prices, 10)
let sma10_avg = gpu_mean(sma_10)
print("SMA(10) average: {sma10_avg}")
```

## Expected Output
```
SMA(5) average: 50.12
EMA(0.3) average: 49.87
SMA(10) average: 50.34
```

## Common Mistakes
- DON'T: `sma(prices)` → DO: `sma(prices, 5)` (window size required)
- DON'T: `ema(prices, 5)` → DO: `ema(prices, 0.3)` (alpha is a float 0-1, not window)
- DON'T: forget `use stats/timeseries` → DO: add import at the top

## Edge Cases
- sma window must be <= array length
- ema alpha near 1.0 gives more weight to recent values
- Both return GPU arrays of the same length as input
