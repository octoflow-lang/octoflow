# data_composite (L2)
loom/data/composite — High-level GPU data wrappers.

## Functions
gpu_describe(arr: array) → map
  Descriptive stats (mean, std, min, max, count)
gpu_outlier_detection(arr: array, threshold: float) → array
  Detect outliers by z-score threshold
gpu_standardize(arr: array) → array
  Standardize to zero mean, unit variance
gpu_bollinger_bands(prices: array, period: int) → map
  Bollinger Bands (upper, mid, lower)
gpu_atr(high: array, low: array, close: array, period: int) → array
  Average True Range indicator
gpu_ewma(arr: array, span: int) → array
  Exponentially weighted moving average
gpu_log_returns(prices: array) → array
  Logarithmic returns from price series
