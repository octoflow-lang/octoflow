# patterns (L2)
loom/ops/patterns — One-import convenience layer with auto-init GPU.

## Functions
gpu_mean_auto(arr: array) → float
  Mean with auto GPU/CPU dispatch
gpu_sum_auto(arr: array) → float
  Sum with auto dispatch
gpu_std_auto(arr: array) → float
  Standard deviation with auto dispatch
gpu_matmul_auto(a: array, b: array, m: int, n: int, k: int) → array
  Matrix multiply with auto dispatch
gpu_sort_auto(arr: array) → array
  Sort with auto dispatch
gpu_filter_auto(arr: array, pred: any) → array
  Filter with auto dispatch
gpu_normalize_auto(arr: array) → array
  Normalize to [0,1] with auto dispatch
gpu_bollinger(prices: array, period: int) → map
  Bollinger Bands (upper, mid, lower)
gpu_ema_auto(arr: array, span: int) → array
  Exponential moving average
