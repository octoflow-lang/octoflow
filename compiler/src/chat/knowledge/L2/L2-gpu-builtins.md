# gpu builtins (L2)
(built-in) — GPU compute via Vulkan. No import needed.

## Functions
gpu_fill(val: float, n: int) → gpu_array
  Create array of n copies of val on GPU
gpu_random(n: int) → gpu_array
  Generate n random floats on GPU
gpu_range(start: float, end: float, step: float) → gpu_array
  Create range array on GPU
gpu_add(a: gpu_array, b: gpu_array) → gpu_array
  Element-wise addition
gpu_sub(a: gpu_array, b: gpu_array) → gpu_array
  Element-wise subtraction
gpu_mul(a: gpu_array, b: gpu_array) → gpu_array
  Element-wise multiplication
gpu_div(a: gpu_array, b: gpu_array) → gpu_array
  Element-wise division
gpu_scale(a: gpu_array, s: float) → gpu_array
  Scale array by scalar
gpu_sqrt(a: gpu_array) → gpu_array
  Element-wise square root
gpu_exp(a: gpu_array) → gpu_array
  Element-wise exponentiation
gpu_log(a: gpu_array) → gpu_array
  Element-wise natural log
gpu_sum(a: gpu_array) → float
  Sum all elements
gpu_min(a: gpu_array) → float
  Minimum element
gpu_max(a: gpu_array) → float
  Maximum element
gpu_mean(a: gpu_array) → float
  Mean of elements
gpu_matmul(a: gpu_array, b: gpu_array, m: int, n: int, k: int) → gpu_array
  Matrix multiply (m x k) * (k x n)
gpu_where(cond: gpu_array, a: gpu_array, b: gpu_array) → gpu_array
  Conditional select per element
gpu_sort(a: gpu_array) → gpu_array
  Sort array ascending
