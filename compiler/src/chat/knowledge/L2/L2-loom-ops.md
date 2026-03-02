# ops (L2)
loom/ops/ops — GPU compute op wrappers.

## Functions
gpu_abs_run(rt: map, a: map) → map
  Element-wise absolute value
gpu_sqrt_run(rt: map, a: map) → map
  Element-wise square root
gpu_exp_run(rt: map, a: map) → map
  Element-wise exp
gpu_log_run(rt: map, a: map) → map
  Element-wise natural log
gpu_add_run(rt: map, a: map, b: map) → map
  Element-wise addition
gpu_sub_run(rt: map, a: map, b: map) → map
  Element-wise subtraction
gpu_mul_run(rt: map, a: map, b: map) → map
  Element-wise multiplication
gpu_div_run(rt: map, a: map, b: map) → map
  Element-wise division
gpu_where_run(rt: map, cond: map, a: map, b: map) → map
  Conditional select per element
