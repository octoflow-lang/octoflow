# linalg (L2)
ml/linalg — Linear algebra on flat row-major matrices

## Functions
mat_create(rows: int, cols: int) → array — Create zero matrix
mat_identity(n: int) → array — Identity matrix
mat_get(m: array, cols: int, r: int, c: int) → float — Get element
mat_set(m: array, cols: int, r: int, c: int, val: float) → array — Set element
mat_mul(a: array, b: array, ar: int, ac: int, bc: int) → array — Matrix multiply
mat_transpose(m: array, rows: int, cols: int) → array — Transpose
mat_add(a: array, b: array) → array — Element-wise add
mat_scale(m: array, s: float) → array — Scalar multiply
mat_det_2x2(m: array) → float — 2x2 determinant
mat_inverse_2x2(m: array) → array — 2x2 inverse
solve_2x2(a: array, b: array) → array — Solve 2x2 system Ax=b
