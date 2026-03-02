# matrix (L2)
science/matrix — Matrix operations (norms, diagonal, solvers)

## Functions
mat_norm(m: array, rows: int, cols: int) → float — Frobenius norm
mat_diag(m: array, rows: int, cols: int) → array — Extract diagonal
mat_from_diag(diag: array) → array — Diagonal matrix from vector
mat_solve_triangular(l: array, b: array, n: int) → array — Solve Lx=b (lower)
mat_solve_upper(u: array, b: array, n: int) → array — Solve Ux=b (upper)
mat_row_norm(m: array, rows: int, cols: int) → array — L2 norm per row
mat_col_norm(m: array, rows: int, cols: int) → array — L2 norm per column
