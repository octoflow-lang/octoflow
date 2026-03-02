# optimize (L2)
science/optimize — Optimization and root-finding

## Functions
gradient_descent(f: string, x0: float, lr: float, epochs: int) → float — Minimize f
golden_section(f: string, a: float, b: float, tol: float) → float — Min in [a,b]
newton_raphson(f: string, x0: float, tol: float, max_iter: int) → float — Find root
bisection(f: string, a: float, b: float, tol: float) → float — Root in [a,b]
integrate_trapezoid(f: string, a: float, b: float, n: int) → float — Trapezoid rule
integrate_simpson(f: string, a: float, b: float, n: int) → float — Simpson's rule
differentiate(f: string, x: float, h: float) → float — Numerical derivative
