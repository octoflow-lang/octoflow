# calculus (L2)
science/calculus — Numerical calculus (differentiation and integration)

## Functions
derivative(f: string, x: float, h: float) → float
  Numerical derivative using central difference
integrate_trapz(y: array, dx: float) → float
  Trapezoidal integration over sampled data
cumulative_trapz(y: array, dx: float) → array
  Cumulative trapezoidal integral
second_derivative(f: string, x: float, h: float) → float
  Second derivative via central difference
