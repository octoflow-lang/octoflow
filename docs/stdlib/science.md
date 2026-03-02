# science — Scientific Computing

Numerical methods, signal processing, interpolation, optimization,
physics simulation, and matrix utilities.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `calculus` | 4 | Derivatives, integration |
| `signal` | 11 | Convolution, filters, windowing |
| `matrix` | 7 | Matrix norms, solvers, diagnostics |
| `constants` | 17 | Mathematical and physical constants |
| `interpolate` | 4 | Linear, bilinear, nearest interpolation |
| `optimize` | 7 | Gradient descent, root finding, integration |
| `physics` | 8 | Euler/RK4 integration, mechanics |

## calculus

```
use science.calculus
```

| Function | Description |
|----------|-------------|
| `derivative(arr, dx)` | Numerical derivative (central differences) |
| `integrate_trapz(arr, dx)` | Trapezoidal integration |
| `cumulative_trapz(arr, dx)` | Cumulative trapezoidal integration |
| `second_derivative(arr, dx)` | Second numerical derivative |

```
let positions = [0, 1, 4, 9, 16, 25]
let dt = 1.0
let velocity = derivative(positions, dt)
let distance = integrate_trapz(velocity, dt)
```

## signal

```
use science.signal
```

| Function | Description |
|----------|-------------|
| `convolve(signal, kernel)` | Discrete convolution |
| `moving_avg_filter(signal, window)` | Moving average smoothing |
| `gaussian_kernel(size, sigma)` | Create Gaussian kernel |
| `hamming_window(n)` | Hamming window function |
| `hanning_window(n)` | Hanning window function |
| `blackman_window(n)` | Blackman window function |
| `cross_correlate(a, b)` | Cross-correlation |
| `bandpass(signal, low_freq, high_freq, sample_rate)` | Bandpass FIR filter |
| `envelope(signal)` | Amplitude envelope |
| `zero_crossings(signal)` | Count zero crossings |
| `peak_detect(signal, threshold)` | Detect peaks above threshold |

```
let raw = csv_column(csv_read("sensor.csv"), "reading")
let smoothed = moving_avg_filter(raw, 5)
let peaks = peak_detect(smoothed, 2.0)
print("Found {len(peaks)} peaks")
```

## matrix

```
use science.matrix
```

| Function | Description |
|----------|-------------|
| `mat_norm(a, n)` | Frobenius norm of n x n matrix |
| `mat_diag(a, n)` | Extract diagonal from n x n matrix |
| `mat_from_diag(d)` | Create diagonal matrix from vector |
| `mat_solve_triangular(a, b, n)` | Solve lower triangular system (forward substitution) |
| `mat_solve_upper(a, b, n)` | Solve upper triangular system (back substitution) |
| `mat_row_norm(a, n, row)` | L2 norm of matrix row |
| `mat_col_norm(a, n, col)` | L2 norm of matrix column |

## constants

```
use science.constants
```

Mathematical constants:

| Constant | Value |
|----------|-------|
| `PI` | 3.14159265 |
| `E` | 2.71828182 |
| `TAU` | 6.28318530 |
| `GOLDEN_RATIO` | 1.61803398 |
| `SQRT2` | 1.41421356 |
| `LN2` | 0.69314718 |
| `LN10` | 2.30258509 |

Physical constants (SI units):

| Constant | Description |
|----------|-------------|
| `SPEED_OF_LIGHT` | 299,792,458 m/s |
| `PLANCK` | 6.626e-34 J·s |
| `BOLTZMANN` | 1.380e-23 J/K |
| `AVOGADRO` | 6.022e23 mol⁻¹ |
| `GRAVITY` | 9.80665 m/s² |
| `GAS_CONSTANT` | 8.314 J/(mol·K) |
| `ELECTRON_MASS` | 9.109e-31 kg |
| `PROTON_MASS` | 1.672e-27 kg |
| `ELEMENTARY_CHARGE` | 1.602e-19 C |
| `VACUUM_PERMITTIVITY` | 8.854e-12 F/m |

## interpolate

```
use science.interpolate
```

| Function | Description |
|----------|-------------|
| `linear_interp(x_data, y_data, x)` | Linear interpolation at point x |
| `linear_interp_array(x_data, y_data, x_new)` | Linear interpolation at multiple points |
| `bilinear_interp(x, y, x0, x1, y0, y1, q00, q10, q01, q11)` | Bilinear interpolation on 2D grid |
| `nearest_interp(x_data, y_data, x)` | Nearest-neighbor interpolation |

## optimize

```
use science.optimize
```

| Function | Description |
|----------|-------------|
| `gradient_descent(x0, lr, max_iter, grad_fn_name)` | Gradient descent minimization |
| `golden_section(a, b, tol, max_iter)` | Golden section search for minimum |
| `newton_raphson(x0, tol, max_iter)` | Newton-Raphson root finding |
| `bisection(a, b, tol, max_iter)` | Bisection root finding |
| `integrate_trapezoid(y_data, dx)` | Trapezoidal integration |
| `integrate_simpson(y_data, dx)` | Simpson's rule integration |
| `differentiate(y_data, dx)` | Numerical differentiation |

## physics

```
use science.physics
```

| Function | Description |
|----------|-------------|
| `integrate_euler(state, dt, accel_fn_name)` | Euler integration for ODEs |
| `integrate_rk4(x, v, a, dt)` | 4th-order Runge-Kutta |
| `spring_damper(x, v, k, c, m, dt)` | Spring-damper system step |
| `projectile(x, y, vx, vy, dt)` | Projectile motion step |
| `kinetic_energy(m, v)` | Kinetic energy (½mv²) |
| `potential_energy(m, h)` | Gravitational potential energy (mgh) |
| `gravitational_force(m1, m2, r)` | Newton's gravitational force |
| `wave_equation_1d(u, c, dx, dt, steps)` | 1D wave equation solver |

```
// Simulate projectile
let mut x = 0.0
let mut y = 0.0
let mut vx = 10.0
let mut vy = 20.0
while y >= 0.0
    let state = projectile(x, y, vx, vy, 0.01)
    x = state.x
    y = state.y
    vx = state.vx
    vy = state.vy
end
```
