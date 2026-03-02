# interpolate (L2)
science/interpolate — Interpolation methods

## Functions
linear_interp(x0: float, y0: float, x1: float, y1: float, x: float) → float
  Linear interpolation between two points
linear_interp_array(xs: array, ys: array, x: float) → float
  Piecewise linear interpolation over arrays
bilinear_interp(grid: array, x: float, y: float, rows: int, cols: int) → float
  Bilinear interpolation on 2D grid
nearest_interp(xs: array, ys: array, x: float) → float
  Nearest-neighbor interpolation
