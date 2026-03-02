# transform (L2)
data/transform — Data transformations (normalize, standardize, one-hot, clip, interpolate, resample, bin)

## Functions
normalize(arr: array) → array
  Scale values to [0, 1] range
standardize(arr: array) → array
  Zero-mean, unit-variance scaling
one_hot(arr: array) → array
  One-hot encode categorical values
clip(arr: array, lo: float, hi: float) → array
  Clamp values to [lo, hi]
interpolate_missing(arr: array) → array
  Fill missing values via linear interpolation
resample(arr: array, n: int) → array
  Resample array to n evenly-spaced points
bin_data(arr: array, n: int) → array
  Bin values into n buckets
scale_to_range(arr: array, lo: float, hi: float) → array
  Scale values to [lo, hi] range
