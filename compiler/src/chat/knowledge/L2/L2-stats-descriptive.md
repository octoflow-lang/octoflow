# descriptive (L2)
stats/descriptive — Descriptive statistics

## Functions
skewness(data: array) → float — Distribution asymmetry
kurtosis(data: array) → float — Tail weight measure
iqr(data: array) → float — Interquartile range
percentile(data: array, p: float) → float — P-th percentile
weighted_mean(data: array, weights: array) → float — Weighted mean
trimmed_mean(data: array, trim: float) → float — Trimmed mean
describe(data: array) → map — Summary stats (mean, std, min, max, quartiles)
mode(data: array) → float — Most frequent value
geometric_mean | harmonic_mean(data: array) → float — Geometric/harmonic mean
coeff_of_variation(data: array) → float — CV (std/mean)
zscore(data: array) → array — Standardize to z-scores
