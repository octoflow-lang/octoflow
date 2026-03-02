# correlation (L2)
stats/correlation — Correlation and regression fitting

## Functions
pearson(x: array, y: array) → float — Pearson correlation
spearman(x: array, y: array) → float — Spearman rank correlation
rank_array(data: array) → array — Convert values to ranks
covariance(x: array, y: array) → float — Sample covariance
linear_fit(x: array, y: array) → map — Least-squares line fit
residuals(x: array, y: array, model: map) → array — Residuals from model
polynomial_fit(x: array, y: array, degree: int) → array — Polynomial coefficients
