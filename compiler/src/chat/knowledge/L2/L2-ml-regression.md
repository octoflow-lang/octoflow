# regression (L2)
ml/regression — Regression (linear, ridge, gradient descent)

## Functions
linear_regression(x: array, y: array) → map
  Fit ordinary least squares, return slope + intercept
ridge_regression(x: array, y: array, lambda: float) → map
  Fit ridge regression with L2 penalty
predict_linear(x: array, model: map) → array
  Predict using fitted linear model
r_squared(y_true: array, y_pred: array) → float
  Coefficient of determination (R^2)
gradient_descent_linear(x: array, y: array, lr: float, epochs: int) → map
  Fit linear model via gradient descent
