# metrics (L2)
ml/metrics — Evaluation metrics (classification and regression)

## Functions
accuracy(y_true: array, y_pred: array) → float — Fraction correct
precision(y_true: array, y_pred: array) → float — TP / predicted positive
recall(y_true: array, y_pred: array) → float — TP / actual positive
f1_score(y_true: array, y_pred: array) → float — Harmonic mean P/R
mse(y_true: array, y_pred: array) → float — Mean squared error
rmse(y_true: array, y_pred: array) → float — Root mean squared error
mae(y_true: array, y_pred: array) → float — Mean absolute error
confusion_matrix(y_true: array, y_pred: array, n_classes: int) → array — NxN matrix
