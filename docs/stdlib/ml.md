# ml — Machine Learning

Classification, regression, clustering, neural networks, linear algebra,
preprocessing, and evaluation metrics.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `classify` | 5 | KNN, logistic regression, Naive Bayes |
| `regression` | 5 | Linear, ridge, gradient descent |
| `cluster` | 4 | K-means, distance metrics, silhouette |
| `nn` | 12 | Neural network layers, activations, optimizers |
| `tree` | 3 | Decision tree primitives |
| `ensemble` | 5 | Bagging, bootstrap, weighted voting |
| `preprocess` | 7 | Train/test split, scaling, encoding |
| `linalg` | 14 | Matrix operations, solvers |
| `metrics` | 9 | Accuracy, precision, recall, F1, MSE |

## classify

```
use ml.classify
```

| Function | Description |
|----------|-------------|
| `knn_predict(x_train, y_train, x_test, k)` | K-nearest neighbors classification |
| `logistic_regression(x, y, lr, epochs)` | Train logistic regression model |
| `logistic_predict(model, x_arr)` | Predict with logistic regression |
| `naive_bayes_train(x, y, num_classes)` | Train Gaussian Naive Bayes classifier |
| `naive_bayes_predict(model, x_arr)` | Predict with Naive Bayes |

```
use ml.classify
use ml.metrics

let model = logistic_regression(x_train, y_train, 0.01, 1000)
let preds = logistic_predict(model, x_test)
let acc = accuracy(y_test, preds)
print("Accuracy: {acc}")
```

## regression

```
use ml.regression
```

| Function | Description |
|----------|-------------|
| `linear_regression(x, y)` | Ordinary least squares regression |
| `ridge_regression(x, y, alpha)` | Ridge regression with L2 regularization |
| `predict_linear(model, x_arr)` | Predict with trained linear model |
| `r_squared(y_actual, y_predicted)` | R-squared goodness of fit |
| `gradient_descent_linear(x, y, lr, epochs)` | Linear regression via gradient descent |

```
let model = linear_regression(x, y)
let preds = predict_linear(model, x_test)
let r2 = r_squared(y_test, preds)
print("R² = {r2}")
```

## cluster

```
use ml.cluster
```

| Function | Description |
|----------|-------------|
| `kmeans(data, k, max_iter)` | K-means clustering |
| `euclidean_distance(a, b)` | Euclidean distance between vectors |
| `manhattan_distance(a, b)` | Manhattan distance between vectors |
| `silhouette_score(data, labels, k)` | Cluster quality score |

## nn

```
use ml.nn
```

| Function | Description |
|----------|-------------|
| `dense_forward(input, weights, bias, n_in, n_out)` | Dense layer forward pass |
| `relu_forward(arr)` | ReLU activation |
| `sigmoid_forward(arr)` | Sigmoid activation |
| `tanh_forward(arr)` | Tanh activation |
| `softmax(arr)` | Softmax activation |
| `cross_entropy_loss(predicted, target)` | Cross-entropy loss |
| `mse_loss(predicted, target)` | Mean squared error loss |
| `init_weights(n_in, n_out)` | Xavier weight initialization |
| `init_bias(n)` | Zero bias initialization |
| `sgd_update(params, grads, lr)` | SGD parameter update |
| `dropout(arr, rate)` | Dropout regularization |
| `batch_norm(arr)` | Batch normalization |

```
use ml.nn

let w1 = init_weights(784, 128)
let b1 = init_bias(128)
let h = relu_forward(dense_forward(input, w1, b1, 784, 128))
let out = softmax(dense_forward(h, w2, b2, 128, 10))
let loss = cross_entropy_loss(out, target)
```

## tree

```
use ml.tree
```

| Function | Description |
|----------|-------------|
| `gini_impurity(labels, n_classes)` | Gini impurity for split evaluation |
| `decision_stump(x, y, thresholds)` | Single-split decision stump |
| `find_best_split(x, y, n_thresholds)` | Find optimal split threshold |

## ensemble

```
use ml.ensemble
```

| Function | Description |
|----------|-------------|
| `bagging_predict(predictions)` | Majority vote from multiple predictions |
| `weighted_vote(predictions, weights)` | Weighted majority vote |
| `bootstrap_sample(arr)` | Random bootstrap sample |
| `bagging_predict_array(all_predictions, n_predictors, n_samples)` | Batch bagging predictions |
| `bootstrap_indices(n)` | Generate bootstrap index array |

## preprocess

```
use ml.preprocess
```

| Function | Description |
|----------|-------------|
| `train_test_split(arr, test_ratio)` | Split data into train/test sets |
| `shuffle_array(arr)` | Random shuffle |
| `minmax_scale(arr)` | Min-max normalization to [0, 1] |
| `zscore_scale(arr)` | Z-score standardization |
| `feature_scale(arr, method)` | Scale by method name ("minmax" or "zscore") |
| `encode_labels(labels)` | Encode categorical labels as integers |
| `impute_missing(arr, strategy, missing_val)` | Fill missing values (mean, median, zero) |

## linalg

```
use ml.linalg
```

| Function | Description |
|----------|-------------|
| `mat_create(rows, cols, fill_val)` | Create matrix (flat row-major array) |
| `mat_identity(n)` | n x n identity matrix |
| `mat_get(m, rows, cols, r, c)` | Get element at (r, c) |
| `mat_set(m, rows, cols, r, c, val)` | Set element at (r, c) |
| `mat_mul(a, b, m, n, k)` | Matrix multiply (m x n) * (n x k) |
| `mat_transpose(a, rows, cols)` | Transpose matrix |
| `mat_add(a, b)` | Element-wise addition |
| `mat_scale(a, scalar)` | Scalar multiplication |
| `mat_det_2x2(a)` | Determinant of 2x2 matrix |
| `mat_det_3x3(a)` | Determinant of 3x3 matrix |
| `mat_inverse_2x2(a)` | Inverse of 2x2 matrix |
| `outer_product(a, b)` | Outer product of two vectors |
| `mat_trace(a, n)` | Trace of n x n matrix |
| `solve_2x2(a, b)` | Solve 2x2 linear system (Cramer's rule) |

## metrics

```
use ml.metrics
```

| Function | Description |
|----------|-------------|
| `accuracy(y_true, y_pred)` | Classification accuracy |
| `precision(y_true, y_pred, positive)` | Precision for class |
| `recall(y_true, y_pred, positive)` | Recall for class |
| `f1_score(y_true, y_pred, positive)` | F1 score |
| `mse(y_true, y_pred)` | Mean squared error |
| `rmse(y_true, y_pred)` | Root mean squared error |
| `mae(y_true, y_pred)` | Mean absolute error |
| `confusion_matrix(y_true, y_pred, num_classes)` | Confusion matrix |
| `mean_absolute_percentage_error(y_true, y_pred)` | MAPE |
