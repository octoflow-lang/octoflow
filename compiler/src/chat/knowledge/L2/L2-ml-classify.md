# classify (L2)
ml/classify — Classification (KNN, logistic regression, Naive Bayes)

## Functions
knn_predict(data: array, labels: array, point: array, k: int) → float
  Predict label for point using k-nearest neighbors
logistic_regression(x: array, y: array, lr: float, epochs: int) → array
  Train logistic regression, return weights
logistic_predict(x: array, weights: array) → float
  Predict class probability from trained weights
naive_bayes_train(x: array, y: array, n_classes: int) → map
  Train Naive Bayes model, return class statistics
naive_bayes_predict(sample: array, model: map) → float
  Predict class label from trained Naive Bayes model
