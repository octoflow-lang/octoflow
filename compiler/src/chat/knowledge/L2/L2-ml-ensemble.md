# ensemble (L2)
ml/ensemble — Ensemble learning (bagging, voting, bootstrap)

## Functions
bagging_predict(models: array, sample: array) → float
  Predict by majority vote across models
weighted_vote(predictions: array, weights: array) → float
  Weighted majority vote across predictions
bootstrap_sample(data: array, size: int) → array
  Draw random sample with replacement
bagging_predict_array(models: array, data: array) → array
  Predict labels for all samples via bagging
