# tree (L2)
ml/tree — Decision tree primitives

## Functions
gini_impurity(labels: array) → float
  Compute Gini impurity for a set of labels
decision_stump(data: array, labels: array, n_features: int) → map
  Train single-split decision stump
find_best_split(data: array, labels: array, n_features: int) → map
  Find feature and threshold that minimizes impurity
