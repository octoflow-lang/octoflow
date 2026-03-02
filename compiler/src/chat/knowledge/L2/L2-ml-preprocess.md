# preprocess (L2)
ml/preprocess — ML preprocessing (split, scale, encode, impute)

## Functions
train_test_split(data: array, ratio: float) → map — Split by ratio
shuffle_array(data: array) → array — Random shuffle
minmax_scale(data: array) → array — Scale to [0,1]
zscore_scale(data: array) → array — Standardize to zero mean
feature_scale(data: array, n_features: int) → array — Per-feature scaling
encode_labels(labels: array) → array — Labels to integer codes
impute_missing(data: array, strategy: string) → array — Fill missing values
