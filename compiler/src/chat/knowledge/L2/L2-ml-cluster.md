# cluster (L2)
ml/cluster — Clustering (K-means, distance metrics, silhouette)

## Functions
kmeans(data: array, k: int, n_features: int, max_iter: int) → array
  Assign data points to k clusters, return labels
euclidean_distance(a: array, b: array) → float
  Compute Euclidean distance between two points
manhattan_distance(a: array, b: array) → float
  Compute Manhattan distance between two points
silhouette_score(data: array, labels: array, n_features: int) → float
  Compute mean silhouette score for clustering quality
