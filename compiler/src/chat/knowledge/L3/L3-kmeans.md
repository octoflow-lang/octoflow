# kmeans (L3)

## Working Example
```flow
use ml/cluster

// 15 points, 2 features each (flat array: x1,y1,x2,y2,...)
let data = [
    1.0, 2.0,  1.5, 1.8,  1.2, 2.3,  0.8, 1.5,  1.1, 2.1,
    5.0, 5.5,  5.2, 5.0,  4.8, 5.3,  5.1, 4.9,  5.3, 5.2,
    9.0, 1.0,  8.5, 1.2,  9.2, 0.8,  8.8, 1.5,  9.1, 1.1
]
let labels = kmeans(data, 3, 2, 100)
let n = len(labels)
print("Points clustered: {n}")
for i in range(0, n)
    let l = labels[i]
    print("Point {i}: cluster {l}")
end
```

## Expected Output
```
Points clustered: 15
Point 0: cluster 0.0
Point 1: cluster 0.0
...
Point 14: cluster 2.0
```

## Common Mistakes
- DON'T: `kmeans(data, 3)` → DO: `kmeans(data, 3, 2, 100)` (all 4 params required)
- DON'T: `kmeans(data, k=3, features=2)` → DO: positional args only
- DON'T: pass 2D arrays → DO: flatten to 1D with n_features to specify dimensionality

## Edge Cases
- Data is a flat array; with n_features=2 and 30 elements, that is 15 points
- Labels are 0-indexed floats (0.0, 1.0, 2.0 for k=3)
- max_iter=100 is a safe default; increase for large datasets
- k must not exceed the number of data points
