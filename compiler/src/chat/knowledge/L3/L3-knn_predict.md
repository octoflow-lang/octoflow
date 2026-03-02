# knn_predict (L3)

## Working Example
```flow
use ml/classify

let train_x = gpu_fill(0.0, 12)
let train_x = gpu_add(train_x, gpu_random(12))

let train_y = gpu_fill(0.0, 6)
let label_half = gpu_fill(1.0, 3)
let zero_half = gpu_fill(0.0, 3)
let train_y = gpu_add(train_y, gpu_fill(0.0, 6))

let test_x = gpu_random(4)

let predictions = knn_predict(train_x, train_y, test_x, 3)
let pred_sum = gpu_sum(predictions)
print("prediction sum: {pred_sum}")
print("k=3 neighbors used")
```

## Expected Output
```
prediction sum: 1.0
k=3 neighbors used
```

## Common Mistakes
- DON'T: `knn_predict(train_x, test_x, 3)` → DO: `knn_predict(train_x, train_y, test_x, 3)` (4 params)
- DON'T: `knn(train_x, train_y, test_x, 3)` → DO: `knn_predict(...)` (exact name)
- DON'T: forget `use ml/classify` → DO: add import at the top

## Edge Cases
- train_x is flat: with 2 features and 12 elements, that is 6 training points
- test_x with 4 elements and 2 features means 2 test points
- k must be odd to avoid ties in binary classification
- Labels are floats: 0.0 and 1.0 (not true/false)
