# train_test_split (L3)

## Working Example
```flow
use ml/preprocess

let data = gpu_random(100)
let split = train_test_split(data, 0.8)
let train = split.train
let test = split.test

let train_ones = gpu_fill(1.0, 80)
let train_count = gpu_sum(train_ones)
let test_ones = gpu_fill(1.0, 20)
let test_count = gpu_sum(test_ones)
print("train size: {train_count}")
print("test size: {test_count}")

let train_mean = gpu_mean(train)
let test_mean = gpu_mean(test)
print("train mean: {train_mean}")
print("test mean: {test_mean}")
```

## Expected Output
```
train size: 80.0
test size: 20.0
train mean: 0.4932
test mean: 0.5104
```

## Common Mistakes
- DON'T: `train_test_split(data, 80)` → DO: `train_test_split(data, 0.8)` (ratio is a float 0-1)
- DON'T: forget `use ml/preprocess` → DO: always import the module first
- DON'T: `split[0]` → DO: `split.train` and `split.test` (use field access)

## Edge Cases
- Ratio 0.8 means 80% train, 20% test
- Data is shuffled before splitting
- Both .train and .test are GPU arrays
