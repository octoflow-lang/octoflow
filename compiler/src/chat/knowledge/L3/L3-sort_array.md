# sort_array (L3)

## Working Example
```flow
let mut nums = [5.0, 2.0, 8.0, 1.0, 9.0, 3.0]
let sorted = sort_array(nums)

let first = get(sorted, 0)
let last = get(sorted, 5)
print("sorted first: {first}, last: {last}")

let mut names = ["delta", "alpha", "charlie", "bravo"]
let alpha_sorted = sort_array(names)
let s0 = get(alpha_sorted, 0)
let s1 = get(alpha_sorted, 1)
print("alphabetical: {s0}, {s1}")

let mut scores = [90.0, 75.0, 88.0, 42.0, 100.0]
let desc = sort_by(scores, fn(a, b)
  return b - a
end)
let top = get(desc, 0)
let bottom = get(desc, 4)
print("top score: {top}, lowest: {bottom}")
```

## Expected Output
```
sorted first: 1.0, last: 9.0
alphabetical: alpha, bravo
top score: 100.0, lowest: 42.0
```

## Common Mistakes
- DON'T: `nums.sort()` → DO: `sort_array(nums)`
- DON'T: `sort_by(arr, fn(a, b) a < b end)` → DO: return numeric difference `b - a`
- DON'T: `sort_array(nums);` → DO: `sort_array(nums)` (no semicolons)

## Edge Cases
- sort_array on an empty list returns an empty list
- sort_array is stable for equal elements
- sort_by comparator must return a number: negative, zero, or positive
