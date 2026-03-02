# filter (L3)

## Working Example
```flow
let scores = [85, 42, 91, 67, 73, 55, 98, 30]

let passing = filter(scores, fn(x) x >= 60 end)
print("Passing: {passing}")

let boosted = map_each(passing, fn(x) x + 5 end)
print("Boosted: {boosted}")

let total = reduce(boosted, 0, fn(acc, x) acc + x end)
let avg = total / len(boosted)
print("Count: {len(boosted)}")
print("Average: {avg}")
```

## Expected Output
```
Passing: [85, 67, 73, 91, 98]
Boosted: [90, 72, 78, 96, 103]
Count: 5
Average: 87.8
```

## Common Mistakes
- DON'T: `scores.filter(fn)` --> DO: `filter(scores, fn(x) cond end)`
- DON'T: `map(arr, fn)` --> DO: `map_each(arr, fn(x) expr end)` (map() creates a map)
- DON'T: `reduce(arr, fn)` --> DO: `reduce(arr, init, fn(acc, x) expr end)` (initial value required)
- DON'T: forget `end` in inline fn --> DO: `fn(x) x > 5 end`

## Edge Cases
- filter preserves order of matching elements
- map_each returns a new array; original is unchanged
- reduce with empty array returns the initial value
