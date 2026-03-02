# print (L3)

## Working Example
```flow
let name = "Alice"
let age = 30
let score = 95.5
let items = [1, 2, 3]

print("Hello, {name}!")
print("Age: {age}, Score: {score}")
print("Items: {items}")

let mut total = 0.0
for i in range(0, len(items))
    let val = items[i]
    total = total + val
    print("  Item {i}: {val}")
end
print("Total: {total}")

let greeting = "Hi"
let target = "world"
print("{greeting}, {target}!")
```

## Expected Output
```
Hello, Alice!
Age: 30, Score: 95.5
Items: [1, 2, 3]
  Item 0: 1
  Item 1: 2
  Item 2: 3
Total: 6.0
Hi, world!
```

## Common Mistakes
- DON'T: `print(name)` --> DO: `print("{name}")` (must use string literal)
- DON'T: `print("a" + b)` --> DO: `print("a {b}")` (use interpolation, not concatenation)
- DON'T: `print(str(x))` --> DO: `print("{x}")` (interpolation handles conversion)
- DON'T: `print("value: " + str(num))` --> DO: `print("value: {num}")`
- DON'T: `println(x)` --> DO: `print("{x}")` (println also needs string literal)

## Edge Cases
- print() ONLY accepts string literals with {var} interpolation
- Any expression in braces is auto-converted: {len(arr)}, {x + 1} work
- Arrays and maps are printed in their display format: [1, 2, 3], {key: val}
