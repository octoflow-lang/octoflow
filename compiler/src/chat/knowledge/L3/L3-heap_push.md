# heap_push (L3)

## Working Example
```flow
use collections/heap

let h = heap_create()

heap_push(h, 5.0)
heap_push(h, 1.0)
heap_push(h, 8.0)
heap_push(h, 3.0)
heap_push(h, 2.0)

let top = heap_peek(h)
print("min element: {top}")

let size = heap_size(h)
print("heap size: {size}")

let mut sorted = []
for i in range(0, 5)
  let val = heap_pop(h)
  push(sorted, val)
  print("popped: {val}")
end

let final_size = heap_size(h)
print("heap size after drain: {final_size}")

let h2 = heap_create()
heap_push(h2, 42.0)
heap_push(h2, 7.0)
heap_push(h2, 99.0)
let first = heap_pop(h2)
let second = heap_pop(h2)
print("priority order: {first}, {second}")
```

## Expected Output
```
min element: 1.0
heap size: 5
popped: 1.0
popped: 2.0
popped: 3.0
popped: 5.0
popped: 8.0
heap size after drain: 0.0
priority order: 7.0, 42.0
```

## Common Mistakes
- DON'T: `h.push(5.0)` → DO: `heap_push(h, 5.0)` (functions, not methods)
- DON'T: `heap_pop()` → DO: `heap_pop(h)` (pass the heap)
- DON'T: `heap_create(10)` with capacity → DO: `heap_create()` (no args)

## Edge Cases
- heap_create builds a min-heap by default (smallest first)
- heap_pop on an empty heap returns 0.0
- heap_peek does not remove the element
