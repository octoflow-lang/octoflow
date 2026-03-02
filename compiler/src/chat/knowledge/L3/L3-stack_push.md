# stack_push (L3)

## Working Example
```flow
use collections/stack

let s = stack_create()

stack_push(s, 10.0)
stack_push(s, 20.0)
stack_push(s, 30.0)

let top = stack_peek(s)
print("top of stack: {top}")

let size = stack_size(s)
print("stack size: {size}")

let a = stack_pop(s)
let b = stack_pop(s)
print("popped: {a}")
print("popped: {b}")

let new_top = stack_peek(s)
print("remaining top: {new_top}")

let remaining = stack_size(s)
print("remaining size: {remaining}")

fn is_balanced(parens)
  let check = stack_create()
  let n = len(parens)
  for i in range(0, n)
    let ch = get(parens, i)
    if ch == 1.0
      stack_push(check, 1.0)
    end
    if ch == 0.0
      let sz = stack_size(check)
      if sz == 0.0
        return 0.0
      end
      stack_pop(check)
    end
  end
  let final_sz = stack_size(check)
  if final_sz == 0.0
    return 1.0
  end
  return 0.0
end

let balanced = is_balanced([1.0, 1.0, 0.0, 0.0])
print("balanced: {balanced}")

let unbalanced = is_balanced([1.0, 0.0, 0.0])
print("unbalanced: {unbalanced}")
```

## Expected Output
```
top of stack: 30.0
stack size: 3.0
popped: 30.0
popped: 20.0
remaining top: 10.0
remaining size: 1.0
balanced: 1.0
unbalanced: 0.0
```

## Common Mistakes
- DON'T: `s.push(10.0)` → DO: `stack_push(s, 10.0)`
- DON'T: `stack_create(100)` → DO: `stack_create()` (no capacity arg)
- DON'T: `if balanced == true` → DO: `if balanced == 1.0`

## Edge Cases
- stack_pop on empty stack returns 0.0
- stack_peek on empty stack returns 0.0
- Stack grows dynamically with no fixed limit
