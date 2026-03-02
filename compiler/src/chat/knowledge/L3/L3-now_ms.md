# now_ms (L3)

## Working Example
```flow
let t0 = now_ms()

let mut sum = 0.0
for i in range(0, 1000000)
  sum = sum + i
end

let t1 = now_ms()
let elapsed = t1 - t0
print("sum: {sum}")
print("loop took {elapsed} ms")

let t2 = now_ms()

fn fib(n)
  if n < 2.0
    return n
  end
  return fib(n - 1.0) + fib(n - 2.0)
end

let result = fib(25.0)
let t3 = now_ms()
let fib_time = t3 - t2
print("fib(25) = {result}")
print("fib took {fib_time} ms")

let total = t3 - t0
print("total runtime: {total} ms")
```

## Expected Output
```
sum: 499999500000.0
loop took 42.0 ms
fib(25) = 75025.0
fib took 187.0 ms
total runtime: 229.0 ms
```

*(Timing values are approximate and vary by machine.)*

## Common Mistakes
- DON'T: `now()` or `time()` → DO: `now_ms()`
- DON'T: `let elapsed = t1 - t0;` → DO: `let elapsed = t1 - t0` (no semicolons)
- DON'T: `print(elapsed)` → DO: `print("elapsed: {elapsed}")` (must use string literal)

## Edge Cases
- now_ms() returns a float representing milliseconds since epoch
- Precision is platform-dependent, typically 1ms granularity
- Calling now_ms() itself has negligible overhead
