# loom_dispatch_jit (L3)

## Working Example
```flow
let gpu = loom_boot()

let n = 512.0
let buf = loom_build(gpu, n)

let mut data = []
for i in range(0, 512)
  push(data, i * 1.0)
end
loom_write(buf, data)

let kernel_src = "fn(x) return x * x + 1.0 end"
loom_dispatch_jit(gpu, buf, kernel_src, n)

let result = loom_read(buf, n)

let r0 = get(result, 0)
let r1 = get(result, 1)
let r5 = get(result, 5)
let r100 = get(result, 100)
print("f(0) = {r0}")
print("f(1) = {r1}")
print("f(5) = {r5}")
print("f(100) = {r100}")

loom_free(buf)
print("JIT kernel complete")
```

## Expected Output
```
f(0) = 1.0
f(1) = 2.0
f(5) = 26.0
f(100) = 10001.0
```

## Common Mistakes
- DON'T: `loom_dispatch(gpu, buf, kernel_src, n)` → DO: `loom_dispatch_jit(...)` for runtime-compiled kernels
- DON'T: `vm_dispatch_jit(...)` → DO: `loom_dispatch_jit(...)` (never vm_ prefix)
- DON'T: pass a named function → DO: pass kernel source as a string literal

## Edge Cases
- JIT compilation adds latency on first call; subsequent calls with same source are cached
- Kernel source must be a single-argument function: `"fn(x) ... end"`
- loom_dispatch_jit validates syntax before GPU upload; invalid source returns error
