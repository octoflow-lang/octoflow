# loom_boot (L3)

## Working Example
```flow
let gpu = loom_boot()

let n = 1024.0
let buf = loom_build(gpu, n)

let mut data = []
for i in range(0, 1024)
  push(data, i * 2.0)
end

loom_write(buf, data)

loom_dispatch(gpu, buf, "square", n)

let result = loom_read(buf, n)

let v0 = get(result, 0)
let v1 = get(result, 1)
let v10 = get(result, 10)
print("result[0]: {v0}")
print("result[1]: {v1}")
print("result[10]: {v10}")

loom_free(buf)
print("GPU buffer released")
```

## Expected Output
```
result[0]: 0.0
result[1]: 4.0
result[10]: 400.0
GPU buffer released
```

*(Each value i*2.0 is squared by the kernel: (i*2)^2.)*

## Common Mistakes
- DON'T: `vm_boot()` → DO: `loom_boot()` (never use vm_ prefix)
- DON'T: `loom_dispatch(gpu, buf, "square")` → DO: include element count `n`
- DON'T: forget `loom_free(buf)` → DO: always free GPU buffers

## Edge Cases
- loom_boot() fails if no GPU is available; check return value
- loom_read returns a list of floats regardless of kernel type
- Buffer size must match the data length written with loom_write
