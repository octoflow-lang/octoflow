# OctoFlow Examples — Annotated

Complete, tested, working examples. Every line compiles and runs.

---

## 1. Hello World
```
print("Hello, World!")
```
**Run**: `octoflow run hello.flow`

---

## 2. GPU Hello World — 1M Elements
```
let a = gpu_fill(1.0, 1000000)
let b = gpu_fill(2.0, 1000000)
let c = gpu_add(a, b)
let total = gpu_sum(c)
print("Hello from the GPU!")
print("1,000,000 elements: 1.0 + 2.0 = {total}")
```
**Output**: `1,000,000 elements: 1.0 + 2.0 = 3000000`
**Key**: GPU arrays stay in VRAM until gpu_sum downloads the result.

---

## 3. Variables and Control Flow
```
let mut x = 0
while x < 5
    let msg = "x = " + str(x)
    print("{msg}")
    x = x + 1
end

for i in range(0, 3)
    if i == 0
        print("zero")
    elif i == 1
        print("one")
    else
        print("two")
    end
end
```
**Key**: All numbers are f32. `str()` converts to string. `print("{var}")` for interpolation.

---

## 4. Functions and Lambdas
```
fn fibonacci(n)
    if n <= 1.0
        return n
    end
    return fibonacci(n - 1.0) + fibonacci(n - 2.0)
end

let result = fibonacci(10.0)
print("fib(10) = {result}")

// Higher-order functions with lambdas
let nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
let evens = filter(nums, fn(x) x % 2 == 0 end)
let doubled = map_each(evens, fn(x) x * 2 end)
let total = reduce(doubled, 0, fn(acc, x) acc + x end)
print("Sum of doubled evens: {total}")
```
**Output**: `fib(10) = 55`, `Sum of doubled evens: 60`
**Key**: Lambdas use `fn(params) expr end` syntax.

---

## 5. Hashmaps and JSON
```
let mut config = map()
map_set(config, "host", "localhost")
map_set(config, "port", 8080)
map_set(config, "debug", 1)

let keys = map_keys(config)
for k in keys
    let v = map_get(config, k)
    let line = k + " = " + str(v)
    print("{line}")
end

// JSON serialization
let json = json_stringify(config)
print("JSON: {json}")
```
**Key**: `map()` creates empty map. `map_set/get/has/keys` for operations. `json_stringify` for serialization.

---

## 6. GPU Mandelbrot Fractal
```
let W = 200
let H = 112
let N = W * H
let max_iter = 50
let cx = -0.745
let cy = 0.186
let zoom = 0.01

let gpu_one = gpu_fill(1.0, N)
let gpu_four = gpu_fill(4.0, N)
let gpu_W = gpu_fill(W, N)

let aspect = H / W
let x_min = cx - zoom * 0.5
let y_min = cy - zoom * aspect * 0.5
let dx = zoom / W
let dy = zoom * aspect / H

let idx = gpu_range(0.0, N, 1.0)
let div_tmp = gpu_div(idx, gpu_W)
let row = gpu_floor(div_tmp)
let mul_tmp = gpu_mul(row, gpu_W)
let col = gpu_sub(idx, mul_tmp)
let col_sc = gpu_scale(col, dx)
let xmin_arr = gpu_fill(x_min, N)
let cr = gpu_add(col_sc, xmin_arr)
let row_sc = gpu_scale(row, dy)
let ymin_arr = gpu_fill(y_min, N)
let ci = gpu_add(row_sc, ymin_arr)

let mut zr = gpu_fill(0.0, N)
let mut zi = gpu_fill(0.0, N)
let mut count = gpu_fill(0.0, N)

for k in range(0, max_iter)
    let zr2 = gpu_mul(zr, zr)
    let zi2 = gpu_mul(zi, zi)
    let mag = gpu_add(zr2, zi2)
    let diff = gpu_sub(gpu_four, mag)
    let clamped = gpu_clamp(diff, 0.0, 1.0)
    let mask = gpu_ceil(clamped)
    let inv = gpu_sub(gpu_one, mask)
    let dz = gpu_sub(zr2, zi2)
    let nr = gpu_add(dz, cr)
    let prod = gpu_mul(zr, zi)
    let prod2 = gpu_scale(prod, 2.0)
    let ni = gpu_add(prod2, ci)
    let nr_m = gpu_mul(nr, mask)
    let zr_k = gpu_mul(zr, inv)
    let zr = gpu_add(nr_m, zr_k)
    let ni_m = gpu_mul(ni, mask)
    let zi_k = gpu_mul(zi, inv)
    let zi = gpu_add(ni_m, zi_k)
    let count = gpu_add(count, mask)
end

let s = gpu_sum(count)
let sn = str(int(N))
let ss = str(s)
print("{sn} pixels computed, total iterations: {ss}")
```
**Key**: GPU conditional update uses mask * new + (1-mask) * old pattern. `let` re-binding in loops overwrites GPU arrays.

---

## 7. Error Handling with try()
```
let r = try(read_file("nonexistent.txt"))
if r.ok == 0.0
    print("Caught: {r.error}")
end

let s = try(str(42))
print("value={s.value} ok={s.ok}")

fn safe_divide(a, b)
    if b == 0.0
        return 0.0
    end
    return a / b
end

let result = safe_divide(10.0, 3.0)
print("10/3 = {result}")
```
**Key**: `try()` decomposes to `.value`, `.ok`, `.error` fields.

---

## 8. String Operations
```
let name = "  Hello World  "
let trimmed = trim(name)
let upper = to_upper(trimmed)
let lower = to_lower(trimmed)
let has_hello = contains(name, "Hello")
let idx = index_of(name, "World")
let sub = substr(name, 8, 5)
let replaced = replace(name, "World", "OctoFlow")

print("trimmed: {trimmed}")
print("upper: {upper}")
print("lower: {lower}")
print("contains Hello: {has_hello}")
print("index of World: {idx}")
print("substr: {sub}")
print("replaced: {replaced}")

// No escape sequences! Use chr() for special characters
let nl = chr(10)
let tab = chr(9)
let line = "col1" + tab + "col2" + nl + "val1" + tab + "val2"
print("{line}")
```
**Key**: No `\n` or `\t`. Use `chr(10)` and `chr(9)`.

---

## 9. File I/O and CSV
```
// Requires: --allow-write --allow-read
let nl = chr(10)
let mut content = "name,score" + nl
content = content + "Alice,95" + nl
content = content + "Bob,82" + nl
content = content + "Charlie,91" + nl
write_file("test_data.csv", content)

let rows = read_csv("test_data.csv")
for row in rows
    let name = map_get(row, "name")
    let score = map_get(row, "score")
    let line = name + ": " + str(score)
    print("{line}")
end
```
**Key**: `read_csv` returns array of maps. Each map has column names as keys.

---

## 10. Standard Library — Collections
```
use collections.heap

let mut h = heap_create()
heap_push(h, 42)
heap_push(h, 17)
heap_push(h, 88)
heap_push(h, 5)

let mut sorted = ""
while heap_is_empty(h) == 0.0
    let val = heap_pop(h)
    sorted = sorted + str(val) + " "
end
print("Sorted: {sorted}")
```
**Output**: `Sorted: 5 17 42 88`
**Key**: `use domain.module` imports stdlib. `heap_create/push/pop` for priority queue.
