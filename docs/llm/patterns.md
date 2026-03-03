# OctoFlow Patterns — Copy-Paste Code Blocks

## Hello World
```
print("Hello, World!")
```

## GPU Hello World
```
let a = gpu_fill(1.0, 1000000)
let b = gpu_fill(2.0, 1000000)
let c = gpu_add(a, b)
let total = gpu_sum(c)
print("Sum of 1M elements: {total}")
```

## Read File → Process → Write File
```
let lines = read_lines("input.txt")
let mut result = ""
let nl = chr(10)
for line in lines
    let upper = to_upper(line)
    result = result + upper + nl
end
write_file("output.txt", result)
```

## GPU Array in Loop (Mutable Pattern)
```
let N = 10000
let mut arr = gpu_fill(1.0, N)
for i in range(0, 10)
    let doubled = gpu_scale(arr, 2.0)
    let arr = gpu_add(doubled, arr)    // re-bind with let, NOT assignment
end
let total = gpu_sum(arr)
print("result: {total}")
```

## Build String Without Escapes
```
let nl = chr(10)        // newline
let tab = chr(9)        // tab
let esc = chr(27)       // ESC (for ANSI)
let line = "col1" + tab + "col2" + tab + "col3" + nl
```

## CSV Generation (GPU + String Building)
```
let N = 1000
let ids = gpu_range(1.0, N + 1, 1.0)
let vals = gpu_sin(ids)
let nl = chr(10)
let mut csv = "id,value" + nl
for i in range(0, N)
    let sid = str(int(ids[i]))
    let sv = str(vals[i])
    csv = csv + sid + "," + sv + nl
end
write_file("output.csv", csv)
```

## JSON Parse and Process
```
let text = read_file("data.json")
let obj = json_parse(text)
let name = map_get(obj, "name")
let score = map_get(obj, "score")
print("name={name} score={score}")
```

## HTTP Request with Error Handling
```
let resp = try(http_get("https://api.example.com/data"))
if resp.ok == 1.0
    let data = json_parse(resp.value.body)
    let result = map_get(data, "result")
    print("Got: {result}")
else
    print("Error: {resp.error}")
end
```

## Mandelbrot on GPU (Mask Pattern for Conditionals)
```
// GPU has no if/else — use mask arithmetic instead
// mask = 1.0 where condition true, 0.0 where false
// result = new_value * mask + old_value * (1 - mask)

let threshold = gpu_fill(4.0, N)
let mag = gpu_add(zr2, zi2)
let diff = gpu_sub(threshold, mag)
let clamped = gpu_clamp(diff, 0.0, 1.0)
let mask = gpu_ceil(clamped)              // 1 where mag < 4
let inv = gpu_sub(gpu_fill(1.0, N), mask) // 0 where mag < 4

// Conditional update: only modify elements where mask = 1
let new_masked = gpu_mul(new_val, mask)
let old_masked = gpu_mul(old_val, inv)
let result = gpu_add(new_masked, old_masked)
```

## Timer / Benchmark
```
let t0 = now_ms()
// ... work ...
let elapsed = int(now_ms() - t0)
let se = str(elapsed)
print("Elapsed: {se}ms")
```

## ANSI Terminal Colors
```
let esc = chr(27)
let red = esc + "[31m"
let green = esc + "[32m"
let yellow = esc + "[33m"
let blue = esc + "[34m"
let bold = esc + "[1m"
let reset = esc + "[0m"
let block = chr(9608)  // █ full block character

// Truecolor (24-bit): ESC[38;2;R;G;Bm
let r = "255"
let g = "100"
let b = "50"
let color = esc + "[38;2;" + r + ";" + g + ";" + b + "m"
let msg = color + "colored text" + reset
print("{msg}")
```

## Map Iteration
```
let mut m = map()
map_set(m, "alice", 95)
map_set(m, "bob", 82)
let keys = map_keys(m)
for k in keys
    let v = map_get(m, k)
    let msg = k + ": " + str(v)
    print("{msg}")
end
```

## Standard Library Import
```
use collections.heap

let mut h = heap_create()
heap_push(h, 5)
heap_push(h, 2)
heap_push(h, 8)
let smallest = heap_pop(h)     // 2
let msg = "smallest: " + str(smallest)
print("{msg}")
```

## Matrix Multiply on GPU
```
let m = 64
let k = 128
let n = 32
let a = gpu_random(m * k)     // 64x128 matrix
let b = gpu_random(k * n)     // 128x32 matrix
let c = gpu_matmul(a, b, m, n, k)  // 64x32 result
let total = gpu_sum(c)
print("matmul sum: {total}")
```

## PPM Image Output
```
let W = 200
let H = 100
let N = W * H
// ... compute r, g, b GPU arrays (0-255 range) ...
let nl = chr(10)
let sw = str(int(W))
let sh = str(int(H))
let mut ppm = "P3" + nl + sw + " " + sh + nl + "255" + nl
for y in range(0, H)
    let mut row = ""
    for x in range(0, W)
        let i = y * W + x
        let ri = str(int(r[i]))
        let gi = str(int(g[i]))
        let bi = str(int(b[i]))
        row = row + ri + " " + gi + " " + bi + " "
    end
    ppm = ppm + row + nl
end
write_file("image.ppm", ppm)
```

## Error-Proof Print (The Golden Pattern)
```
// ALWAYS assign to variable first, then interpolate
let x = 42
let msg = "The answer is " + str(x) + " units"
print("{msg}")

// NEVER do any of these:
// print(msg)           ← ERROR: variable, not literal
// print("x=" + str(x)) ← ERROR: expression, not literal
// print(str(x))        ← ERROR: function call, not literal
```
