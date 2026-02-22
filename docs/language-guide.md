# OctoFlow Language Guide

> This document is designed as LLM context. Feed it to your AI assistant (Claude, ChatGPT, Copilot) and it can write, debug, and explain OctoFlow code. Use it as RAG, system prompt, or project context.

## Overview

OctoFlow is a GPU-native general-purpose programming language. The GPU is the primary execution target. The CPU handles I/O. Data born on the GPU stays on the GPU.

- Three types: `float` (f32), `string` (UTF-8), `map` (key-value)
- Booleans are floats: `1.0` = true, `0.0` = false
- Arrays hold floats, strings, or both
- Immutable by default, `let mut` for mutable
- No semicolons, no braces — blocks end with `end`
- No string escape sequences — use `ord(c)` / `chr(n)` for special characters
- No scientific notation — use `6.674 * pow(10.0, -11.0)` instead of `6.674e-11`

## Variables

```
let x = 10              // immutable
let mut y = 20          // mutable
y = y + 1               // reassignment (mut only)
let name = "OctoFlow"   // string
let items = [1, 2, 3]   // array
let mut m = map()        // empty hashmap
```

## Operators

```
// Arithmetic
+  -  *  /  %

// Comparison
==  !=  <  >  <=  >=

// Logical
&&  ||  !

// String concatenation
let greeting = "Hello, " + name + "!"

// String interpolation (variables only, not expressions)
print("Hello, {name}!")
print("x is {x}")
```

## Control Flow

```
// If / elif / else
if x > 10
    print("big")
elif x > 5
    print("medium")
else
    print("small")
end

// While loop
let mut i = 0
while i < 10
    print("{i}")
    i = i + 1
end

// For loop (range)
for i in range(0, 10)
    print("{i}")
end

// For-each (arrays)
let items = [10, 20, 30]
for item in items
    print("{item}")
end

// Break and continue
for i in range(0, 100)
    if i == 50
        break
    end
    if i % 2 == 0
        continue
    end
    print("{i}")
end
```

## Functions

```
fn add(a, b)
    return a + b
end

fn greet(name)
    print("Hello, {name}!")
end

let result = add(3, 4)
greet("World")

// Functions can return floats, strings, arrays, or maps
fn make_list(n)
    let mut arr = []
    for i in range(0, n)
        push(arr, i)
    end
    return arr
end

// Recursive
fn fibonacci(n)
    if n <= 1
        return n
    end
    return fibonacci(n - 1) + fibonacci(n - 2)
end
```

Functions get a snapshot of caller scalars (changes don't propagate back). Arrays propagate back (pass by reference).

## Lambdas and Higher-Order Functions

```
let double = fn(x) x * 2 end
let result = double(21)    // 42

let nums = [1, 2, 3, 4, 5]
let evens = filter(nums, fn(x) x % 2 == 0 end)       // [2, 4]
let doubled = map_each(nums, fn(x) x * 2 end)         // [2, 4, 6, 8, 10]
let total = reduce(nums, 0, fn(acc, x) acc + x end)    // 15
let sorted = sort_by(nums, fn(a, b) a - b end)         // ascending
```

Lambdas capture outer variables by snapshot (value copy at creation time).

## Arrays

```
// Creation
let arr = [1, 2, 3, 4, 5]
let empty = []
let mixed = [1.0, "hello", 3.14]
let generated = range_array(0, 100)

// Access
let first = arr[0]
let last = arr[len(arr) - 1]
let length = len(arr)

// Mutation (requires let mut)
let mut items = [10, 20]
push(items, 30)           // [10, 20, 30]
let popped = pop(items)   // 30
items[0] = 99             // [99, 20]

// Slicing
let sub = slice(arr, 1, 3)    // elements at index 1, 2
```

## HashMaps

```
let mut m = map()
m["name"] = "OctoFlow"
m["version"] = 1.0

let val = m["name"]
let has = map_has(m, "name")     // 1.0
let keys = map_keys(m)           // ["name", "version"]
let vals = map_values(m)
map_remove(m, "version")
```

## Strings

```
let s = "Hello, World!"
let length = len(s)
let upper = to_upper(s)
let lower = to_lower(s)
let trimmed = trim(s)
let parts = split(s, ", ")
let joined = join(parts, " - ")
let has = contains(s, "World")
let idx = index_of(s, "World")
let sub = substring(s, 0, 5)       // "Hello"
let rep = replace(s, "World", "GPU")

// Character operations
let code = ord("A")     // 65.0
let char = chr(65)       // "A"
```

## Structs and Vectors

```
struct Point(x, y)
struct Color(r, g, b)

let p = Point(3.0, 4.0)
print("{p.x}, {p.y}")

let v3 = vec3(1.0, 2.0, 3.0)
print("{v3.x}, {v3.y}, {v3.z}")
```

## Modules and Imports

```
use csv
use timeseries
use descriptive

let data = read_csv("prices.csv")
let avg = mean(data)
```

`use` imports all public functions from a module. Search path:
1. `stdlib/<name>.flow`
2. `stdlib/<domain>/<name>.flow`
3. `./<name>.flow` (relative to current file)

## Streams and Pipelines

```
stream prices = tap("input.csv")
stream processed = prices |> ema(0.1) |> scale(100) |> clamp(0, 100)
emit(processed, "output.csv")

// Pipeline functions
fn warm_filter: brightness(20) |> contrast(1.2) |> saturate(1.1)
stream result = tap("photo.jpg") |> warm_filter
emit(result, "output.png")
```

## GPU Operations

All GPU data stays in VRAM between operations. No CPU round-trips until you need the result.

```
// Creation
let a = gpu_fill(1.0, 10000000)     // 10M elements
let r = gpu_range(0, 1000, 1)       // 0, 1, 2, ..., 999

// Element-wise binary
let c = gpu_add(a, b)
let d = gpu_sub(a, b)
let e = gpu_mul(a, b)
let f = gpu_div(a, b)

// Element-wise unary
let g = gpu_scale(a, 2.0)
let h = gpu_abs(a)
let i = gpu_sqrt(a)
let j = gpu_exp(a)
let k = gpu_log(a)
let l = gpu_sin(a)
let m = gpu_cos(a)
let n = gpu_pow(a, 2.0)
let o = gpu_clamp(a, 0.0, 1.0)

// Reductions (return CPU scalar)
let total = gpu_sum(a)
let maximum = gpu_max(a)
let minimum = gpu_min(a)
let average = gpu_mean(a)
let prefix = gpu_cumsum(a)

// Conditional
let selected = gpu_where(cond, a, b)   // a where cond!=0, else b

// Matrix multiply
let result = gpu_matmul(a, b, rows_a, cols_a, cols_b)
```

## GPU Virtual Machine

The GPU VM runs autonomous dispatch chains. The CPU writes input, submits once, reads output. Everything between happens on GPU with zero CPU round-trips.

```
let vm = vm_boot()

// Write input data to register 0 of instance 0
let data = [1.0, 2.0, 3.0, 4.0]
let _w = vm_write_register(vm, 0, 0, data)

// Chain multiple dispatches — all run in one GPU submit
let _d1 = vm_dispatch(vm, "scale.spv", [0.0, 3.0, 4.0], 1.0)
let _d2 = vm_dispatch(vm, "relu.spv", [0.0, 4.0], 1.0)

// Build and execute
let prog = vm_build(vm)
vm_execute(prog)

// Read result
let result = vm_read_register(vm, 0, 0)
vm_shutdown(vm)
```

5 memory regions: Registers (per-instance I/O), Globals (shared), Heap (weights), Metrics (GPU→CPU polling), Control (CPU→GPU live commands). See [GPU Guide](gpu-guide.md) for full details.

## File I/O

Requires `--allow-read` and/or `--allow-write`.

```
let text = read_file("data.txt")
let lines = read_lines("log.txt")
let bytes = read_bytes("image.png")
write_file("output.txt", "Hello!")
append_file("log.txt", "entry")
let exists = file_exists("data.txt")
let size = file_size("data.txt")
let files = list_dir("./")
```

## CSV and JSON

```
// CSV (requires --allow-read/write)
let data = read_csv("data.csv")
write_csv("output.csv", data)

// JSON
let obj = json_parse(text)
let arr = json_parse_array(text)
let text = json_stringify(obj)
```

## HTTP Client

Requires `--allow-net`. Returns `.status`, `.body`, `.ok`, `.error`.

```
let r = http_get("https://api.example.com/data")
if r.ok == 1.0
    let data = json_parse(r.body)
    print("Status: {r.status}")
end
```

Also: `http_post(url, body)`, `http_put(url, body)`, `http_delete(url)`.

## Shell Execution

Requires `--allow-exec`. Returns `.status`, `.output`, `.ok`, `.error`.

```
let r = exec("git", "status")
if r.ok == 1.0
    print(r.output)
end
```

## Error Handling

```
let r = try(read_file("maybe.txt"))
if r.ok == 1.0
    print(r.value)
else
    print("Error: {r.error}")
end
```

`try()` catches errors and returns `.value`, `.ok` (1.0/0.0), `.error`.

## Security

Sandboxed by default. Scripts need explicit flags:

```
octoflow run script.flow                          // no I/O
octoflow run script.flow --allow-read             // file read
octoflow run script.flow --allow-read --allow-net // file read + network
```

| Flag | Grants |
|------|--------|
| `--allow-read` | File system read |
| `--allow-write` | File system write |
| `--allow-net` | Network (HTTP, TCP, UDP) |
| `--allow-exec` | Subprocess execution |
| `--allow-ffi` | Foreign function interface |

## Built-in Functions Reference

### Math
`abs(x)` `sqrt(x)` `pow(x, n)` `exp(x)` `log(x)` `sin(x)` `cos(x)` `floor(x)` `ceil(x)` `round(x)` `min(a, b)` `max(a, b)`

### String
`len(s)` `trim(s)` `to_upper(s)` `to_lower(s)` `contains(s, sub)` `starts_with(s, pre)` `ends_with(s, suf)` `index_of(s, sub)` `char_at(s, i)` `repeat(s, n)` `substr(s, start, len)` `replace(s, old, new)` `split(s, delim)` `join(arr, delim)` `ord(c)` `chr(n)` `str(val)` `float(val)` `int(val)`

### Array
`len(arr)` `first(arr)` `last(arr)` `push(arr, val)` `pop(arr)` `find(arr, val)` `reverse(arr)` `slice(arr, start, end)` `sort_array(arr)` `unique(arr)` `range_array(start, end)`

### Higher-Order
`filter(arr, fn(x) cond end)` `map_each(arr, fn(x) expr end)` `reduce(arr, init, fn(acc, x) expr end)` `sort_by(arr, fn(a, b) expr end)`

### Statistics
`mean(arr)` `median(arr)` `stddev(arr)` `variance(arr)` `quantile(arr, q)` `correlation(a, b)` `min_val(arr)` `max_val(arr)` `dot(a, b)` `norm(arr)` `normalize(arr)`

### HashMap
`map()` `map_has(m, key)` `map_get(m, key)` `map_keys(m)` `map_values(m)` `map_remove(m, key)` `len(m)`

### Type
`type_of(val)` `float(val)` `int(val)` `str(val)`

### Regex
`regex_match(text, pat)` `is_match(text, pat)` `regex_find(text, pat)` `regex_find_all(text, pat)` `regex_split(text, pat)` `regex_replace(text, pat, rep)` `capture_groups(text, pat)`

### Date/Time
`now()` `now_ms()` `time()` `timestamp(iso_str)` `format_datetime(ts, fmt)` `add_seconds(ts, n)` `add_minutes(ts, n)` `add_hours(ts, n)` `add_days(ts, n)` `diff_seconds(a, b)` `diff_hours(a, b)` `diff_days(a, b)`

### Encoding
`base64_encode(s)` `base64_decode(s)` `hex_encode(s)` `hex_decode(s)`

### Bitwise
`bit_and(a, b)` `bit_or(a, b)` `bit_xor(a, b)` `bit_test(n, bit)` `bit_shl(a, n)` `bit_shr(a, n)` `float_to_bits(f)` `bits_to_float(n)` `float_byte(f, idx)`

### System
`os_name()` `env(name)` `random()` `read_line()` `sleep(ms)` `print_raw(s)` `gpu_info()`

### File I/O
`read_file(path)` `read_lines(path)` `read_bytes(path)` `read_csv(path)` `write_file(path, text)` `append_file(path, text)` `write_csv(path, data)` `write_bytes(path, arr)` `file_exists(path)` `file_size(path)` `is_file(path)` `is_dir(path)` `list_dir(path)` `join_path(parts...)` `dirname(path)` `basename(path)` `file_ext(path)`

### JSON
`json_parse(text)` `json_parse_array(text)` `json_stringify(val)`

### HTTP
`http_get(url)` `http_post(url, body)` `http_put(url, body)` `http_delete(url)` `http_listen(port)` `http_accept(fd)` `http_respond(fd, status, body)` `http_respond_json(fd, status, json)`

### TCP/UDP
`tcp_connect(host, port)` `tcp_send(fd, data)` `tcp_recv(fd, max)` `tcp_close(fd)` `tcp_listen(port)` `tcp_accept(fd)` `udp_socket()` `udp_send_to(fd, host, port, data)` `udp_recv_from(fd, max)`

### Shell
`exec(cmd, ...args)` — returns `.status`, `.output`, `.ok`, `.error`

### Error Handling
`try(expr)` — returns `.value`, `.ok`, `.error`

### GPU
`gpu_fill(val, n)` `gpu_range(start, end, step)` `gpu_random(n)` `gpu_add(a, b)` `gpu_sub(a, b)` `gpu_mul(a, b)` `gpu_div(a, b)` `gpu_scale(a, s)` `gpu_abs(a)` `gpu_negate(a)` `gpu_sqrt(a)` `gpu_exp(a)` `gpu_log(a)` `gpu_sin(a)` `gpu_cos(a)` `gpu_floor(a)` `gpu_ceil(a)` `gpu_round(a)` `gpu_pow(a, n)` `gpu_clamp(a, lo, hi)` `gpu_reverse(a)` `gpu_sum(a)` `gpu_min(a)` `gpu_max(a)` `gpu_mean(a)` `gpu_product(a)` `gpu_variance(a)` `gpu_stddev(a)` `gpu_cumsum(a)` `gpu_where(cond, a, b)` `gpu_concat(a, b)` `gpu_gather(data, idx)` `gpu_scatter(vals, idx, size)` `gpu_ema(a, alpha)` `gpu_matmul(a, b, m, k, n)` `gpu_compute(spv_path, arr)` `gpu_run(spv, arrs..., scalars...)` `gpu_info()`

### GPU VM
`vm_boot()` `vm_shutdown(vm)` `vm_write_register(vm, inst, reg, data)` `vm_read_register(vm, inst, reg)` `vm_write_globals(vm, data)` `vm_read_globals(vm)` `vm_dispatch(vm, spv, pc, wg)` `vm_dispatch_indirect(vm, spv, pc)` `vm_build(vm)` `vm_execute(prog)` `vm_execute_async(prog)` `vm_poll(prog)` `vm_wait(prog)` `vm_poll_status(vm)` `vm_write_control_live(vm, off, val)`

### Video / Terminal
`video_open(path)` `video_frame(handle, idx)` `term_image(r, g, b, w, h)` `term_supports_graphics()` `term_clear()`

### Window
`window_open(title, w, h)` `window_close(h)` `window_alive(h)` `window_draw(h, pixels, w, h)` `window_poll(h)` `window_event_key(h)`

## Common Patterns

### Read CSV, analyze, write results
```
let data = read_csv("sales.csv")
let revenue = csv_column(data, "revenue")
let avg = mean(revenue)
let std = stddev(revenue)
print("Revenue: avg={avg}, std={std}")
write_file("report.txt", "Average revenue: " + str(avg))
```

### GPU compute pipeline
```
let prices = gpu_fill(100.0, 1000000)
let noise = gpu_scale(gpu_fill(1.0, 1000000), 0.02)
let adjusted = gpu_add(prices, noise)
let total = gpu_sum(adjusted)
print("Total: {total}")
```

### HTTP API client
```
let r = http_get("https://api.example.com/users")
if r.ok == 1.0
    let users = json_parse_array(r.body)
    for user in users
        print("{user}")
    end
end
```

### Error-safe file processing
```
let r = try(read_file("config.txt"))
if r.ok == 1.0
    let config = json_parse(r.value)
    print("Loaded config")
else
    print("Using defaults: {r.error}")
end
```

## Stdlib Domains

Import with `use <module_name>`:

| Domain | Modules |
|--------|---------|
| **ai** | transformer, inference, generate, weight_loader |
| **collections** | stack, queue, heap, graph, collections |
| **compiler** | lexer, eval, parser, preflight, codegen, ir |
| **crypto** | hash, encoding, random |
| **data** | csv, io, pipeline, transform, validate |
| **db** | core, query, schema |
| **formats** | gguf, json |
| **gpu** | VM, emitters, runtime, kernels |
| **gui** | widgets, layout, themes, events |
| **llm** | generate, stream, chat, decompose |
| **media** | image (PNG/JPEG/GIF/BMP), video (AVI/MP4/H.264), audio (WAV) |
| **ml** | regression, classify, cluster, nn, tree, ensemble, linalg, metrics, preprocess |
| **science** | calculus, constants, interpolate, matrix, physics, signal, optimize |
| **stats** | descriptive, correlation, distribution, hypothesis, risk, timeseries, math_ext |
| **string** | string, regex, format |
| **sys** | args, env, memory, platform, timer |
| **terminal** | term_image, colors |
| **time** | datetime |
| **web** | http, json_util, url |
| **core** | math, sort, array_utils, io |
