# OctoFlow LLM Context

> Paste this into any LLM system prompt to generate correct `.flow` code.

## Language Overview

OctoFlow is a GPU-native programming language. All numbers are `f32`. Strings are UTF-8. No semicolons. Indentation is conventional, not syntactic. Comments: `//` only.

## Variables

```
let x = 42              // immutable
let mut y = 0           // mutable (can reassign)
y = y + 1               // reassignment (scalar only)
```

Types are inferred: `42` is float, `"hello"` is string. No explicit type annotations.

## Arithmetic & Comparison

```
+ - * / %               // arithmetic (all f32)
== != < > <= >=          // comparison (returns 1.0 or 0.0)
&& || !                  // logical
```

**Important**: Division by zero returns `inf`, not an error.

## Strings

```
let name = "world"
let msg = "hello " + name     // concatenation with +
let c = chr(27)                // character from code point
let n = ord("A")               // code point from character (65.0)
```

**CRITICAL**: OctoFlow has NO string escape sequences.
- `"\n"` is the literal two characters `\` and `n`, NOT a newline
- Use `chr(10)` for newline, `chr(9)` for tab, `chr(27)` for ESC
- `chr(34)` for double-quote inside strings is impossible (quote terminates string)

## Print

```
print("hello world")           // literal string only
print("x = {x}, y = {y}")     // {var} interpolation
let msg = "result: " + str(n)
print("{msg}")                 // variable interpolation
```

**CRITICAL**: `print()` ONLY accepts a string literal argument.
- `print(x)` → ERROR (variable, not literal)
- `print("a" + "b")` → ERROR (expression, not literal)
- `print(str(x))` → ERROR (function call, not literal)
- Correct pattern: `let tmp = "a" + str(x); print("{tmp}")`

## Control Flow

```
if x > 10
    print("big")
elif x > 5
    print("medium")
else
    print("small")
end

while x < 100
    x = x + 1
end

for i in range(0, 10)          // i goes 0, 1, 2, ..., 9
    print("{i}")
end

for item in my_array            // iterate array elements
    print("{item}")
end

break                          // exit loop
continue                       // next iteration
```

All blocks end with `end`. No braces.

## Functions

### Scalar Functions (return a value)
```
fn add(a, b)
    return a + b
end

let result = add(3, 4)
```

### Pipeline Functions (stream processing)
```
fn double(x):
    multiply(2.0)
```

### Lambda (inline anonymous functions)
```
let doubled = map_each(arr, fn(x) x * 2 end)
let evens = filter(arr, fn(x) x % 2 == 0 end)
```

## Arrays

```
let arr = [1, 2, 3, "mixed"]   // heterogeneous (float + string)
let first = arr[0]              // index access
let n = len(arr)                // length

let mut list = [1, 2, 3]
push(list, 4)                   // append
let last = pop(list)            // remove last
```

### Array Functions (LetDecl level — use with `let`)
```
let lines = read_lines("file.txt")        // → string array
let entries = list_dir("./")               // → string array
let parts = split("a,b,c", ",")           // → ["a", "b", "c"]
let items = regex_split("a1b2c", "[0-9]") // → ["a", "b", "c"]
let groups = capture_groups(text, pattern) // → string array
let matches = regex_find_all(text, pat)    // → string array
let rows = read_csv("data.csv")           // → array of maps
let data = json_parse_array(json_str)     // → array of values
let bytes = read_bytes("file.bin")        // → byte array (f32)
let rev = reverse(arr)                     // → reversed copy
let sub = slice(arr, 2, 5)                // → arr[2..5]
let sorted = sort_array(arr)              // → sorted copy
let uniq = unique(arr)                     // → deduplicated
let nums = range_array(0, 10)             // → [0,1,...,9]
```

### Higher-Order Array Functions
```
let evens = filter(arr, fn(x) x % 2 == 0 end)
let doubled = map_each(arr, fn(x) x * 2 end)
let ordered = sort_by(arr, fn(a, b) a - b end)
let total = reduce(arr, 0, fn(acc, x) acc + x end)
```

## Maps (Hash Maps)

```
let mut m = map()               // create empty map
map_set(m, "key", "value")      // set key
let v = map_get(m, "key")       // get value
let exists = map_has(m, "key")  // 1.0 if exists
map_remove(m, "key")            // delete key
let keys = map_keys(m)          // → array of keys
let vals = map_values(m)        // → array of values
let n = map_size(m)             // number of entries
```

### JSON
```
let obj = json_parse(json_string)          // → map
let arr = json_parse_array(json_string)    // → array
let s = json_stringify(map_name)           // → JSON string
```

## GPU Arrays

GPU arrays live in VRAM. Operations are parallel across all elements. Data only moves to CPU when you access individual elements, print, or write to file.

**CRITICAL**: GPU functions must appear on the right side of a `let` declaration. Never nest GPU calls. Never use GPU calls in assignment RHS.

```
// CORRECT:
let a = gpu_fill(1.0, 10000)
let b = gpu_add(a, a)

// WRONG — nested GPU call:
let c = gpu_add(gpu_fill(1.0, N), a)    // ERROR

// WRONG — GPU call in assignment:
a = gpu_add(a, b)                        // ERROR

// CORRECT — re-bind with let:
let a = gpu_add(a, b)                    // overwrites previous 'a'
```

### Create
```
let a = gpu_fill(val, n)         // n elements, all = val
let b = gpu_range(start, end, step) // arithmetic sequence
let c = gpu_random(n)            // n random values [0,1)
```

### Element-wise Binary (two GPU arrays → GPU array)
```
let c = gpu_add(a, b)           // a + b
let c = gpu_sub(a, b)           // a - b
let c = gpu_mul(a, b)           // a * b
let c = gpu_div(a, b)           // a / b
```

### Element-wise Unary/Scalar (GPU array → GPU array)
```
let b = gpu_scale(a, 2.0)       // a * scalar
let b = gpu_abs(a)               // |a|
let b = gpu_negate(a)            // -a
let b = gpu_sqrt(a)              // sqrt(a)
let b = gpu_exp(a)               // e^a
let b = gpu_log(a)               // ln(a)
let b = gpu_sin(a)               // sin(a)
let b = gpu_cos(a)               // cos(a)
let b = gpu_pow(a, n)            // a^n
let b = gpu_floor(a)             // floor
let b = gpu_ceil(a)              // ceil
let b = gpu_round(a)             // round
let b = gpu_clamp(a, lo, hi)    // clamp to [lo, hi]
let b = gpu_reverse(a)           // reverse order
let b = gpu_cumsum(a)            // cumulative sum
let b = gpu_ema(a, alpha)        // exponential moving average
```

### Data Movement
```
let c = gpu_concat(a, b)            // concatenate two arrays
let g = gpu_gather(data, indices)    // indexed lookup: g[i] = data[indices[i]]
let s = gpu_scatter(vals, idx, size) // scatter: result[idx[i]] = vals[i]
```

### Reductions (GPU array → scalar)
```
let s = gpu_sum(a)               // sum of all elements
let m = gpu_min(a)               // minimum
let m = gpu_max(a)               // maximum
let m = gpu_mean(a)              // arithmetic mean
let p = gpu_product(a)           // product of all elements
let v = gpu_variance(a)          // population variance
let d = gpu_stddev(a)            // population standard deviation
let n = norm(a)                  // L2 norm
```

### Conditional
```
let c = gpu_where(cond, a, b)   // select a where cond≠0, else b
```

### Matrix
```
let c = gpu_matmul(a, b, m, n, k)  // A is m×k, B is k×n, result is m×n
let t = mat_transpose(a)            // transpose
let n = normalize(a)                 // L2 normalize
```

### GPU Array Access (triggers download to CPU)
```
let val = a[0]                   // single element
let s = gpu_sum(a)               // reduction → scalar
print("{a}")                     // prints all elements (downloads)
```

## Structs

```
struct Point { x, y }
let p = Point(3.0, 4.0)
let px = p.x                    // field access via decomposed scalars
```

Structs decompose to `name.field` scalars. Constructor is the struct name as a function.

## Vectors (Built-in)

```
let v = vec2(1.0, 2.0)          // decomposed to v.x, v.y
let w = vec3(1.0, 2.0, 3.0)    // v.x, v.y, v.z
let q = vec4(1.0, 2.0, 3.0, 4.0) // v.x, v.y, v.z, v.w
```

## Modules (Standard Library)

```
use collections.heap             // import module
use stats.descriptive

let mut h = heap_create()
heap_push(h, 5)
```

Available via `tap("path/to/module.flow")` for relative imports.

## Error Handling

```
let result = try(some_function(x))
if result.ok == 1.0
    print("value: {result.value}")
else
    print("error: {result.error}")
end
```

## File I/O

```
// Reading (requires --allow-read)
let text = read_file("path.txt")        // → string
let lines = read_lines("path.txt")      // → string array
let rows = read_csv("data.csv")         // → array of maps
let bytes = read_bytes("file.bin")      // → byte array

// Writing (requires --allow-write)
write_file("out.txt", content)           // write string
write_csv("out.csv", data_array)         // write array of maps
write_bytes("out.bin", byte_array)       // write byte array
```

## Network (requires --allow-net)

```
let resp = http_get(url)
// resp.status, resp.body, resp.ok, resp.error

let resp = http_post(url, body_string)
let resp = http_put(url, body_string)
let resp = http_delete(url)
```

## Time & System

```
let t = now_ms()                 // milliseconds since epoch
let t = time()                   // formatted time string
sleep(1000)                      // sleep 1 second
let name = os_name()             // "windows", "linux", "macos"
let val = env("HOME")            // environment variable
```

## Scalar Builtin Reference

### Math (0 args)
`random()` → random float [0,1)

### Math (1 arg)
`abs(x)` `sqrt(x)` `exp(x)` `log(x)` `sin(x)` `cos(x)`
`floor(x)` `ceil(x)` `round(x)` `int(x)` `float(x)`

### Math (2 args)
`pow(base, exp)` `clamp(x, lo, hi)`

### String (1 arg)
`str(x)` `trim(s)` `to_upper(s)` `to_lower(s)` `len(s)` `ord(c)` `chr(n)`

### String (2 args)
`contains(s, sub)` `starts_with(s, pre)` `ends_with(s, suf)`
`index_of(s, sub)` `char_at(s, idx)` `repeat(s, n)`

### String (3 args)
`substr(s, start, len)` `replace(s, old, new)`

### Regex (2 args)
`regex_match(text, pat)` `is_match(text, pat)` `regex_find(text, pat)`

### Regex (3 args)
`regex_replace(text, pat, rep)`

### Type
`type_of(x)` → "float", "string", or "map"

### Statistics
`mean(arr)` `median(arr)` `stddev(arr)` `variance(arr)`
`quantile(arr, q)` `correlation(arr1, arr2)`

### Date/Time
`timestamp(s)` `timestamp_from_unix(n)` `format_datetime(ts, fmt)`
`add_seconds(ts, n)` `add_minutes(ts, n)` `add_hours(ts, n)` `add_days(ts, n)`
`diff_seconds(ts1, ts2)` `diff_days(ts1, ts2)` `diff_hours(ts1, ts2)`

### Crypto/Encoding
`base64_encode(s)` `base64_decode(s)` `hex_encode(s)` `hex_decode(s)`

### File System
`dirname(path)` `basename(path)` `canonicalize_path(path)`
`is_file(path)` `is_dir(path)` `is_symlink(path)`

### Low-level
`float_to_bits(f)` `bits_to_float(n)` `float_byte(f, idx)`
`mem_alloc(size)` `mem_free(ptr)` `mem_size(ptr)` `mem_from_str(s)` `mem_to_str(ptr, len)`
`mem_get_u8(ptr, off)` `mem_get_u32(ptr, off)` `mem_get_f32(ptr, off)` `mem_get_u64(ptr, off)` `mem_get_ptr(ptr, off)`
`mem_set_u8(ptr, off, val)` `mem_set_u32(ptr, off, val)` `mem_set_f32(ptr, off, val)` `mem_set_u64(ptr, off, val)` `mem_set_ptr(ptr, off, val)`
`mem_copy(dst, dst_off, src, src_off, len)` `bit_and(a, b)` `bit_or(a, b)` `bit_test(val, bit)`

## Stream Pipelines

For data processing workflows (ETL-style):

```
let data = [1.0, 2.0, 3.0] |> multiply(2.0) |> add(10.0) |> emit("output.csv")
```

Pipeline operations: `add` `subtract` `multiply` `divide` `negate` `abs` `sqrt` `exp` `log` `sin` `cos` `pow` `clamp` `scale` `offset` `normalize` `floor` `ceil` `round`

## Common Patterns

### GPU computation → file output
```
let data = gpu_range(1.0, 1001, 1.0)
let result = gpu_mul(data, data)
let s = gpu_sum(result)
print("sum of squares: {s}")
```

### String building (no escape sequences)
```
let nl = chr(10)                    // newline character
let tab = chr(9)                    // tab character
let esc = chr(27)                   // escape character
let line = "col1" + tab + "col2" + nl
```

### Mutable GPU arrays in loops
```
let mut arr = gpu_fill(0.0, N)
for i in range(0, 10)
    let arr = gpu_add(arr, delta)   // re-bind with let (NOT assignment)
end
```

### ANSI terminal colors
```
let esc = chr(27)
let red = esc + "[31m"
let reset = esc + "[0m"
let colored = red + "error" + reset
print("{colored}")
```

## Security Flags

Programs run sandboxed by default. Enable capabilities with flags:
- `--allow-read` — file reading
- `--allow-write` — file writing
- `--allow-net` — network access
- `--allow-exec` — subprocess execution
- `--allow-all` — all permissions

## Running

```bash
octoflow run program.flow                    # run a program
octoflow run program.flow --allow-write      # with file write permission
octoflow repl                                # interactive REPL
```
