# OctoFlow — Coding Guide

**Version:** 1.28 (Phase 128)
**Date:** February 21, 2026
**Purpose:** Complete reference for writing OctoFlow code — from primitives to advanced patterns
**Audience:** Developers, LLMs, future compiler implementers

---

## Table of Contents

### Part I: Primitives (What You Have)
1. Values and Types
2. Variables and Assignment
3. Operators
4. Control Flow
5. Functions
6. Data Structures

### Part II: Standard Library (Native Functions)
7. Math and Statistics
8. String Operations
9. Array Operations
10. File I/O
11. Network Operations
12. Date/Time
13. Encoding

### Part III: Advanced (Modules and Patterns)
14. Modules and Imports
15. Error Handling
16. Closures and Higher-Order Functions
17. GPU Acceleration (Automatic)
18. Security Model

### Part IV: Domain Foundations
19. Data Science and Analytics
20. Finance and Quantitative
21. DevOps and Automation
22. Systems and Infrastructure
23. Web and Networked Applications
24. GUI Toolkit
25. AI and LLM Inference

### Part V: Language Design
26. Syntax Philosophy
27. Type System
28. Memory Model
29. Execution Model

---

# PART I: PRIMITIVES

## 1. Values and Types

### 1.1 The Three Value Types

OctoFlow has exactly three value types:

```flow
let num = 42.0              // Value::Float (f32)
let text = "hello"          // Value::Str (String)
let data = map()            // Value::Map (HashMap<String, Value>)
```

**All numbers are floats.** There is no integer type at runtime (though `int()` converts to whole numbers).

**Strings are immutable.** String operations return new strings.

**Maps are mutable.** Use `map_set()`, `map_get()`, `map_remove()`.

### 1.2 Type Checking

```flow
let x = type_of(42.0)        // "float"
let y = type_of("hello")     // "string"
let z = type_of(map())       // "map"
```

### 1.3 Type Conversion

```flow
// To string
let s1 = str(42.5)           // "42.5"
let s2 = str(map())          // "{key: value, ...}"

// To float
let f1 = float("42.5")       // 42.5
let f2 = float("invalid")    // Error: not a valid number

// To integer (truncate)
let i = int(42.9)            // 42.0 (truncated, still a float)
```

---

## 2. Variables and Assignment

### 2.1 Immutable by Default

```flow
let x = 10.0
x = 20.0                     // ERROR: x is immutable
```

### 2.2 Mutable Variables

```flow
let mut count = 0.0
count = count + 1.0          // OK: declared with mut
count = 100.0                // OK
```

### 2.3 Scope Rules

Variables are scoped to:
- Functions
- Loop bodies
- If/elif/else blocks

```flow
let outer = 1.0

if true
    let inner = 2.0
    let check = outer + inner  // OK: can see outer
end

let fail = inner              // ERROR: inner not in scope
```

---

## 3. Operators

### 3.1 Arithmetic

```flow
let sum = a + b              // Addition
let diff = a - b             // Subtraction
let prod = a * b             // Multiplication
let quot = a / b             // Division
let remainder = a % b        // Modulo (NOT YET IMPLEMENTED in scalar expr)
```

**Operator precedence:** `*` `/` (highest) → `+` `-` (lowest)

### 3.2 Comparison

```flow
let gt = a > b               // Greater than → 1.0 or 0.0
let lt = a < b               // Less than
let gte = a >= b             // Greater or equal
let lte = a <= b             // Less or equal
let eq = a == b              // Equal
let neq = a != b             // Not equal
```

**All comparisons return floats:** 1.0 (true) or 0.0 (false)

### 3.3 Boolean Logic

```flow
let both = (a > 0.0) && (b > 0.0)    // AND
let either = (a > 0.0) || (b > 0.0)  // OR
```

**Use parentheses** for clarity. `&&` and `||` work on float values (0.0 = false, non-zero = true).

### 3.4 Bitwise Operators (Phase 43) ✅

```flow
let flags = FLAG_A | FLAG_B          // Bitwise OR
let masked = value & 0xFF            // Bitwise AND
let shifted = opcode << 16           // Left shift
let extracted = (word >> 8) & 0xFF   // Right shift
let toggled = value ^ 0x01           // XOR

// Hex literals supported
let color = 0xFF8800AA
let red = (color >> 24) & 0xFF
```

**Use cases:** Flag manipulation, SPIR-V byte building, encoding algorithms, bit packing

---

## 4. Control Flow

### 4.1 If Expression (Returns a Value)

```flow
let max = if a > b then a else b

let category = if score >= 90.0 then 3.0
               elif score >= 70.0 then 2.0
               else 1.0
```

**Must have `else` branch** when used in `let` (every path must return a value).

### 4.2 If Statement Block (Multiple Statements)

```flow
if score >= 90.0
    print("Excellent")
    let mut grade = 4.0
elif score >= 70.0
    print("Good")
    let mut grade = 3.0
else
    print("Needs improvement")
    let mut grade = 1.0
end
```

**Use `end` to close the block.** Elif/else are optional.

### 4.3 While Loops

```flow
let mut i = 10.0
while i > 0.0
    print("{i}")
    i = i - 1.0
end
```

**Safety limit:** 10,000 iterations max (prevents infinite loops).

### 4.4 For Loops (Range)

```flow
for i in range(0, 10)
    print("{i}")
end
```

**Range is exclusive:** `range(0, 10)` = 0, 1, 2, ..., 9 (not 10)

### 4.5 For-Each Loops (Array)

```flow
let numbers = [1.0, 2.0, 3.0]
for n in numbers
    print("{n}")
end
```

**Works on heterogeneous arrays** (mix of floats and strings).

### 4.6 Break and Continue

```flow
for i in range(0, 100)
    if i > 10.0
        break              // Exit loop
    end
    if i % 2.0 == 0.0
        continue           // Skip to next iteration
    end
    print("{i}")
end
```

**Only valid inside loops.** Compile error if used outside.

---

## 5. Functions

### 5.1 Function Definition

```flow
fn double(x)
    return x * 2.0
end

fn add_multiply(a, b, factor)
    let sum = a + b
    return sum * factor
end
```

**Must have `return` statement.** Functions cannot fall through.

### 5.2 Function Calls

```flow
let result = double(21.0)
let complex = add_multiply(10.0, 5.0, 2.0)

// Bare function call (expression statement) — result discarded
emit_word(42.0)
process_item(x)
```

### 5.3 Closures and Lambdas

```flow
let doubled = map_each([1.0, 2.0, 3.0], fn(x) x * 2.0 end)

let filtered = filter([1.0, -2.0, 3.0, -4.0], fn(x) x > 0.0 end)

let sorted = sort_by([3.0, 1.0, 2.0], fn(x) -x end)  // Descending

let total = reduce([1.0, 2.0, 3.0], 0.0, fn(acc, x) acc + x end)
```

**Lambda syntax:** `fn(params) expression end`

**Capture:** Lambdas capture outer variables by snapshot (value at definition time).

---

## 6. Data Structures

### 6.1 Arrays

```flow
let numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
let mixed = [1.0, "text", 3.0]         // Heterogeneous (Float or Str)

let first = numbers[0.0]               // 1.0 (index is float, truncated to usize)
let length = len(numbers)              // 5.0

// Array mutation (requires mutable binding)
let mut arr = [1.0, 2.0]
arr[0.0] = 10.0
push(arr, 3.0)                         // arr = [10.0, 2.0, 3.0]
let last = pop(arr)                    // arr = [10.0, 2.0], last = 3.0
```

### 6.2 HashMaps

```flow
let mut person = map()
map_set(person, "name", "Alice")
map_set(person, "age", 30.0)

let name = map_get(person, "name")     // "Alice"
let has_age = map_has(person, "age")   // 1.0 (true)
let keys = map_keys(person)            // ["name", "age"] (array of keys)

map_remove(person, "age")

// Bracket access
let name2 = person["name"]             // "Alice"
```

### 6.3 Structs

```flow
struct Point(x, y)
struct Person(name, age, city)

let p = Point(10.0, 20.0)
let alice = Person("Alice", 30.0, "NYC")

print("{p.x}, {p.y}")                  // 10, 20
print("{alice.name} lives in {alice.city}")  // Alice lives in NYC
```

**Structs are scalar-decomposed:** `Point(10, 20)` creates `p.x = 10.0`, `p.y = 20.0` as separate scalars.

### 6.4 Vectors (Fixed-Size)

```flow
let pos = vec3(1.0, 2.0, 3.0)          // pos.x, pos.y, pos.z
let color = vec4(1.0, 0.0, 0.0, 1.0)   // color.x, .y, .z, .w
let uv = vec2(0.5, 0.5)                // uv.x, uv.y

let dist = sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z)
```

---

# PART II: STANDARD LIBRARY

## 7. Math and Statistics

### 7.1 Scalar Math Functions

```flow
abs(x)                       // Absolute value
sqrt(x)                      // Square root
pow(x, y)                    // Power (x^y)
exp(x)                       // e^x
log(x)                       // Natural log
sin(x), cos(x)               // Trigonometry
floor(x), ceil(x), round(x)  // Rounding
clamp(x, min, max)           // Constrain to range
```

### 7.2 Statistics on Arrays (Phase 41)

```flow
let data = [1.0, 2.0, 3.0, 4.0, 5.0]

mean(data)                   // Arithmetic mean: 3.0
median(data)                 // Middle value: 3.0
stddev(data)                 // Standard deviation: ~1.58
variance(data)               // Variance: ~2.5
quantile(data, 0.95)         // 95th percentile
correlation(arr1, arr2)      // Pearson correlation coefficient
```

### 7.3 Array Aggregates

```flow
sum(arr)                     // Sum all elements
min_val(arr), max_val(arr)   // Min/max of array
len(arr)                     // Array length
```

### 7.4 Random Numbers

```flow
let r = random()             // Random float in [0.0, 1.0)

// Set seed for reproducibility
// Command line: flowgpu-cli run program.flow --set seed=42
```

---

## 8. String Operations

### 8.1 String Basics

```flow
let s1 = "hello"
let s2 = "world"
let combined = s1 + " " + s2         // "hello world"

let length = len(s1)                 // 5.0
let has = contains(s1, "ell")        // 1.0 (true)
```

### 8.2 String Manipulation (Phase 32)

```flow
substr(s, start, length)             // Extract substring
replace(s, old, new)                 // Replace all occurrences
trim(s)                              // Remove whitespace
to_upper(s), to_lower(s)             // Case conversion
starts_with(s, prefix)               // Check prefix
ends_with(s, suffix)                 // Check suffix
index_of(s, needle)                  // Find position (-1 if not found)
char_at(s, index)                    // Get character at position
repeat(s, count)                     // Repeat string n times
split(s, delimiter)                  // Split to array
```

### 8.3 String Formatting

```flow
print("Value: {x}")                  // Basic interpolation
print("Precise: {x:.2}")             // 2 decimal places
print("Escaped: {{not_a_var}}")      // Literal braces
```

### 8.4 Regex Operations (Phase 43) ✅

```flow
// Pattern matching
let has_error = regex_match(log_line, "ERROR")     // 1.0 or 0.0
let is_valid = is_match(email, "@")                // Alias for regex_match

// Find first match
let found = regex_find(text, "pattern")            // Returns match or ""

// Replace all matches
let cleaned = regex_replace(phone, "[^0-9]", "")   // Remove non-digits

// Split by pattern
let parts = regex_split("one,two,three", ",")      // Returns array

// Capture groups
let groups = capture_groups(text, "Name: (.*), Age: (.*)")  // Returns array of captures
```

**Limitations (Phase 43):**
- String literals don't support backslash escapes (`\d`, `\s`, etc.)
- Use literal patterns: `"ERROR"`, `"[0-9]"`, or define patterns as variables
- Full regex escapes in Phase 44 (when string escape sequences added)

**Use cases:** Log parsing, data validation, text extraction, URL parsing

---

### 8.5 Extern FFI (Phase 44) ✅

Call native shared library functions directly from OctoFlow.

```flow
// Declare foreign functions
extern "msvcrt" {
    fn ceil(x: f64) -> f64
    fn floor(x: f64) -> f64
    fn pow(base: f64, exp: f64) -> f64
}

// Use them like normal functions
let value = -3.7
let ceiled = ceil(value)
let floored = floor(value)
let p = pow(2.0, 10.0)
print("ceil={ceiled} floor={floored} 2^10={p}")
```

**Syntax:**
```
extern "<library-name>" {
    fn <name>(<param>: <type>, ...) -> <return-type>
    fn <name>(<param>: <type>)         // void return — omit -> type
}
```

**Supported types:**

| Type | Description |
|------|-------------|
| `f32` | 32-bit float |
| `f64` | 64-bit float |
| `i32` | 32-bit integer |
| `i64` | 64-bit integer |
| `u32` | 32-bit unsigned |
| `u64` | 64-bit unsigned |
| `ptr` | Opaque pointer (strings pass as `*const u8`) |

**Library names:**
- Windows: `"msvcrt"` (C runtime), `"kernel32"`, `"user32"`, full path
- Linux: `"c"` (libc), `"m"` (libm), `"dl"`, full path

**Security:** Requires `--allow-ffi` flag:
```
flowgpu --allow-ffi myscript.flow
```

Without `--allow-ffi`, any extern fn call raises a security error.

**No allocations:** OctoFlow does not manage foreign memory. Pass strings as `ptr` (read-only, null-terminated). Do not free memory returned by foreign functions through OctoFlow — handle cleanup in the native library.

**Use cases:** OS APIs, math libraries, Vulkan extensions, hardware interfaces, calling compiled C/C++ modules

---

### 8.6 TCP/UDP Sockets (Phase 45) ✅

Raw OS socket calls — no crate, no abstraction layer.

```flow
// TCP client
let fd = tcp_connect("127.0.0.1", 8080)
if fd > 0.0
    let sent  = tcp_send(fd, "hello")
    let reply = tcp_recv(fd, 4096)
    print("reply: {reply}")
    let r = tcp_close(fd)
end

// TCP server (requires --allow-net)
let srv = tcp_listen(8080)
if srv > 0.0
    let client = tcp_accept(srv)
    let data   = tcp_recv(client, 4096)
    let sent   = tcp_send(client, "pong")
    let r1 = tcp_close(client)
    let r2 = tcp_close(srv)
end

// UDP
let sock = udp_socket()
let sent = udp_send_to(sock, "127.0.0.1", 9999, "ping")
let msg  = udp_recv_from(sock, 1024)
let r    = tcp_close(sock)
```

**Socket functions:**

| Function | Args | Returns | Notes |
|----------|------|---------|-------|
| `tcp_connect(host, port)` | string, number | fd or -1 | requires `--allow-net` |
| `tcp_send(fd, data)` | number, string | bytes sent or -1 | |
| `tcp_recv(fd, max)` | number, number | string | "" on error |
| `tcp_close(fd)` | number | 0.0 | closes any socket type |
| `tcp_listen(port)` | number | fd or -1 | requires `--allow-net` |
| `tcp_accept(fd)` | number | client fd or -1 | |
| `udp_socket()` | — | fd or -1 | requires `--allow-net` |
| `udp_send_to(fd, host, port, data)` | | bytes or -1 | |
| `udp_recv_from(fd, max)` | number, number | string | blocks until data arrives |

**Security:** `tcp_connect`, `tcp_listen`, `tcp_accept`, `udp_socket`, `udp_send_to`, `udp_recv_from` all require `--allow-net`. `tcp_close` does not (just closes an fd).

**Use cases:** Custom protocols, IoT messaging, local IPC, building an HTTP layer

---

### 8.7 HTTP Server (Phase 46) ✅

Full HTTP/1.1 server built on top of TCP sockets — zero external dependencies.

```flow
-- Start server
let srv = http_listen(8080)      // bind + listen; returns server fd

-- Accept loop
let fd = http_accept(srv)        // blocks until request, returns client fd

-- Inspect request
let method = http_method(fd)     // "GET", "POST", "PUT", "DELETE", …
let path   = http_path(fd)       // "/api/hello"
let query  = http_query(fd)      // "key=value&foo=bar" (raw query string)
let body   = http_body(fd)       // request body as string
let ct     = http_header(fd, "content-type")   // any header by name (lowercase)

-- Send response
http_respond(fd, 200.0, "Hello!")               // plain text
http_respond_json(fd, 200.0, "{\"ok\":true}")   // sets Content-Type: application/json
```

**Routing example:**

```flow
let srv = http_listen(8080)
loop
  let fd   = http_accept(srv)
  let path = http_path(fd)
  if path == "/hello" then http_respond(fd, 200.0, "Hello!")
  if path == "/ping"  then http_respond_json(fd, 200.0, "{\"pong\":true}")
  if path != "/hello" then
    if path != "/ping" then
      http_respond(fd, 404.0, "Not Found")
```

**Requires:** `--allow-net` flag.

**Security:** `http_listen`, `http_accept`, `http_respond`, and `http_respond_json` all require `--allow-net`. Requests are size-limited to 64 KB headers + `Content-Length` body.

---

### 8.8 Image I/O (Phase 48) ✅

Pure PNG, JPEG, GIF, and AVI/MJPEG codecs — zero external dependencies, no `image` crate.

```flow
-- Read an image (PNG or JPEG, detected by extension)
let img = read_image("photo.jpg")    // → img.pixels, img.width, img.height
let w   = img.width                  // float
let h   = img.height                 // float
-- img.pixels = flat RGBA array [r,g,b,a, r,g,b,a, ...], values 0..255

-- Write an image (PNG or JPEG by extension)
write_image("out.png", img.pixels, img.width, img.height)
write_image("out.jpg", img.pixels, img.width, img.height)
```

**Pixel manipulation example:**

```flow
-- Invert all RGB channels (keep alpha)
let img = read_image("input.png")
let n   = len(img.pixels)
let i   = 0.0
let out = []
loop while i < n
  let ch = i % 4.0
  if ch < 3.0
    push(out, 255.0 - img.pixels[i])
  else
    push(out, img.pixels[i])   -- alpha unchanged
  i = i + 1.0
write_image("inverted.png", out, img.width, img.height)
```

**Limits:** 100 MB max file, 16384×16384 max dimension. PNG encode uses store blocks (valid but uncompressed); JPEG encode uses standard quantization tables (quality ~75). **Requires:** `--allow-read` / `--allow-write`.

**JPEG chroma subsampling (Phase 83e):** Full support for 4:2:0, 4:2:2, and 4:4:4 chroma subsampling. Per-component buffers at component resolution with proper upsampling. DRI (Define Restart Interval) marker parsing and RST markers 0xD0-0xD7.

**GIF decoder (Phase 83e):** ~250 lines pure Rust. GIF87a/GIF89a, LZW decompression (variable-width codes up to 12 bits), frame compositing with disposal methods (0-3), interlace support, transparency via Graphics Control Extension.

**AVI/MJPEG parser (Phase 83e):** ~150 lines pure Rust. RIFF chunk walker, AVI header parsing, frame offset extraction. Each frame decoded via existing JPEG decoder.

**Video builtins (Phase 83e):**

```flow
-- Open a video (GIF or AVI/MJPEG) from a byte array
let vid = video_open(byte_array)    // → vid.width, vid.height, vid.frames, vid.fps
let w      = vid.width              // float
let h      = vid.height             // float
let nframe = vid.frames             // total frame count
let fps    = vid.fps                // frames per second

-- Extract a single frame as GPU arrays (R, G, B channels)
let f = video_frame(vid_handle, 0.0)  // → f.r, f.g, f.b (GPU arrays)
-- GIF: eager decode (all frames pre-split into R,G,B at open time)
-- AVI: lazy decode (JPEG decoded on demand per frame)
```

**Requires:** `--allow-read` / `--allow-net` (if downloading). Auto-detects GIF vs AVI format.

---

### 8.9 Regex Engine (Phase 47) ✅

Pure NFA bytecode backtracking regex engine — zero external dependencies, no `regex` crate.

```flow
-- Basic matching
let matched = regex_match("hello world", "\\w+")   // true/false
let found   = regex_find("hello 42", "\\d+")        // "42" or ""
let replaced = regex_replace("foo bar foo", "foo", "baz")  // "baz bar baz"

-- Split by pattern
let parts = regex_split("a,b;;c", "[,;]+")           // ["a","b","c"]

-- Capture groups
let caps = capture_groups("2026-02-17", "(\\d{4})-(\\d{2})-(\\d{2})")
-- caps = ["2026", "02", "17"]  (group 0..n-1, no full-match prefix)
```

**Supported syntax:** `.` `*` `+` `?` `{n,m}` `|` `()` `[]` `[^]` `^` `$` `\d` `\w` `\s` `\D` `\W` `\S` `\b`. Lazy quantifiers: `*?` `+?` `??`. Greedy by default. No lookahead/lookbehind.

---

## 9. Array Operations

### 9.1 Array Construction

```flow
let arr = [1.0, 2.0, 3.0]            // Literal
let empty = []                       // Empty array
```

### 9.2 Array Access

```flow
let first = arr[0.0]                 // Index (float, truncated to usize)
let length = len(arr)                // Length as float
```

### 9.3 Array Manipulation (Phase 33)

```flow
join(arr, separator)                 // Join to string: "1,2,3"
find(arr, value)                     // Find first matching element
first(arr), last(arr)                // First/last element
reverse(arr)                         // Reverse order
slice(arr, start, end)               // Extract subarray
sort_array(arr)                      // Sort ascending
unique(arr)                          // Remove duplicates
```

### 9.4 Array Mutation

```flow
let mut arr = [1.0, 2.0]
push(arr, 3.0)                       // arr = [1.0, 2.0, 3.0]
let last = pop(arr)                  // arr = [1.0, 2.0], last = 3.0
arr[0.0] = 10.0                      // arr = [10.0, 2.0]
```

**Requires `let mut`** for array variable.

### 9.5 Higher-Order Array Functions (Phase 38)

```flow
let doubled = map_each([1.0, 2.0, 3.0], fn(x) x * 2.0 end)
// [2.0, 4.0, 6.0]

let positives = filter([1.0, -2.0, 3.0], fn(x) x > 0.0 end)
// [1.0, 3.0]

let sorted_by_abs = sort_by([3.0, -1.0, 2.0], fn(x) abs(x) end)
// [-1.0, 2.0, 3.0]

let total = reduce([1.0, 2.0, 3.0], 0.0, fn(acc, x) acc + x end)
// 6.0
```

---

## 10. File I/O

### 10.1 Security Model

**All file operations require `--allow-read` or `--allow-write`:**

```bash
flowgpu-cli run program.flow --allow-read --allow-write
```

Without permission:
```flow
let content = read_file("data.txt")  // ERROR: file read not permitted
```

### 10.2 Reading Files (Phase 31)

```flow
let content = read_file("data.txt")          // Entire file as string

let lines = read_lines("data.txt")           // Array of lines (strings)

let files = list_dir("/path/to/directory")   // Array of filenames
```

### 10.3 Writing Files (Phase 31)

```flow
write_file("output.txt", "content")          // Overwrite file

append_file("log.txt", "new line\n")         // Append to file
```

### 10.4 Path Operations (Phase 41)

```flow
let path = join_path("/var", "log", "app.log")   // Safe path joining
let dir = dirname(path)                           // Parent directory
let file = basename(path)                         // Filename only

let canonical = canonicalize_path("./data.txt")   // Resolve . and .. safely

let exists = file_exists("config.json")           // Check existence (1.0 or 0.0)
let is_f = is_file("data.txt")                    // Check if regular file
let is_d = is_dir("folder")                       // Check if directory
let is_link = is_symlink("link")                  // Check if symlink
```

**Security:** `canonicalize_path()` prevents path traversal attacks. `join_path()` validates components.

### 10.5 CSV Files (Phase 39)

```flow
let data = read_csv("sales.csv")     // Array of maps (one per row)
// data = [{name: "Alice", amount: 100.0}, {name: "Bob", amount: 150.0}, ...]

for row in data
    let name = row["name"]
    let amount = row["amount"]
    print("{name}: {amount}")
end

write_csv("output.csv", data)        // Write array of maps to CSV
```

**Headers:** First row is headers, maps use header names as keys.

**Type detection:** Parseable as float → Value::Float, otherwise → Value::Str

### 10.6 JSON Files (Phase 36)

```flow
let obj = json_parse("{\"name\":\"Alice\",\"age\":30}")
// Returns Map with dot-notation for nesting

let name = obj["name"]               // "Alice"
let age = obj["age"]                 // 30.0

let arr = json_parse_array("[1,2,3]")  // [1.0, 2.0, 3.0]

let json_str = json_stringify(obj)     // Serialize back to JSON
```

**Nested objects:** `{"user":{"name":"Alice"}}` → key `"user.name"` = `"Alice"`

### 10.7 OctoData Files (Phase 37)

```flow
let mut config = load_data("config.od")   // Parse .od file → Map
map_set(config, "version", 2.0)
save_data("config.od", config)            // Save Map → .od file
```

**.od format:** Pure-data subset of .flow (only let declarations allowed).

---

## 11. Network Operations

### 11.1 HTTP Client (Phase 35)

**Requires `--allow-net`:**

```flow
let r = http_get("https://api.example.com/data")

if r.ok == 1.0
    print("Status: {r.status}")      // HTTP status code
    print("Body: {r.body}")          // Response body as string
else
    print("Error: {r.error}")        // Error message
end

// Other methods
let r2 = http_post(url, body)
let r3 = http_put(url, body)
let r4 = http_delete(url)
```

**Response fields:** `.status` (HTTP code), `.body` (response), `.ok` (1.0 if 2xx), `.error` (error message)

---

## 12. Date/Time (Phase 42)

### 12.1 Timestamps (Unix Seconds as Floats)

```flow
let ts = timestamp("2024-01-15T13:30:00Z")   // Parse ISO8601
let unix_ts = timestamp_from_unix(1234567890.0)
let current = now()                          // Current UTC time
```

**Representation:** Timestamps are floats (unix seconds since epoch)

### 12.2 Date Arithmetic

```flow
let later = add_seconds(ts, 60.0)
let tomorrow = add_days(ts, 1.0)
let next_hour = add_hours(ts, 1.0)
let next_minute = add_minutes(ts, 1.0)
```

**All pure math** (portable to .flow, self-hosting friendly).

### 12.3 Date Differences

```flow
let diff_s = diff_seconds(ts2, ts1)
let diff_d = diff_days(ts2, ts1)
let diff_h = diff_hours(ts2, ts1)
```

### 12.4 Formatting (Phase 42 - Basic)

```flow
let formatted = format_datetime(ts, "%Y-%m-%d")              // "2024-01-15"
let detailed = format_datetime(ts, "%Y-%m-%d %H:%M:%S")      // "2024-01-15 13:30:00"
let time_only = format_datetime(ts, "%H:%M:%S")              // "13:30:00"
```

**Supported formats:** Basic strftime patterns only (Phase 42). More in future phases.

---

## 13. Encoding (Phase 41)

### 13.1 Base64

```flow
let encoded = base64_encode("Hello, OctoFlow!")
let decoded = base64_decode(encoded)   // "Hello, OctoFlow!"
```

**Error handling:** `base64_decode()` errors on invalid input.

### 13.2 Hexadecimal

```flow
let hex = hex_encode("ABC")            // "414243"
let decoded = hex_decode("414243")     // "ABC"
```

**Use case:** Data inspection, auth tokens, binary data display.

---

# PART III: ADVANCED

## 14. Modules and Imports

### 14.1 Creating a Module

```flow
// filters.flow
fn brightness(amount)
    return add(amount) |> clamp(0.0, 255.0)
end

fn contrast(factor)
    return subtract(128.0) |> multiply(factor) |> add(128.0)
end

let default_brightness = 20.0
```

### 14.2 Using a Module

```flow
// main.flow
use filters

stream img = tap("photo.jpg")
stream bright = img |> filters.brightness(filters.default_brightness)
emit(bright, "output.jpg")
```

**Import scope:** `use` imports ALL definitions (functions, constants, arrays) from the module.

**Dotted calls:** `filters.brightness(...)` calls the imported function.

---

## 15. Error Handling (Phase 34)

### 15.1 The try() Pattern

```flow
let r = try(http_get("https://example.com"))

if r.ok == 1.0
    print("Success: {r.value}")      // The actual result
else
    print("Error: {r.error}")        // Error message
end
```

**Decomposition:** `try(expr)` catches any runtime error → `.value`, `.ok` (1.0 or 0.0), `.error` (string)

### 15.2 Error Propagation

```flow
let r = try(float("invalid"))
if r.ok == 0.0
    print("Conversion failed: {r.error}")
    // Handle error or return early
end
```

---

## 16. Closures and Higher-Order Functions

See section 5.3 and 9.5 above. Key functions:

```flow
filter(arr, fn(x) condition end)
map_each(arr, fn(x) transform end)
sort_by(arr, fn(x) key_function end)
reduce(arr, init, fn(acc, x) combine end)
```

---

## 17. GPU Acceleration (Automatic)

### 17.1 Stream Pipelines (Phases 0-6)

```flow
stream data = tap("input.csv")
stream normalized = data |> subtract(mean) |> divide(stddev)
stream clamped = normalized |> clamp(-3.0, 3.0)
emit(clamped, "output.csv")
```

**GPU Native, CPU on demand.** All compute runs on GPU by default. CPU handles I/O boundaries (file, network, console) and truly sequential scalar logic. The GPU is the computer; the CPU is the I/O bus.

### 17.2 .octo Binary Format (Phase 18)

```flow
// Write GPU-friendly binary (10-50x faster than CSV)
stream processed = data |> multiply(2.0)
emit(processed, "output.octo")

// Read back
stream loaded = tap("output.octo")
```

**Format:** Columnar f32 storage, zero-copy GPU upload.

### 17.3 GPU-Resident Arrays (Phase 79)

```flow
let a = gpu_fill(1.5, 10000000)    // 10M elements — stays in GPU VRAM
let b = gpu_fill(2.5, 10000000)    // No CPU round-trip
let c = gpu_add(a, b)              // GPU→GPU: zero PCIe transfer
let d = gpu_mul(c, c)              // Chained: still in VRAM
let s = gpu_sum(d)                 // Only downloads when leaving GPU
print("result: {s}")
```

**Data born on GPU stays on GPU.** Chained GPU operations bind directly from VRAM — no PCIe upload/download between steps. Data only materializes to CPU when needed (print, write_csv, for-each iteration).

**GPU array ops:** `gpu_fill`, `gpu_range`, `gpu_add`, `gpu_sub`, `gpu_mul`, `gpu_div`, `gpu_abs`, `gpu_sqrt`, `gpu_exp`, `gpu_log`, `gpu_negate`, `gpu_scale`, `gpu_pow`, `gpu_clamp`, `gpu_where`, `gpu_sum`, `gpu_min`, `gpu_max`, `gpu_random`, `gpu_matmul`, `gpu_ema`, `gpu_cumsum`.

### 17.4 GPU Filesystem (Phase 80)

```flow
// Binary: raw f32 — fastest (625 MB/s save, 890 MB/s load)
let data = gpu_load_binary("prices.bin")     // Disk → GPU VRAM
let result = gpu_mul(data, data)              // Compute in VRAM
gpu_save_binary(result, "output.bin")         // GPU VRAM → Disk

// CSV: text format — compatible with spreadsheets
let csv_data = gpu_load_csv("prices.csv")    // Parse → GPU VRAM
gpu_save_csv(csv_data, "output.csv")          // GPU VRAM → text
```

**Binary format** is raw little-endian f32 — zero parse overhead, 10-20x faster than CSV.
Both `gpu_load_*` return GPU-resident arrays (chained ops stay in VRAM).
`gpu_save_*` return the element count written.

### 17.5 Multi-Input IR + GPU Base64 (Phase 88)

**Multi-input SPIR-V kernels** via `ir_load_input_at(block, binding, idx)`:

```flow
use "stdlib/compiler/ir"
ir_new()
ir_input_count = 4.0   // 4 input bindings (set directly — scalar snapshot semantics)
let bb = ir_block("entry")
let gid = ir_load_gid(bb)
let pixel = ir_udiv(bb, gid, ir_const_u(bb, 4.0))
let rv = ir_load_input_at(bb, 1.0, pixel)   // Read R from binding 1
let gv = ir_load_input_at(bb, 2.0, pixel)   // Read G from binding 2
let bv = ir_load_input_at(bb, 3.0, pixel)   // Read B from binding 3
```

**Output binding auto-shifts:** with `ir_input_count = N`, input bindings are 0..N-1, output is binding N. The `gpu_run` dispatch code already numbers bindings this way.

**GPU base64 encoding** (`stdlib/gpu/b64_emit.flow`): SPIR-V compute shader encodes RGB pixels to base64 ASCII entirely on GPU. 4N threads, one base64 character each. Arithmetic encoding (udiv/umod instead of bitwise), branchless ASCII lookup via `ir_select` chains. Used by kitty terminal renderer for zero-string-copy image output.

```flow
use "stdlib/gpu/b64_emit"
emit_b64_kernel()                                     // Compile once (cached)
let b64 = gpu_run("b64_encode.spv", dummy, r, g, b)  // GPU dispatch
print_bytes(b64)                                       // OS boundary: stdout
```

### 17.6 Pure .flow Terminal Renderers (Phase 88)

4 terminal image rendering protocols, all pure .flow (zero Rust):

| Renderer | Protocol | Quality | File |
|----------|----------|---------|------|
| `halfblock` | ANSI truecolor (default) | Character grid | `stdlib/terminal/halfblock.flow` |
| `kitty` | Kitty Graphics Protocol | Pixel-perfect, GPU-accelerated base64 | `stdlib/terminal/kitty.flow` |
| `sixel` | Sixel graphics | 255-color adaptive palette | `stdlib/terminal/sixel.flow` |
| `digits` | ASCII digit art | Works on any terminal | `stdlib/terminal/digits.flow` |

```flow
use "stdlib/terminal/render"
term_render(w, h, r, g, b, "halfblock")   // Unified dispatcher
term_render(w, h, r, g, b, "kitty")       // GPU base64 → stdout
term_up(30)                                 // Cursor up (for animation)
```

**Output builtins (OS boundary):**
- `print_raw(str)` — write string to stdout without trailing newline
- `print_bytes(arr)` — write array of byte values (0-255) to stdout (no string building)
- `to_str(n)` — convert number to string ("42.0" → "42", "3.14" → "3.14")

### 17.7 Boot Once GPU Runtime (Phase 89)

Pre-record N dispatches into a single `VkCommandBuffer`, single `vkQueueSubmit`.
GPU runs all dispatches autonomously with pipeline barriers. CPU is passive until fence.

**Module:** `stdlib/gpu/runtime.flow` (~580 lines, pure .flow, zero Rust)

```flow
use "stdlib/gpu/runtime"
rt_init()                                          // Boot Vulkan once

let pipe = rt_load_pipeline("shader.spv", 2.0)    // Load SPIR-V, create pipeline
let buf_a = rt_create_buffer(1024.0)               // HOST_VISIBLE buffer
let buf_b = rt_create_buffer(1024.0)
rt_upload(buf_a, data)                             // CPU → GPU

rt_chain_begin(200.0, 2.0)                         // Begin recording chain
for i in range(0, 200)
  let mut bufs = [buf_a, buf_b]                    // Ping-pong buffers
  rt_chain_dispatch(pipe, bufs, 1.0)               // Record dispatch + barrier
end
rt_chain_end()                                     // End recording

rt_chain_submit_wait()                             // Single submit, GPU runs 200 dispatches
rt_download(buf_a, 256.0)                          // GPU → CPU (results in rt_result[])
rt_cleanup()
```

**Key design decisions:**
- Module state in arrays (`rt_handles[]`, `rt_pipes[]`, `rt_bufs[]`) for snapshot-safe cross-function access
- Pipeline barriers after every dispatch (`SHADER_WRITE → SHADER_READ|WRITE`) ensure correct ordering
- Per-dispatch descriptor sets allocated from a pooled `VkDescriptorPool`
- Download results stored in module-level `rt_result[]` array (copy-back semantics)
- FFI trampoline extended to 12 args for `vkCmdPipelineBarrier` (10 params)

**Struct layouts (x64, verified against vk_sys.rs):**
| Struct | Size | Key offsets |
|--------|------|-------------|
| VkMemoryBarrier | 24 | sType@0, srcAccess@16, dstAccess@20 |
| VkWriteDescriptorSet | 64 | sType@0, dstSet@16, dstBinding@24, count@32, type@36, pBufferInfo@48 |
| VkDescriptorBufferInfo | 24 | buffer@0, offset@8, range@16 |
| VkSubmitInfo | 72 | sType@0, cmdBufCount@40, pCmdBufs@48 |

### 17.8 GPU Indirect Dispatch (Phase 90)

GPU self-scheduling: the GPU decides its own workgroup count via `vkCmdDispatchIndirect`.
A compute shader writes `VkDispatchIndirectCommand` to a buffer, and the next dispatch
reads that buffer for workgroup dimensions. CPU never sees the workgroup count.

```flow
use "stdlib/gpu/runtime"
rt_init()

let pipe_set = rt_load_pipeline("12_set_dispatch.spv", 2.0)
let pipe_work = rt_load_pipeline("01_double.spv", 2.0)

let data_buf = rt_create_buffer(1024.0)
let result_buf = rt_create_buffer(1024.0)
let indirect_buf = rt_create_indirect_buffer(12.0)  // {x, y, z} uint32s
rt_upload(data_buf, data)

rt_chain_begin(2.0, 2.0)
  // Dispatch 1: GPU computes workgroup count → writes to indirect buffer
  let mut bufs1 = [data_buf, indirect_buf]
  rt_chain_dispatch(pipe_set, bufs1, 1.0)

  // Dispatch 2: GPU reads workgroup count from indirect buffer
  let mut bufs2 = [data_buf, result_buf]
  rt_chain_dispatch_indirect(pipe_work, bufs2, indirect_buf)
rt_chain_end()
rt_chain_submit_wait()
```

**New API:**
- `rt_create_indirect_buffer(size_bytes)` — creates buffer with `STORAGE|INDIRECT` usage
- `rt_upload_u32(buf_idx, data)` — uploads uint32 array (for dispatch params)
- `rt_chain_dispatch_indirect(pipe_idx, buf_indices, indirect_buf_idx)` — records indirect dispatch

**Barrier update:** All barriers now include `INDIRECT_COMMAND_READ_BIT` in dstAccess and
`DRAW_INDIRECT_BIT` in dstStage. Zero cost when indirect is unused.

### 17.9 Timeline Semaphores (Phase 91)

**GPU self-synchronization.** Multiple `vkQueueSubmit` calls chained via timeline
semaphore values. Zero fences between submissions. The GPU manages all inter-submission
ordering. CPU only calls `vkWaitSemaphores` once at the end.

**Vulkan 1.2 upgrade:** `rt_init()` now creates a Vulkan 1.2 instance with
`VkPhysicalDeviceVulkan12Features.timelineSemaphore` enabled.

**New API:**
- `rt_create_timeline_semaphore(initial_value)` — creates timeline semaphore
- `rt_get_semaphore_value(sem_idx)` — reads current counter value
- `rt_wait_semaphore(sem_idx, value)` — CPU blocks until counter >= value
- `rt_signal_semaphore(sem_idx, value)` — CPU signals counter to value
- `rt_chain_submit_timeline(wait_sem, wait_val, signal_sem, signal_val)` — submit
  with timeline sync (no fence). Pass -1 for wait_sem to skip waiting.
- `rt_chain_reset()` — cleanup after all timeline submits complete

**sType byte decomposition:** Timeline semaphore struct sType values (1000207002-
1000207005) exceed f32 precision. Written byte-by-byte via `vk_set_stype()` helper:
```
1000207002 → [154, 242, 157, 59]  SEMAPHORE_TYPE_CREATE_INFO
1000207003 → [155, 242, 157, 59]  TIMELINE_SEMAPHORE_SUBMIT_INFO
1000207004 → [156, 242, 157, 59]  SEMAPHORE_WAIT_INFO
1000207005 → [157, 242, 157, 59]  SEMAPHORE_SIGNAL_INFO
```

**Multi-submit pattern:**
```
rt_chain_begin(N, 2.0)
  // ... dispatches ...
rt_chain_end()
rt_chain_submit_timeline(-1.0, 0.0, sem, 1.0)  // signal=1

rt_chain_begin(N, 2.0)
  // ... dispatches ...
rt_chain_end()
rt_chain_submit_timeline(sem, 1.0, sem, 2.0)   // wait>=1, signal=2

rt_wait_semaphore(sem, 2.0)  // CPU waits for timeline >= 2
rt_chain_reset()              // cleanup all pools
```

### 17.10 Push Constants (Phase 92)

**Zero-allocation shader parameters.** Push constants pass small values (up to 128 bytes)
directly in the command buffer. No buffer allocation, no descriptor binding, no memory mapping.

**Pipeline creation with push constants:**
```
// 4 bytes = 1 float push constant
let pipe = rt_load_pipeline("shader.spv", 2.0, 4.0)

// No push constants (backward compatible)
let pipe = rt_load_pipeline("shader.spv", 2.0, 0.0)
```

**New API:**
- `rt_load_pipeline(spv_path, num_bindings, pc_size)` — pc_size in bytes (0 = none)
- `rt_chain_push_constants(pipe_idx, values)` — set push constant values (f32 array)

**Usage pattern:**
```
rt_chain_begin(1.0, 2.0)
let mut pc = [3.0]                       // scale factor
rt_chain_push_constants(pipe, pc)        // zero-alloc parameter
let mut bufs = [buf_in, buf_out]
rt_chain_dispatch(pipe, bufs, 1.0)
rt_chain_end()
rt_chain_submit_wait()
```

**GLSL shader side:**
```glsl
layout(push_constant) uniform PC { float scale; } pc;
void main() { out_.data[i] = inp.data[i] * pc.scale; }
```

**Internal:** rt_pipes stride changed from 4 to 5: `[shader_mod, ds_layout, pipe_layout,
pipeline, pc_size]`. VkPushConstantRange (12 bytes) added to pipeline layout when pc_size > 0.

### 17.11 Pipeline Composition (Phase 93)

**Heterogeneous GPU compute chains.** Multiple different shaders with per-dispatch push
constants in a single command buffer submission. This is how real GPU workloads compose:
normalize → transform → reduce → postprocess.

**Pattern: multi-stage pipeline with buffer reuse**
```
let pipe_scale = rt_load_pipeline("13_scale.spv", 2.0, 4.0)
let pipe_offset = rt_load_pipeline("14_offset.spv", 2.0, 4.0)

rt_chain_begin(4.0, 2.0)  // 4 dispatches, 2 bindings each

let mut pc = [5.0]
rt_chain_push_constants(pipe_scale, pc)     // scale by 5
rt_chain_dispatch(pipe_scale, [a, b], 1.0)

let mut pc = [100.0]
rt_chain_push_constants(pipe_offset, pc)    // add 100
rt_chain_dispatch(pipe_offset, [b, c], 1.0)

rt_chain_end()
rt_chain_submit_wait()
```

**Buffer reuse is safe:** `rt_chain_dispatch` inserts pipeline barriers after each
dispatch (SHADER_WRITE → SHADER_READ|WRITE). Subsequent dispatches can safely read
from or write to buffers used by previous dispatches.

---

### 17.12 GPU Database Engine (Phase 94)

**Query plan as dispatch chain.** Process 65,536 rows entirely on GPU.
6 dispatches, 1 submit, 0 CPU row iterations. Multi-pass parallel reduction.

```
// SQL: SELECT SUM(salary), COUNT(*) WHERE salary > 32768
// GPU dispatch chain:
//   [filter: gt_scalar] → [mask: mul_ab] →
//   [reduce pass 1: 256→256] → [reduce pass 2: 256→1] ×2

let pipe_gt = rt_load_pipeline("15_gt_scalar.spv", 2.0, 4.0)
let pipe_mul = rt_load_pipeline("18_mul_ab.spv", 3.0, 0.0)
let pipe_reduce = rt_load_pipeline("reduce_sum.spv", 2.0, 0.0)

rt_chain_begin(6.0, 3.0)

// Stage 1: filter — salary > threshold → boolean mask
let mut pc = [THRESHOLD]
rt_chain_push_constants(pipe_gt, pc)
rt_chain_dispatch(pipe_gt, [salary, mask], NUM_WG)

// Stage 2: mask multiply — salary × mask → masked
rt_chain_dispatch(pipe_mul, [salary, mask, masked], NUM_WG)

// Stages 3-4: reduce sum (2-pass for 65536 elements)
rt_chain_dispatch(pipe_reduce, [masked, partial_sum], NUM_WG)
rt_chain_dispatch(pipe_reduce, [partial_sum, result_sum], 1.0)

// Stages 5-6: reduce count (2-pass)
rt_chain_dispatch(pipe_reduce, [mask, partial_count], NUM_WG)
rt_chain_dispatch(pipe_reduce, [partial_count, result_count], 1.0)

rt_chain_end()
rt_chain_submit_wait()
```

**GPU kernel library (Phase 102):**
| Kernel | Bindings | Push Constants | Operation |
|--------|----------|----------------|-----------|
| `15_gt_scalar.spv` | 2 | threshold (4B) | out[i] = (in[i] > threshold) ? 1.0 : 0.0 |
| `16_lt_scalar.spv` | 2 | threshold (4B) | out[i] = (in[i] < threshold) ? 1.0 : 0.0 |
| `17_eq_scalar.spv` | 2 | value (4B) | out[i] = (in[i] == value) ? 1.0 : 0.0 |
| `18_mul_ab.spv` | 3 | none | out[i] = a[i] × b[i] |
| `reduce_sum.spv` | 2 | none | workgroup tree reduction → 1 sum per WG |
| `22_gemv_relu.spv` | 4 | n_input (4B) | GEMV + ReLU, shared-memory tree reduction |
| `23_gemv.spv` | 4 | n_input (4B) | GEMV (no activation), output layer |
| `24_sha256.spv` | 2 | n_messages (4B) | SHA-256 hash, 1 thread per message |
| `25_bfs_expand.spv` | 5 | level (4B) | BFS frontier expansion (CSR graph) |
| `26_bfs_check.spv` | 2 | none | Frontier empty check (reduce OR) |
| `27_nbody.spv` | 2 | N,dt,soft (12B) | N-body gravity, shared-memory tiling |
| `28_lz4_decompress.spv` | 2 | N,blk_sz (8B) | LZ4 block decompress, 1 thread per block |
| `29_ecs_move.spv` | 5 | dt (4B) | ECS move system: pos += vel × dt |
| `30_ecs_gravity.spv` | 3 | gravity,dt (8B) | ECS gravity: vel_y += g × dt |
| `31_ecs_bounce.spv` | 4 | floor,restit (8B) | ECS bounce: floor collision + restitution |
| `32_scan_sum.spv` | 2 | N (4B) | Blelloch prefix sum, 512 elements per WG |
| `33_scan_add_offset.spv` | 2 | N (4B) | Multi-pass fixup: add scanned block totals |
| `34_histogram.spv` | 2 | N,bins,min,w (16B) | Shared-memory atomic histogram |
| `35_uint_to_float.spv` | 2 | N (4B) | Convert uint buffer to float |
| `36_bitonic_sort.spv` | 1 | k,j,N (12B) | Bitonic sort compare-and-swap step |
| `37_argmin.spv` | 3 | N (4B) | Argmin reduction: value + index |
| `38_argmax.spv` | 3 | N (4B) | Argmax reduction: value + index |
| `39_sliding_sum.spv` | 2 | N,W (8B) | Sliding window sum |
| `40_sliding_avg.spv` | 2 | N,W (8B) | Sliding window average (moving avg) |

**Multi-pass reduction:** For N elements with workgroup_size=256, Pass 1 produces
N/256 partial sums, Pass 2 reduces those to 1. Buffers zero-initialized beyond N
ensures threads reading past data contribute 0.0 (identity for summation).

---

### 17.13 GPU GROUP BY (Phase 95)

**Dispatch fan-out.** Each group in a GROUP BY becomes a sub-chain of dispatches.
K groups × 8 dispatches = one command buffer. Single submit. Zero CPU row iteration.

```
// SQL: SELECT dept, SUM(salary), COUNT(*) GROUP BY dept
// 8 departments × 8 dispatches = 64 dispatches, 1 submit

let pipe_eq = rt_load_pipeline("17_eq_scalar.spv", 2.0, 4.0)
let pipe_mul = rt_load_pipeline("18_mul_ab.spv", 3.0, 0.0)
let pipe_reduce = rt_load_pipeline("reduce_sum.spv", 2.0, 0.0)
let pipe_store = rt_load_pipeline("19_store_at.spv", 2.0, 4.0)

rt_chain_begin(64.0, 3.0)
let mut g = 0.0
while g < K
  // filter: dept == g → mask
  rt_chain_push_constants(pipe_eq, [g])
  rt_chain_dispatch(pipe_eq, [dept, mask], NUM_WG)
  // mask multiply: salary × mask → masked
  rt_chain_dispatch(pipe_mul, [salary, mask, masked], NUM_WG)
  // 2-pass reduce → sum, then scatter to sums[g]
  rt_chain_dispatch(pipe_reduce, [masked, partial], NUM_WG)
  rt_chain_dispatch(pipe_reduce, [partial, temp], 1.0)
  rt_chain_push_constants(pipe_store, [g])
  rt_chain_dispatch(pipe_store, [temp, sums], 1.0)
  // 2-pass reduce → count, then scatter to counts[g]
  rt_chain_dispatch(pipe_reduce, [mask, partial_c], NUM_WG)
  rt_chain_dispatch(pipe_reduce, [partial_c, temp], 1.0)
  rt_chain_push_constants(pipe_store, [g])
  rt_chain_dispatch(pipe_store, [temp, counts], 1.0)
  g = g + 1.0
end
rt_chain_end()
rt_chain_submit_wait()
```

**New kernel:** `19_store_at.spv` — scatter single value to offset: `dst[offset] = src[0]`.
This is the GROUP BY primitive: stores each group's reduction result at its index.

**Buffer reuse:** mask, masked, partial, temp buffers are reused across groups.
Only sums and counts accumulate (store_at writes to different offsets per group).

---

### 17.14 GPU Broadcast Join (Phase 96)

**Dimension table lookup as dispatch chain.** Join a fact table (65,536 rows) with
a dimension table (8 rows) entirely on GPU. No hash table needed — broadcast join
iterates dimension rows and accumulates via fused multiply-add.

```
// SQL: SELECT SUM(salary + bonus) FROM employees e
//      JOIN departments d ON e.dept_id = d.dept_id

let pipe_eq = rt_load_pipeline("17_eq_scalar.spv", 2.0, 4.0)
let pipe_fma = rt_load_pipeline("21_fma_scalar.spv", 2.0, 4.0)
let pipe_add = rt_load_pipeline("20_add_ab.spv", 3.0, 0.0)

// Phase 1: broadcast join — build bonus column
while g < K
  rt_chain_push_constants(pipe_eq, [g])          // dept == g → mask
  rt_chain_dispatch(pipe_eq, [dept, mask], NUM_WG)
  rt_chain_push_constants(pipe_fma, [bonus_d])   // bonus_col += mask × bonus_d
  rt_chain_dispatch(pipe_fma, [mask, bonus_col], NUM_WG)
end

// Phase 2: combine + reduce
rt_chain_dispatch(pipe_add, [salary, bonus_col, total], NUM_WG)
rt_chain_dispatch(pipe_reduce, [total, partial], NUM_WG)
rt_chain_dispatch(pipe_reduce, [partial, result], 1.0)
```

**New kernels:**
| Kernel | Operation | Use |
|--------|-----------|-----|
| `20_add_ab.spv` | out[i] = a[i] + b[i] | Column combination |
| `21_fma_scalar.spv` | out[i] += in[i] × scalar | Broadcast join accumulate |

**In-place fma_scalar:** `out[i] += in[i] * scalar` is safe because each GPU thread
accesses a unique index. The bonus column accumulates across K iterations — after all
departments processed, `bonus_col[i]` contains the correct bonus for employee i.

**GPU query algebra complete (v0.94-0.96):**
| Operation | Kernel Pattern | Phase |
|-----------|---------------|-------|
| SELECT WHERE | filter → mask → reduce | v0.94 |
| SUM | parallel reduction (multi-pass) | v0.94 |
| COUNT | reduce on boolean mask | v0.94 |
| GROUP BY | dispatch fan-out (K × sub-chain) | v0.95 |
| JOIN | broadcast join (fma_scalar accumulate) | v0.96 |

---

### 17.15 GPU Neural Network Inference (Phase 97)

**Forward pass as dispatch chain.** Shared-memory GEMV kernels compute matrix-vector
products with workgroup-level tree reduction. Each workgroup = one output neuron.
256 threads cooperatively reduce the dot product.

```
// Architecture: 256 → 64 (ReLU) → 16
// 2 dispatches, 1 submit, 17,408 parameters

let pipe_hidden = rt_load_pipeline("22_gemv_relu.spv", 4.0, 4.0)
let pipe_output = rt_load_pipeline("23_gemv.spv", 4.0, 4.0)

rt_chain_begin(2.0, 4.0)

// Layer 1: h = relu(W1 * x + b1) — 64 workgroups
rt_chain_push_constants(pipe_hidden, [256.0])   // input_dim
rt_chain_dispatch(pipe_hidden, [W1, x, b1, h], 64.0)

// Layer 2: y = W2 * h + b2 — 16 workgroups
rt_chain_push_constants(pipe_output, [64.0])    // input_dim
rt_chain_dispatch(pipe_output, [W2, h, b2, y], 16.0)

rt_chain_end()
rt_chain_submit_wait()
```

**GEMV kernel architecture:**
- 4 bindings: weights (M×N flat), input (N), bias (M), output (M)
- Push constant: N (input dimension)
- Each workgroup: row `gl_WorkGroupID.x`, 256 threads per workgroup
- Strided dot product: thread k sums W[row*N + k], W[row*N + k+256], ...
- Tree reduction in `shared float[256]` → single dot product
- Thread 0 writes output + bias (+ optional ReLU for hidden layers)

**Scales to arbitrary layer sizes** — input dims > 256 handled by strided loop,
input dims < 256 handled gracefully (excess threads contribute 0.0).

### 17.16 GPU SHA-256 Cryptographic Hashing (Phase 98)

**Embarrassingly parallel hashing.** Each GPU thread independently computes SHA-256
on one 64-byte pre-padded message. FIPS 180-4 compliant. Byte-per-float encoding
avoids uint32/float precision issues — each byte (0-255) is exact in f32.

```
// 1024 parallel SHA-256 hashes, 1 dispatch, 1 submit
let pipe_sha = rt_load_pipeline("24_sha256.spv", 2.0, 4.0)

rt_chain_begin(1.0, 2.0)
rt_chain_push_constants(pipe_sha, [1024.0])   // n_messages
rt_chain_dispatch(pipe_sha, [input, output], 4.0)  // 1024/256 = 4 WGs
rt_chain_end()
rt_chain_submit_wait()
```

**Input format:** N × 64 floats (byte-per-float, 0-255). Each 64-float block is
one pre-padded SHA-256 message (caller handles NIST padding).

**Output format:** N × 32 floats (byte-per-float, 0-255). Each 32-float block
is one 256-bit digest.

**Kernel internals:**
- Rolling 16-element message schedule (64 bytes vs 256 bytes register pressure)
- All 64 FIPS 180-4 round constants as `const uint K[64]`
- Ch, Maj, big/small Sigma functions as GLSL helper functions
- Big-endian byte packing/unpacking for SHA-256 word operations

**Verified:** SHA-256("abc") = `ba7816bf...f20015ad` — BIT EXACT match.

### 17.17 GPU Graph Processing — BFS (Phase 99)

**BFS as frontier expansion.** Each BFS level = 1 GPU dispatch. The frontier expands
in parallel: every active vertex marks its neighbors in the next frontier simultaneously.
CSR (Compressed Sparse Row) graph format for efficient adjacency traversal.

```
// BFS on 64×64 grid (4096 vertices)
let pipe_expand = rt_load_pipeline("25_bfs_expand.spv", 5.0, 4.0)

let mut level = 1.0
while has_frontier
  rt_chain_begin(1.0, 5.0)
  rt_chain_push_constants(pipe_expand, [level])
  rt_chain_dispatch(pipe_expand, [offsets, edges, frontier, next, visited], WGS)
  rt_chain_end()
  rt_chain_submit_wait()
  // swap frontier ← next, check empty, level++
end
```

**Kernel bindings:** 5 (CSR offsets, CSR edges, frontier, next_frontier, visited).
Push constant: current BFS level.

**Pattern:** Multi-submit iterative dispatch. Each level is one submit because the
CPU must check frontier emptiness. Future: indirect dispatch could eliminate CPU check.

**Verified:** 64×64 grid → 127 BFS levels, all 4096 vertices reached, corner distances BIT EXACT.

### 17.18 GPU N-body Simulation (Phase 100)

**All-pairs gravitational N-body.** Each thread computes total force on one particle
from all N others using shared-memory tiling. Leapfrog integration. Double-buffered
state for ping-pong dispatch.

```
// 4096 particles, 100 timesteps, 1 dispatch per step
let pipe = rt_load_pipeline("27_nbody.spv", 2.0, 12.0)

let mut step = 0.0
while step < 100.0
  rt_chain_begin(1.0, 2.0)
  rt_chain_push_constants(pipe, [N, dt, softening])
  rt_chain_dispatch(pipe, [buf_in, buf_out], 16.0)  // 4096/256
  rt_chain_end()
  rt_chain_submit_wait()
  // swap buffers
  step = step + 1.0
end
```

**Kernel architecture:**
- 2 bindings: input state (x,y,vx,vy per particle), output state
- 3 push constants: N, dt, softening (12 bytes)
- Shared-memory tiling: 256 particles loaded per tile, all-pairs within tile
- `inversesqrt(dist² + ε²)` for softened gravity
- O(N²) interactions per step, embarrassingly parallel

**Verified:** 4-particle symmetry test (CoM conservation exact), 4096-particle
demo (1.67 billion interactions, CoM drift < 0.00005).

### 17.19 GPU LZ4 Block Decompression (Phase 101)

**Parallel block decompression.** Each GPU thread independently decompresses one
LZ4-compressed block. Embarrassingly parallel: blocks are independent. Input uses
block offset table for variable-length compressed data.

```
// 1024 blocks decompressed in parallel, 1 dispatch
let pipe = rt_load_pipeline("28_lz4_decompress.spv", 2.0, 8.0)

rt_chain_begin(1.0, 2.0)
rt_chain_push_constants(pipe, [n_blocks, out_block_size])
rt_chain_dispatch(pipe, [compressed_input, decompressed_output], 4.0)
rt_chain_end()
rt_chain_submit_wait()
```

**Input layout:** `[offset_0, offset_1, ..., offset_N, compressed_data...]`
Offsets index into compressed data. Variable-length blocks supported.

**LZ4 format support:** Token-based literal+match sequences, extended lengths
(255-continuation), overlapping match copies (RLE pattern).

**Verified:** 8:1 ratio (unit test), 21:1 ratio (demo), BIT EXACT decompression.

### 17.20 GPU ECS — Game Logic (Phase 102)

**Entity Component System on GPU.** Structure-of-Arrays memory layout — each component
type is a separate GPU buffer. Systems are GPU dispatches that process all entities
with the relevant components in parallel.

```
// 3 systems × 100 steps = 300 dispatches
pipe_grav   = rt_load_pipeline("30_ecs_gravity.spv", 3.0, 8.0)
pipe_move   = rt_load_pipeline("29_ecs_move.spv", 5.0, 4.0)
pipe_bounce = rt_load_pipeline("31_ecs_bounce.spv", 4.0, 8.0)

while step < 100.0
  dispatch gravity(vel_y, alive, grounded)     // push: gravity, dt
  dispatch move(pos_x, pos_y, vel_x, vel_y, alive)  // push: dt
  dispatch bounce(pos_y, vel_y, alive, grounded)     // push: floor, restitution
end
```

**SoA component arrays:** pos_x, pos_y, vel_x, vel_y, alive, grounded — each a
separate `float[]` buffer. Alive flag controls per-entity processing. Grounded flag
prevents gravity from applying.

**Pattern:** Each system = 1 kernel with different binding sets over the same entity
arrays. Adding new systems = new kernel + new dispatch in the loop. No CPU entity
iteration needed.

**Verified:** 4-entity exact physics test + 4096-entity demo (300 dispatches,
no floor penetration).

---

### 17.21 GPU Prefix Scan — Keystone Primitive (Phase 103)

**Work-efficient Blelloch parallel prefix sum.** The keystone primitive for GPU
computing — prefix scan composes into sort, compact, histogram, radix sort,
stream compaction, and dozens of other algorithms.

```
// Three-phase multi-pass for N > 512:
// Phase 1: Scan each 512-element block independently (N/512 WGs)
// Phase 2: Extract & scan the block totals (1 WG)
// Phase 3: Add scanned totals back (fixup, N/256 WGs)

pipe_scan  = rt_load_pipeline("32_scan_sum.spv", 2.0, 4.0)
pipe_fixup = rt_load_pipeline("33_scan_add_offset.spv", 2.0, 4.0)

rt_chain_begin(1.0, 2.0)
  dispatch scan(input, scanned, N_BLOCKS)        // 512 elems/WG
  dispatch scan(totals, totals_scanned, 1.0)     // scan the totals
  dispatch fixup(scanned, totals_scanned, N/256) // propagate
rt_chain_end()
```

**Algorithm:** 256 threads per WG, each loads 2 elements into shared memory.
Up-sweep (reduce) builds partial sum tree, down-sweep propagates exclusive
prefix sums. Final write: inclusive = exclusive + original.

**Fixup kernel:** Uses `gid / 512` to find scan block (not `gl_WorkGroupID`
which has 256-thread alignment). First scan block skipped (offset = 0).

**Verified:** N=8 and N=256 BIT EXACT (single block) + N=65536 BIT EXACT
(128 blocks, 3-pass pipeline, spot check every 1000 elements).

---

### 17.22 GPU Histogram (Phase 104)

**Shared-memory atomic histogram.** Each workgroup bins elements locally using
`atomicAdd` on shared `uint[256]`, then accumulates into global uint output.
A second kernel converts `uint[]` to `float[]` for OctoFlow consumption.

```
// 2 dispatches, 1 submit, 0 CPU element iterations
pipe_hist = rt_load_pipeline("34_histogram.spv", 2.0, 16.0)
pipe_conv = rt_load_pipeline("35_uint_to_float.spv", 2.0, 4.0)

rt_chain_begin(2.0, 2.0)
  dispatch histogram(input, hist_uint, 256 WGs)  // pc: N, bins, min, width
  dispatch uint_to_float(hist_uint, hist_float, 1 WG)
rt_chain_end()
```

**Push constants:** n_elements, num_bins, min_val, bin_width (16 bytes).
Supports up to 256 bins (limited by shared memory = workgroup size).

**Verified:** N=256 uniform and all-zeros BIT EXACT + N=65536 16-bin
BIT EXACT (all bins = 4096, 256 workgroups).

---

### 17.23 GPU Bitonic Sort (Phase 105)

**Parallel sorting network.** O(log²N) dispatches of XOR-based compare-and-swap.
Each thread finds its partner via `gid XOR j`, compares, and swaps based on
ascending/descending direction `(gid AND k) == 0`. In-place sort on one buffer.

```
// Bitonic network: k=2,4,...,N outer; j=k/2,k/4,...,1 inner
pipe = rt_load_pipeline("36_bitonic_sort.spv", 1.0, 12.0)

while k <= N
  while j >= 1.0
    dispatch sort(data, ceil(N/256) WGs)  // pc: k, j, N
  end
end
// N=65536 → 136 dispatches, 256 WGs each
```

**In-place:** Single buffer, no auxiliary memory. Each dispatch is one step
of the sorting network. N must be power of 2.

**Verified:** N=8 reverse-sorted and N=256 interleaved-shuffle BIT EXACT +
N=65536 fully reverse-sorted BIT EXACT (136 dispatches, spot-checked).

---

### 17.24 GPU Argmin/Argmax (Phase 106)

**Parallel reduction tracking both value and index.** Each workgroup reduces
its chunk to (min_value, min_index) or (max_value, max_index) pairs.
Multi-pass for large arrays: 65536→256→1.

```
pipe_argmin = rt_load_pipeline("37_argmin.spv", 3.0, 4.0)
pipe_argmax = rt_load_pipeline("38_argmax.spv", 3.0, 4.0)

// 2-pass: 65536→256 partial results, 256→1 final
dispatch argmin(data, partial_val, partial_idx, 256 WGs)
dispatch argmin(partial_val, final_val, final_idx, 1 WG)
// Look up original index: partial_idx[final_idx]
```

**Shared memory:** Two arrays `s_val[256]` and `s_idx[256]` — tree reduction
preserves the index of the winning value. Sentinel: +inf for argmin, -inf
for argmax.

**Verified:** N=8 argmin+argmax exact, N=256 argmin at known position,
N=65536 two-pass BIT EXACT.

---

### 17.25 GPU Sliding Window (Phase 107)

**Parallel sliding window sum and moving average.** Each thread computes
its own window sum — O(N×W) total work but fully parallel across N elements.

```
pipe_sum = rt_load_pipeline("39_sliding_sum.spv", 2.0, 8.0)
pipe_avg = rt_load_pipeline("40_sliding_avg.spv", 2.0, 8.0)

// 2 dispatches, 1 submit
dispatch sliding_sum(input, sums, 256 WGs)   // pc: N, W
dispatch sliding_avg(input, avgs, 256 WGs)   // pc: N, W
```

**Window semantics:** Trailing/causal window. `out[i] = f(inp[max(0,i-W+1)..i])`.
Leading edge elements (i < W) use shorter window. Moving average divides by
actual window size, not W.

**Verified:** Sliding sum W=3 exact, moving average W=4 exact,
N=65536 W=64 BIT EXACT (spot-checked every 1000).

**All 5 keystone GPU primitives now complete:**
prefix scan, histogram, bitonic sort, argmin/argmax, sliding window.

---

### 17.26 GPU-Native Standard Library (Phases 108-114)

**100+ GPU-accelerated functions built by composing the 40 GPU kernels.**
Zero Rust compiler changes — pure .flow composition using Architecture C pattern.

**Module Organization:**

1. **`stdlib/gpu/stats.flow`** — Statistical operations (10 functions)
   - `gpu_sum`, `gpu_mean`, `gpu_min`, `gpu_max`, `gpu_variance`, `gpu_std`
   - `gpu_median`, `gpu_percentile`, `gpu_correlation`, `gpu_histogram_counts`
   - **Status**: 17/21 tests passing, production-ready for core ops

2. **`stdlib/gpu/array_ops.flow`** — Array transformations (8 functions)
   - `gpu_filter`, `gpu_compact`, `gpu_unique`, `gpu_top_k`, `gpu_reverse`
   - `gpu_normalize`, `gpu_clamp`, `gpu_map` (sqrt/square/abs/negate/exp/log)
   - **Status**: 6/8 tests passing

3. **`stdlib/gpu/linalg.flow`** — Linear algebra (7 functions)
   - `gpu_matmul`, `gpu_transpose`, `gpu_dot_product`, `gpu_vector_add`
   - `gpu_vector_scale`, `gpu_matrix_vector_mul`, `gpu_outer_product`
   - **Status**: 7/7 tests passing (100%), production-ready

4. **`stdlib/gpu/signal.flow`** — Signal processing (12 functions)
   - Convolution, autocorrelation, cross-correlation, diff, cumsum, EMA
   - **Status**: Implementation complete, syntax fixes pending

5. **`stdlib/gpu/aggregate.flow`** — Data aggregation (13 functions)
   - Group-by operations (sum/count/mean/min/max)
   - Rolling windows, binned statistics
   - **Status**: Implementation complete, syntax fixes pending

6. **`stdlib/gpu/math_advanced.flow`** — Advanced math (14 functions)
   - Neural network activations: sigmoid, relu, tanh, softmax
   - Distance metrics: euclidean, manhattan, cosine similarity
   - Norms: L1, L2, pairwise distances
   - **Status**: Implementation complete, syntax fixes pending

7. **`stdlib/gpu/composite.flow`** — Data science utilities (31 functions)
   - Statistical analysis: describe, outlier detection, standardization
   - Time series: returns, Bollinger bands, RSI, MACD, ATR
   - Clustering helpers: kmeans_assign, dbscan_core
   - **Status**: Implementation complete, syntax fixes pending

**Architecture Pattern:**
```flow
fn gpu_function(input_array)
  // 1. Load pipeline
  let pipe = rt_load_pipeline("path/to/kernel.spv", bindings, pc_bytes)

  // 2. Create buffers
  let buf_in = rt_create_buffer(n * 4.0)
  let buf_out = rt_create_buffer(n * 4.0)

  // 3. Upload, dispatch, download
  rt_upload(buf_in, input_array)
  rt_chain_begin(1.0, bindings)
  rt_chain_dispatch(pipe, [buf_in, buf_out], workgroups)
  rt_chain_end()
  rt_chain_submit_wait()
  rt_download(buf_out, n)

  return rt_result
end
```

**Performance**: GPU faster than CPU for N > 1000 (element-wise), N > 5000 (reductions),
N > 10000 (sort/percentile), matrix dims > 64×64.

**Total Library**: 95+ functions, 40 GPU kernels, 0 Rust changes.
See `stdlib/gpu/README_STDLIB.md` for complete documentation.

---

### 17.27 GPU-Resident Autonomous Agents (Phase 115)

**The boundary-breaking demo that proves OctoFlow is fundamentally different.**

Not just "parallel loops" (OpenMP can do that). Not just "call GPU kernels" (CUDA can do that).
But **natural control flow with GPU doing autonomous decision-making** — a computation pattern
impossible in CPU-native languages.

**Two showcase demos:**

1. **Pathfinding Agent** (`examples/gpu_autonomous_agent.flow`)
   ```flow
   // Agent navigates 32×32 grid with obstacles to reach goal
   // Perception → Decision → Action loop, 52 iterations
   while not_at_goal
     // GPU: Compute attraction field (1024 cells × 5 ops)
     let field = gpu_compute_field(grid, agent_pos, goal_pos)
     // CPU: Pick best direction (simple argmax)
     let next_pos = pick_best_direction(field, agent_pos)
     // Update and check convergence
     agent_pos = next_pos
   end
   // 260 GPU dispatches, agent autonomously seeks goal
   ```

2. **Multi-Dispatch Pipeline** (`examples/gpu_autonomous_multi_dispatch.flow`)
   ```flow
   // Process 1M elements: 5-stage pipeline + 5 reductions
   // 10M element-ops in 131ms (76,000 ops/ms)
   // CPU involvement: 0.00012% (only 12 CPU-GPU interactions)
   // GPU autonomy: 99.9999%
   ```

**Key metrics**:
- Pathfinding: 52 iterations × 5 GPU ops/iter = 260 dispatches
- Pipeline: 1M elements, 10 GPU operations, **99.9999% GPU autonomy**
- No manual memory management, no explicit synchronization
- Natural `.flow` control flow spanning GPU operations

**Why this matters**:
- **Python**: Would run on CPU (100× slower) or need explicit CuPy orchestration
- **CUDA/C++**: 300+ lines of boilerplate, manual memory, explicit kernels
- **OctoFlow**: 150-280 lines, natural loops, automatic GPU dispatch

The GPU isn't just a "math coprocessor" — it's an **autonomous compute engine**
that makes decisions and processes data with minimal CPU involvement.

See `examples/GPU_AUTONOMOUS_AGENTS.md` for complete analysis and comparisons.

---

### 17.28 OpAtomicIAdd — GPU Atomics Foundation (Phase 116)

**Atomic integer operations in SPIR-V IR builder.** Enables lock-free algorithms,
GPU-side counters, and concurrent data structures.

```flow
// IR builder API (ir.flow)
let ptr = ir_access_chain(block, buffer_id, index_id)
let old_val = ir_atomic_iadd(block, ptr, value_id)
// Returns old value before increment
```

**SPIR-V Emission**:
- Opcode: 207 (OpAtomicIAdd)
- Memory Scope: Device (1) — visible to all GPU invocations
- Semantics: AcquireRelease | UniformMemory (72 decimal, 0x48 hex)
- Format: 7 words `[OpAtomicIAdd %uint %result %pointer %scope %semantics %value]`

**Automatic Constants**: IR emitter detects atomic ops and allocates scope/semantics
constants automatically. No manual SPIR-V ID management.

**What This Unlocks**:
1. **GPU Histogram** — Atomic binning without CPU fallback (fixes 29/29 test pass rate)
2. **Decoupled Lookback Scan** — Merrill & Garland 2016 single-pass prefix sum (faster)
3. **GPU Counters** — Thread-safe incrementing for dispatch chain statistics
4. **Lock-Free Structures** — Concurrent queues, stacks, hash tables on GPU

**Verified**: ir.flow generates valid SPIR-V, spirv-val passes. Atomic counter
test emits correct binary structure.

**Foundation for 99% GPU autonomy** — GPU can now coordinate internally without
CPU for counter/accumulation operations.

---

## 18. Security Model (Capability-Based)

### 18.1 Deny by Default

**Without flags, programs are sandboxed:**

```flow
let x = read_file("data.txt")        // ERROR: not permitted
let r = http_get("https://...")      // ERROR: not permitted
let c = exec("ls")                   // ERROR: not permitted
```

### 18.2 Explicit Permissions

```bash
flowgpu-cli run program.flow --allow-read      # File reads
flowgpu-cli run program.flow --allow-write     # File writes
flowgpu-cli run program.flow --allow-net       # Network access
flowgpu-cli run program.flow --allow-exec      # Command execution

# Combine as needed
flowgpu-cli run program.flow --allow-read --allow-write --allow-net
```

### 18.3 Security Guarantees

- No program can access what it hasn't been granted
- Path traversal attacks prevented (`canonicalize_path` validates)
- No null pointer dereferences (no null type)
- No buffer overflows (bounds-checked arrays)
- No uninitialized memory (all variables initialized)

---

# PART IV: DOMAIN FOUNDATIONS

## 19. Data Science and Analytics

**Readiness: 10/10** (Phase 42)

### Core Capabilities

```flow
// Load CSV data
let sales = read_csv("sales_data.csv")

// Extract columns
let revenues = map_each(sales, fn(row) float(row["revenue"]) end)

// Compute statistics
let avg_revenue = mean(revenues)
let std_revenue = stddev(revenues)
let q95 = quantile(revenues, 0.95)

print("Mean: {avg_revenue:.2}, StdDev: {std_revenue:.2}, 95th: {q95:.2}")

// Filter outliers
let filtered = filter(revenues, fn(x) abs(x - avg_revenue) < 3.0 * std_revenue end)

// Correlation analysis
let prices = map_each(sales, fn(row) float(row["price"]) end)
let corr = correlation(revenues, prices)
print("Revenue-Price correlation: {corr:.3}")
```

### GPU Advantage

Large datasets process faster via GPU-accelerated statistics (when implemented in GPU kernels, Phase 47+).

---

## 20. Finance and Quantitative

**Readiness: 10/10** (Phase 42)

### Core Capabilities

```flow
// Backtesting with timestamps
let trades = read_csv("XAUUSD_M5.csv")

for trade in trades
    let ts = timestamp(trade["time"])
    let price = float(trade["close"])

    // Date-based filtering
    let start_date = timestamp("2024-01-01T00:00:00Z")
    let end_date = timestamp("2024-12-31T23:59:59Z")

    if ts >= start_date && ts <= end_date
        // Analyze trade
    end
end

// Rolling statistics
let prices = map_each(trades, fn(t) float(t["close"]) end)
let last_20 = slice(prices, -20, 0)
let sma = mean(last_20)
let volatility = stddev(last_20)
```

### Time-Series Operations

```flow
// Calculate returns
let returns = []
for i in range(1, len(prices))
    let ret = (prices[i] - prices[i-1]) / prices[i-1]
    push(returns, ret)
end

let mean_return = mean(returns)
let sharpe = mean_return / stddev(returns)
```

---

## 21. DevOps and Automation

**Readiness: 10/10** (Phase 42)

### Core Capabilities

```flow
// Safe file operations (no bash quoting bugs)
let logs = list_dir("/var/log")

for file in logs
    if ends_with(file, ".log")
        let full_path = join_path("/var/log", file)
        let lines = read_lines(full_path)

        // Process logs
        let errors = filter(lines, fn(l) contains(l, "ERROR") end)

        if len(errors) > 0.0
            print("{file}: {len(errors)} errors")
        end
    end
end

// Command execution
let git_status = exec("git", "status", "--short")
if git_status.ok == 1.0
    print("{git_status.output}")
end

// HTTP API automation
let response = http_post(
    "https://api.example.com/deploy",
    "{\"version\": \"1.2.3\"}"
)
```

### Type-Safe Paths

**No more bash disasters:**
```flow
// This is safe (typed strings, automatic escaping)
let backup_dir = "quarterly report Q4 2024"
let path = join_path("/backups", backup_dir, "data.csv")
// path is correctly formed, no word-splitting
```

---

## 22. Systems and Infrastructure

**Readiness: 9/10** (Phase 42)

### Core Capabilities

```flow
// System monitoring
let cpu_info = exec("cat", "/proc/cpuinfo")
let mem_info = exec("free", "-h")

// File metadata
let size = file_size("/var/log/app.log")
let exists = file_exists("/etc/config.conf")
let is_dir_check = is_dir("/var/lib/app")

// Environment access
let home = env("HOME")
let os = os_name()                   // "linux", "windows", "macos"
let epoch = time()                   // Unix timestamp (deprecated, use now())
```

### Log Analysis

```flow
let logs = read_lines("/var/log/nginx/access.log")
let error_404 = filter(logs, fn(l) contains(l, " 404 ") end)
print("404 errors: {len(error_404)}")
```

---

## 23. Web and Networked Applications

**Readiness: 7/10** (Phase 42)

### Core Capabilities

```flow
// API client with auth
let credentials = "user:pass"
let auth_header = "Basic {base64_encode(credentials)}"

// HTTP with headers (Phase 35 - basic, headers in Phase 45+)
let response = http_get("https://api.example.com/data")

if response.ok == 1.0
    let data = json_parse(response.body)
    // Process data
end

// Error handling
let r = try(http_get("https://invalid-url"))
if r.ok == 0.0
    print("Request failed: {r.error}")
end
```

### Missing (Future Phases)

- TCP/UDP sockets (Phase 45)
- HTTP server (Phase 46) ✅
- WebSockets (Phase 47+)
- Custom headers in HTTP (Phase 45+)

---

## 24. GUI Toolkit

OctoFlow includes a pure-.flow GUI widget toolkit built on `window_*` builtins. It provides 13 widget types with software rendering (no external GUI dependencies).

### 24.1 Quick Start

```flow
use "gui"

let app = gui_init(800.0, 600.0, "My App")
let btn = gui_button(10.0, 10.0, 120.0, 36.0, "Click Me")
let lbl = gui_label(10.0, 60.0, "Status: ready")

while gui_running() == 1.0
  let _u = gui_update()
  if gui_clicked(btn) == 1.0
    let _s = gui_set_text(lbl, "Button clicked!")
  end
end
```

### 24.2 Widget Types (13 total)

| Widget | Constructor | Description |
|--------|------------|-------------|
| Panel | `gui_panel(x, y, w, h)` | Container with rounded corners |
| Label | `gui_label(x, y, text)` | Static text display |
| Button | `gui_button(x, y, w, h, text)` | Clickable button |
| Checkbox | `gui_checkbox(x, y, text)` | Toggle with checkmark |
| Slider | `gui_slider(x, y, w, min, max, initial)` | Horizontal value slider |
| Text Input | `gui_textinput(x, y, w)` | Editable text field |
| Progress | `gui_progress(x, y, w)` | Progress bar (0-100) |
| Radio | `gui_radio(x, y, text, group)` | Radio button (grouped) |
| Separator | `gui_separator(x, y, w)` | Horizontal line |
| Listbox | `gui_listbox(x, y, w, h)` | Scrollable item list |
| Spinbox | `gui_spinbox(x, y, w, min, max, initial, step)` | Numeric +/- input |
| Dropdown | `gui_dropdown(x, y, w)` | Combo box with popup overlay |
| Tabs | `gui_tabs(x, y, w, h)` | Tabbed panel selector |

### 24.3 Widget Queries and Setters

**Queries:**

```flow
gui_running()              // 1.0 if window open
gui_clicked(id)            // 1.0 if clicked this frame
gui_checked(id)            // 1.0 if checkbox/radio checked
gui_slider_value(id)       // Current slider value
gui_get_text(id)           // Text input content
gui_listbox_selected(id)   // Selected item index (-1 if none)
gui_listbox_text(id, idx)  // Item text at index
gui_spinbox_value(id)      // Current spinbox value
gui_dropdown_selected(id)  // Selected dropdown index
gui_dropdown_text(id, idx) // Dropdown item text
gui_tab_active(id)         // Active tab index
gui_radio_value(group)     // Checked radio ID in group
```

**Setters:**

```flow
gui_set_text(id, text)       // Change label/button text
gui_set_checked(id, val)     // 1.0=checked, 0.0=unchecked
gui_set_slider(id, val)      // Set slider value
gui_set_visible(id, val)     // Show/hide widget
gui_set_enabled(id, val)     // Enable/disable widget
gui_set_progress(id, val)    // Set progress (clamped 0-100)
gui_set_spinbox(id, val)     // Set spinbox (clamped to range)
gui_set_dropdown(id, idx)    // Set selected dropdown item
gui_set_tab(id, idx)         // Switch active tab
```

### 24.4 Compound Widgets

**Listbox items:**
```flow
let lst = gui_listbox(10.0, 10.0, 200.0, 100.0)
let _a = gui_listbox_add(lst, "Item A")
let _b = gui_listbox_add(lst, "Item B")
```

**Dropdown items:**
```flow
let dd = gui_dropdown(10.0, 10.0, 150.0)
let _a = gui_dropdown_add(dd, "Red")
let _b = gui_dropdown_add(dd, "Green")
```

**Tab panels:**
```flow
let tabs = gui_tabs(10.0, 10.0, 300.0, 100.0)
let _t1 = gui_tab_add(tabs, "Home")
let _t2 = gui_tab_add(tabs, "Settings")
```

### 24.5 Themes

Three built-in color themes:

```flow
let _t = gui_theme_dark()     // Dark (default)
let _t = gui_theme_light()    // Light
let _t = gui_theme_ocean()    // Ocean blue
```

### 24.6 Architecture Notes

- **All state in arrays.** Due to OctoFlow's scalar snapshot semantics, mutable state must live in arrays (which propagate back through function calls). The `_gs[]` state array, `_gui_types[]` widget registry, and per-widget arrays like `_gui_clicked[]` use this pattern.
- **Software rendered.** The toolkit renders to pixel buffers (`_gui_px_r/g/b`) using a 4x6 bitmap font. No GPU rendering (yet).
- **Dropdown Z-ordering.** Overlay rendered in a second pass after normal widgets, with overlay clicks handled first for Z-priority.
- **Shared flat arrays.** Listbox, dropdown, and tabs share `_gui_list_items[]` storage. Spinbox and tabs share `_gui_slider_val[]`.

---

## 25. AI and LLM Inference

OctoFlow can load and run GGUF model files for LLM text generation — entirely in .flow with Rust builtins for the compute-intensive inner loops.

### 25.1 GGUF Model Loading

```flow
use "../formats/gguf"

let model = gguf_load_from_file("model.gguf")

// All metadata accessible via kv.* prefix
let arch = map_get(model, "arch")
let n_layer = map_get(model, "n_layer")
let n_head = map_get(model, "n_head")
let n_embd = map_get(model, "n_embd")
```

### 25.2 GGUF API

**Metadata access:**

```flow
// Direct key access
let val = map_get(model, "kv.qwen2.rope.freq_base")

// Convenience helpers (tries arch prefix, then llama fallback)
let eps = gguf_meta_default(model, "attention.layer_norm_rms_epsilon", 0.00001)
let rope = gguf_meta_default(model, "rope.freq_base", 10000.0)
```

**Tensor loading:**

```flow
// Load single tensor by name (auto-dispatches F32/F16/Q4_K dequant)
let weights = gguf_load_tensor(model_path, model, "output_norm.weight")

// Load embedding row (for token ID lookup)
let emb = gguf_load_tensor(model_path, model, "token_embd.weight", token_id)
```

**Vocabulary and tokenization:**

```flow
let vocab = gguf_load_vocab(model_path)
let token_ids = gguf_tokenize(model_path, "Hello world")
let text = vocab[int(token_id)]
```

### 25.3 Transformer Inference

The inference pipeline runs per-layer:

```flow
// Run transformer layer (attention + FFN, with KV cache)
let layer_out = gguf_infer_layer(model_path, model, hidden, layer_idx, seq_pos, max_seq)
```

`gguf_infer_layer` is a Rust builtin that handles:
- RMSNorm (attention + FFN)
- Q/K/V projections with GQA support
- RoPE positional embeddings
- KV cache management
- Multi-head attention with softmax
- SiLU-gated FFN (gate * up * down)
- Residual connections

**GPU acceleration:** `gguf_matvec` dispatches matrix-vector multiplies on GPU with automatic tensor caching (dequant once, dispatch always).

### 25.4 Token Sampling

```flow
// Greedy: argmax over logits
let mut best_id = 0.0
let mut best_val = -999999.0
let mut v = 0.0
while v < vocab_size
  if logits[int(v)] > best_val
    best_val = logits[int(v)]
    best_id = v
  end
  v = v + 1.0
end

// Top-K with temperature: see stdlib/ai/generate.flow for full implementation
```

### 25.5 Complete Generation Example

See `stdlib/ai/generate.flow` for a full working example that:
- Loads any GGUF model (tested with Qwen2.5-Coder)
- Builds chat template with special tokens
- Prefills prompt tokens with KV cache
- Generates tokens with greedy or top-K sampling
- Decodes BPE tokens to readable text

**Supported model formats:** F32, F16, Q4_K quantization. Architecture-adaptive (reads all params from GGUF metadata).

---

# PART V: LANGUAGE DESIGN

## 26. Syntax Philosophy

### 26.1 Minimal Concepts (23 Total)

OctoFlow has exactly 23 language concepts:
1. Values (float, string, map)
2. Variables (let, let mut)
3. Assignment (=)
4. Operators (+, -, *, /, ==, <, >, &&, ||)
5. If expression (if-then-else)
6. If statement block (if-elif-else-end)
7. Print (with interpolation)
8. Functions (fn-return-end)
9. Closures/lambdas (fn(x) expr end)
10. Arrays ([...])
11. HashMaps (map())
12. Structs (struct Name(fields))
13. Vectors (vec2, vec3, vec4)
14. While loops (while-end)
15. For loops (for-in-range-end)
16. For-each loops (for-in-array-end)
17. Break/continue
18. Modules (use)
19. GPU streams (stream = tap |> ops |> emit)
20. Error handling (try)
21. Type conversion (str, float, int, type_of)
22. Command execution (exec)
23. Security flags (--allow-*)

**That's all.** No classes, no inheritance, no macros, no generics, no traits, no async, no unsafe, no lifetimes.

### 26.2 Why 23 Is Enough

**Small enough for LLMs to master completely.** A fine-tuned 1-3B parameter model can achieve near-perfect OctoFlow code generation.

**Large enough for real programs.** 801 tests, 9 domains at 7-10/10 readiness, complete neural network framework buildable.

**The constraint IS the power.** Fewer concepts = higher reliability = cheaper LLM generation = more accessible.

---

## 27. Type System

### 27.1 Current (Phase 42)

**Runtime types:**
- `Value::Float(f32)` — all numbers
- `Value::Str(String)` — all text
- `Value::Map(HashMap<String, Value>)` — structured data

**Compile-time tracking:**
- Scalars (float or string)
- Arrays (Vec<Value>)
- HashMaps (HashMap<String, Value>)
- Streams (GPU pipelines)

**No static type declarations.** Types are inferred.

### 27.2 Future (Phase 48)

**Entity-relation types:**
```flow
type Person {
    name: string,
    age: float
}

type AuthoredBy {
    relates author: Person,
    relates paper: Paper
}
```

**Polymorphic subtyping:**
```flow
type Author sub Person
type Researcher sub Person
```

---

## 28. Memory Model

### 28.1 No Manual Memory Management

**No malloc, no free, no pointers** (except in FFI via `extern` blocks, Phase 44+).

**Automatic cleanup:** Variables deallocate at end of scope.

**No reference counting visible to user:** Internal implementation detail.

### 28.2 GPU Memory (Transparent)

```flow
stream data = tap("input.csv")
stream result = data |> multiply(2.0)
emit(result, "output.csv")
```

**The compiler handles:**
- Uploading data to GPU
- Kernel dispatch
- Downloading results
- Memory allocation/deallocation

**User never thinks about GPU memory.**

---

## 29. Execution Model

### 29.1 Eager Evaluation

```flow
let x = expensive_computation()      // Executes immediately
let y = x + 1.0                      // Uses the result
```

**Not lazy.** Every expression evaluates when encountered.

### 29.2 GPU vs CPU Partitioning (Automatic)

**Compiler decides based on cost model:**
- Math on large arrays → GPU
- String operations → CPU
- File I/O → CPU
- Small arrays (<1000 elements) → CPU (overhead not worth it)
- Large arrays (>1000 elements) → GPU

**User never specifies.** It just works.

---

# APPENDICES

## A. Complete Function Reference (Phase 42)

### A.1 Math

`abs`, `sqrt`, `pow`, `exp`, `log`, `sin`, `cos`, `floor`, `ceil`, `round`, `clamp`

### A.2 Statistics (Phase 41)

`mean`, `median`, `stddev`, `variance`, `quantile`, `correlation`

### A.3 String

`len`, `contains`, `substr`, `replace`, `trim`, `to_upper`, `to_lower`, `starts_with`, `ends_with`, `index_of`, `char_at`, `repeat`, `split`, `join`, `str`, `float`, `int`

### A.4 Array

`len`, `push`, `pop`, `first`, `last`, `reverse`, `slice`, `sort_array`, `unique`, `find`, `sum`, `min_val`, `max_val`, `map_each`, `filter`, `sort_by`, `reduce`

### A.5 HashMap

`map`, `map_set`, `map_get`, `map_has`, `map_remove`, `map_keys`, `json_parse`, `json_stringify`, `load_data`, `save_data`, bracket access `map["key"]`

### A.6 File I/O (requires --allow-read/write)

`read_file`, `write_file`, `append_file`, `read_lines`, `list_dir`, `read_csv`, `write_csv`, `file_exists`, `file_size`, `is_directory` (alias: `is_dir`), `file_ext`, `file_name` (alias: `basename`), `file_dir` (alias: `dirname`), `path_join` (alias: `join_path`), `canonicalize_path`, `is_file`, `is_symlink`

### A.7 Network (requires --allow-net)

`http_get`, `http_post`, `http_put`, `http_delete`

### A.8 System

`exec` (requires --allow-exec), `env`, `os_name`, `time` (deprecated, use `now`)

### A.9 Date/Time (Phase 42)

`timestamp`, `timestamp_from_unix`, `now`, `add_seconds`, `add_minutes`, `add_hours`, `add_days`, `diff_seconds`, `diff_days`, `diff_hours`, `format_datetime`

### A.10 Encoding (Phase 41)

`base64_encode`, `base64_decode`, `hex_encode`, `hex_decode`

### A.11 Type Operations

`type_of`, `str`, `float`, `int`

### A.12 Error Handling

`try`

### A.13 Binary / Low-Level (Phase 66)

`float_to_bits(f)` — IEEE 754 bits as float, `bits_to_float(n)` — float from IEEE 754 bits, `float_byte(f, idx)` — single byte (0-3) of IEEE 754 representation (precision-safe for values > 2^24), `write_bytes(path, array)` — write array of floats as raw bytes (Statement, requires --allow-write)

### A.14 Terminal / Output (Phase 88)

`print_raw(str)`, `print_bytes(arr)`, `to_str(n)`, `term_move_up(n)`, `term_clear()`, `term_supports_graphics()`

### A.15 Video / Animation (Phase 83e)

`video_open(byte_array)` — open GIF or AVI/MJPEG video from byte array, returns handle with `.width`, `.height`, `.frames`, `.fps`; `video_frame(handle, index)` — extract frame as GPU arrays, returns `.r`, `.g`, `.b` channels

---

## B. Naming Conventions

### B.1 Variables and Functions

**Use snake_case:**
```flow
let user_name = "Alice"
let total_count = 100.0

fn calculate_average(values)
    return sum(values) / len(values)
end
```

### B.2 Modules

**Module filenames:** lowercase, underscores
- `stdlib/math.flow`
- `stdlib/array_utils.flow`
- `myapp/data_processing.flow`

### B.3 Structs

**Use PascalCase:**
```flow
struct UserProfile(name, email, age)
struct TradeSignal(timestamp, symbol, direction, price)
```

---

## C. Common Patterns

### C.1 Pipeline Pattern (GPU)

```flow
stream data = tap("input.csv")
stream result = data
    |> subtract(mean)
    |> divide(stddev)
    |> clamp(-3.0, 3.0)
    |> multiply(255.0)
emit(result, "output.csv")
```

### C.2 Array Processing Pattern (CPU)

```flow
let data = read_csv("data.csv")
let processed = map_each(data, fn(row)
    let value = float(row["amount"])
    let adjusted = value * 1.1
    return adjusted
end)
```

### C.3 Error Handling Pattern

```flow
let r = try(risky_operation())
if r.ok == 0.0
    print("Failed: {r.error}")
    // Handle error or return
else
    let result = r.value
    // Use result
end
```

### C.4 File Processing Pattern

```flow
let files = list_dir("/data")
for file in files
    if ends_with(file, ".csv")
        let path = join_path("/data", file)
        let content = read_csv(path)
        // Process content
    end
end
```

---

## D. Anti-Patterns (What NOT to Do)

### D.1 Don't Use Floats as Booleans in Comparisons

**Bad:**
```flow
if some_value                // Confusing: any non-zero is "true"
```

**Good:**
```flow
if some_value > 0.0          // Explicit comparison
if some_value == 1.0         // Or check for exact value
```

### D.2 Don't Ignore try() Results

**Bad:**
```flow
let r = try(http_get(url))
// No check of r.ok — might use r.value when it's an error
```

**Good:**
```flow
let r = try(http_get(url))
if r.ok == 1.0
    // Use r.value
else
    // Handle r.error
end
```

### D.3 Don't Mutate Without let mut

**Bad:**
```flow
let arr = [1.0, 2.0]
push(arr, 3.0)               // ERROR: arr not mutable
```

**Good:**
```flow
let mut arr = [1.0, 2.0]
push(arr, 3.0)               // OK
```

---

## E. Version History

| Phase | Features Added |
|-------|---------------|
| 0-8 | GPU pipelines, conditionals, comparisons, print |
| 9-13 | Interpolation, source locations, parameterization, scalar fns, watch mode |
| 14-18 | Vec types, structs, arrays, mutable state, .octo format |
| 19-24 | Loops, break/continue, if blocks, user functions |
| 25-27 | RNG, strings, REPL, module system |
| 28-30 | For-each, array mutation, stdlib, hashmaps |
| 31-34 | File I/O, string ops, advanced arrays, error handling |
| 35-37 | HTTP, JSON, environment, OctoData |
| 38-39 | Closures, hashmap bracket access, structured CSV |
| 40 | Command execution (exec) |
| 41 | Statistics, path ops, base64/hex |
| 42 | Date/time operations (timestamps as floats) |
| 43 | Bitwise operators, hex literals, regex operations |
| 44 | extern FFI (raw OS dlopen/LoadLibrary, zero-dependency, --allow-ffi) |
| 45 | TCP/UDP sockets (raw WinSock2/POSIX); removed base64/serde_json/time crates |
| 46 | HTTP server (listen/accept/respond); pure HTTP/1.1 client over TCP; removed ureq crate |
| 47 | Pure NFA bytecode regex engine (regex_io.rs); removed regex crate |
| 48 | Pure PNG + JPEG codec (image_io.rs); removed image crate — zero external deps in flowgpu-cli |
| 49 | Raw Vulkan bindings (vk_sys.rs); removed ash crate — ZERO external Rust deps across all crates |
| 50-66 | enum/match, neural networks, self-hosting, SPIR-V binary emission, GPU-native identity |
| 67-68 | ExprStmt (bare function calls), function scope inheritance |
| 69 | IR Foundation: CFG+SSA in .flow, automated IR→SPIR-V, GPU dispatch validation |
| 70 | AST→IR lowering: expression parser with precedence, symbol table, if/else→SelectionMerge |
| 71 | Full automated pipeline: kparse.flow kernel parser, source→kparse→lower→ir→SPIR-V→GPU |
| 72a | Raw memory builtins (mem_alloc/free/set/get), FFI arg encoding fix, call_fn_ptr 8 args |
| 72b | Vulkan FFI dispatch in .flow: vk.flow module, 40 extern Vulkan fns, bit_and/or/test builtins, read_bytes, LIB_CACHE, ExternBlock module import fix |
| 83e | JPEG chroma subsampling (4:2:0/4:2:2/4:4:4), native GIF decoder (~250 lines), AVI/MJPEG parser (~150 lines), video_open/video_frame builtins |
| 88 | Multi-input IR (ir_load_input_at), GPU base64 kernel (b64_emit.flow), print_bytes/print_raw/to_str builtins, 4 pure .flow terminal renderers (halfblock/kitty/sixel/digits), transitive module imports, preflight circular-import guard |

---

## 29. Self-Hosting Compiler Track (Internal Phases 44-52)

A parallel track alongside runtime features: writing the OctoFlow compiler in OctoFlow itself.

### 29.3 eval.flow — Meta-Interpreter (Internal Phases 44-46) ✅

```flow
// Run any .flow program via the meta-interpreter:
FLOW_INPUT=prog.flow octoflow run stdlib/compiler/eval.flow --allow-read
```

**Architecture:** Token-walking interpreter over parallel arrays
- `tok_types[]` / `tok_values[]` — flat token stream from inline lexer
- `env_num{}` / `env_s{}` — scalar environment (float + string views)
- `__arr_name` key → pipe-delimited array encoding
- `__map_name.key` key → map entry encoding

**Coverage:** let/mut, fn, if/elif/else, while, for-each, break, continue, push, map_set, map_get, map_has, join, abs, floor, ceil, sqrt, trim, replace, starts_with, ends_with, chained concat, unary minus, arr[i]=val

**Verified:** eval.flow interprets lexer.flow → produces 1203 tokens (matches native runtime) ✓

### 29.4 parser.flow — Recursive Descent Parser (Internal Phase 47) ✅

```flow
// Parse any .flow program into an AST:
FLOW_INPUT=prog.flow octoflow run stdlib/compiler/parser.flow --allow-read
```

**AST Representation:** 9 parallel arrays (no heap allocation, purely indexed)
```
nd_kind[]   -- Let, Assign, If, Elif, Else, While, ForEach, Fn, Return, Print,
               Push, MapSet, Call, ArrSet, Use, break, continue
nd_lhs[]    -- primary name (var name, fn name, condition string, array name)
nd_op[]     -- operator or secondary key (e.g. array index for ArrSet)
nd_rhs[]    -- rhs value or first argument
nd_rhs2[]   -- third arg (for MapSet) or elif/else chain node index (for If/Elif)
nd_child[]  -- index of first child statement (-1 = none)
nd_next[]   -- index of next sibling in same block (-1 = end of block)
nd_line[]   -- source line number for diagnostics
nd_mut[]    -- "1" if let mut, "" otherwise
```

**Block tracking:** `cur_block_first[]`, `cur_block_last[]`, `block_stack[]`, `parent_stack[]` — pushed/popped at each block boundary (if/while/for/fn). Index = `parse_depth`.

**DFS walk:** Iterative using `walk_stack` + `walk_ind` arrays (LIFO). Push order: next sibling → rhs2 chain → child (child on top = visited first).

**Key pattern:** `nd_rhs2` is overloaded — stores elif/else chain link for If/Elif nodes, but data for other nodes (MapSet value, etc.). DFS must guard: only follow `nd_rhs2` as a node pointer when `wkind == "If" || wkind == "Elif"`.

**Verified:**
- `lexer.flow` → 152 nodes ✓
- `eval.flow` → 1547 nodes ✓
- `parser.flow` (self-parse) → 768 nodes ✓

### 29.5 len() / char_at() Unicode Alignment (Bug Fix, Phase 47)

**Root cause:** `len(str)` returned byte count (`s.len()` in Rust) but `char_at(str, i)` used Unicode char indexing (`s.chars().collect()`). Any source file with multi-byte Unicode chars (e.g. `→` = 3 bytes) caused `char_at()` index-out-of-bounds in the lexer's inner while loops.

**Fix:** `len()` on strings now returns `s.chars().count()` (Unicode char count) to match `char_at()`.

**Rule:** In OctoFlow, string positions are always Unicode character positions, not byte offsets. `len(s)` = char count, `char_at(s, i)` = i-th Unicode character.

---

### 29.6 preflight.flow — Static Analyzer (Internal Phase 48) ✅

```sh
FLOW_INPUT=prog.flow octoflow run stdlib/compiler/preflight.flow --allow-read
```

**Architecture:** Three-pass design over a token stream (inline lexer, identical to parser.flow's):

- **Pass 0**: Pre-scan all `fn` names → forward-reference support
- **Pass 1**: Single linear pass — registers symbols, checks mutations, tracks usage
- **Post-pass**: Emit D002/D003 for unused definitions

**Map-based symbol table** (critical design decision):

```flow
let mut sym_kind_map = map()   // name → "scalar"|"mut"|"fn"|"array"|"map"|"param"
let mut sym_line_map = map()   // name → definition line
let mut sym_used_map = map()   // name → 1.0 if referenced
```

Why maps, not arrays? OctoFlow user-defined functions capture a snapshot of outer **scalars** only — outer arrays are not accessible from fn bodies. Using `sym_find()` with a linear search over arrays would fail inside fn bodies. Maps are accessible because they decompose to scalars internally.

**Checks emitted:**

| Code | Meaning |
|------|---------|
| E001 | Assignment to undeclared variable |
| E002 | Assignment to non-`mut` scalar or function |
| E003 | Call to undefined function |
| D002 | `let` declaration never referenced (suppressed for `_`-prefix names) |
| D003 | `fn` declaration never called |

**Key patterns:**

- `_`-prefix exempts from D002: `let _debug = ...` — intentionally unused convention
- String template scanning: `{ident}` inside format strings marks `ident` as used
- Decomposed names auto-registered: `try()` → `name.value/.ok/.error`, `http_get()` → `name.status/.body/.ok/.error`
- `for item in arr` → `item` registered as "param" (suppresses D002)
- `fn name(params)` → params registered as "param" (suppresses D002)

**Milestone results (Phase 48):**

- `test_hello.flow` → OK, no issues ✓
- `lexer.flow` → OK (26 symbols, 1203 tokens) ✓
- `eval.flow` → OK (374 symbols, 12275 tokens) ✓
- `parser.flow` → OK (137 symbols, 5630 tokens) ✓
- `preflight.flow` (self-analyze) → OK (99 symbols, 4796 tokens) ✓

### 29.7 Iteration Limit: 10K → 1M (Phase 48)

**Problem:** FlowGPU's while loop limit (10,000 by default) was too restrictive for real programs. The lexer in preflight.flow needs ~62,000 iterations to process eval.flow.

**Fix:** Default limit raised from 10,000 to 1,000,000 in both top-level and function-body execution paths. The infinite-loop test now uses `overrides.max_iters = Some(100)` to stay fast.

**Rule:** Use `--max-iters N` on the CLI to override for very large programs. The 1M default handles any realistic source file.

---

### 29.8 codegen.flow — GLSL Compute Shader Emitter (Internal Phase 49) ✅

```sh
FLOW_INPUT=prog.flow octoflow run stdlib/compiler/codegen.flow --allow-read --allow-write
# Output: prog.comp  (compile with: glslc prog.comp -o prog.spv)
```

**Architecture:** Single-pass pipeline-pattern extractor → GLSL text emitter.

1. **Inline lexer** (same as parser.flow/preflight.flow)
2. **Pattern extraction**: scan tokens for `let result = src |> map(fn params: expr)`
3. **GLSL emitter**: build shader string using `NL = chr(10)` for real newlines
4. **Write**: `write_file(out_path, glsl)` emits the compute shader

**Critical pattern: `chr(10)` for newlines**

```flow
// OctoFlow has NO escape sequences (\n is literal backslash-n in strings)
let NL = chr(10)   // actual newline character
glsl = glsl + "#version 450" + NL
glsl = glsl + "void main() {" + NL
```

**Kernel detection pattern:**

```flow
let src = stream(...)                   // input buffer binding 0
let result = src |> map(fn x: x * 2.0) // output buffer binding 1
```

→ Generated GLSL:
```glsl
layout(binding = 0, std430) buffer Buf0 { float data[]; } buf_0;
layout(binding = 1, std430) buffer BufOut { float data[]; } buf_out;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    float x = buf_0.data[gid];
    float result = x * 2.0;
    buf_out.data[gid] = result;
}
```

**Two-input kernels** — multiple lambda params → multiple input buffers:

```flow
let r = a |> map(fn x, y: x + y)  // bindings 0, 1 → output 2
```

**Milestone results (Phase 49):**
- `gpu_double.flow` → valid GLSL, `float result = x * scale;` ✓
- `gpu_sigmoid.flow` → valid GLSL, `float result = 1.0 / (1.0 + exp(0.0 - x));` ✓
- Preflight analysis of codegen.flow → OK, 0 issues (79 symbols, 2971 tokens) ✓

---

### 29.9 bootstrap.flow — Three-Compiler Bootstrap (Internal Phase 50) ✅

```sh
OCTOFLOW_BIN=./target/release/flowgpu-cli.exe
octoflow run stdlib/compiler/bootstrap.flow --allow-read --allow-exec
```

**Milestone:** BOOTSTRAP VERDICT: VERIFIED — eval.flow (pure OctoFlow) produces identical output to the Rust runtime for all test programs.

**Architecture:** Uses `exec()` to run both v1 and v2 as subprocesses and compares output:

- **v1** = `exec(octoflow, "run", tpath)` → Rust runtime directly runs the program
- **v2** = `exec("powershell", "-Command", "$env:FLOW_INPUT='...'; octoflow run eval.flow")` → OctoFlow interpreter runs the program
- Filters v2 output: strips LINT report, eval headers (`--- OctoFlow Eval Done ---`), GPU messages

**Critical patterns for output comparison:**

```flow
let NL = chr(10)                          // real newline — NOT "\n" (literal backslash-n)
let lines = split(raw_output, NL)         // split on real chr(10) newlines
let vl = trim(lines[i])                   // trim strips both \r and \n (handles CRLF)
```

**Bootstrap filter:** Lines starting with LINT, ---, STATUS:, [D0, "let ", Remove, GPU:, Processed are stripped from v2 output before comparison.

**Boolean OR in filter:**

```flow
// OctoFlow's || works with 1.0/0.0 float booleans
let skip_l = starts_with(vl, "LINT") || starts_with(vl, "---") || ...
if skip_l == 0.0 && len(vl) > 0.0
  // include this line
end
```

**Test results (Phase 50):**
- `test_hello.flow` → PASS: "hello OctoFlow" ✓
- `test_fib.flow` → PASS: "fib8=21" ✓
- `test_foreach.flow` → PASS: 4-line output ✓
- `test_fn.flow` → PASS: 3-line output ✓
- parser.flow self-parse → PASS ✓
- preflight.flow self-analysis → PASS (0 issues) ✓
- codegen.flow GPU shader → PASS (GLSL generated) ✓

**Result: 7/7 PASS, 0 FAIL → BOOTSTRAP VERIFIED**

---

### 29.10 rust_audit.flow — OS-Boundary Audit (Internal Phase 51) ✅

```sh
octoflow run stdlib/compiler/rust_audit.flow --allow-read --allow-exec
```

Analyzes all 23 Rust source files and classifies each as:
- **KEEP** — OS boundary (Vulkan dispatch, file I/O, network, image codec) — must remain Rust
- **FLOW** — Already replaced or replaceable by pure OctoFlow
- **THIN** — Minimal CLI glue (arg parsing, crate re-exports)

**Result:** 18,815 of 25,290 Rust lines (74%) are FLOW-replaceable. OS boundary = 5,801 lines (23%).

**Path to thin loader:** Replace compiler.rs eval loop, preflight.rs, and flowgpu-parser with FFI calls to the OctoFlow .flow implementations. Keep Vulkan/OS layer as Rust permanently.

---

### 29.11 benchmark_gpu.flow — GPU Benchmark (Phase 52) ✅

```sh
octoflow run stdlib/compiler/examples/benchmark_gpu.flow --allow-read --allow-write
```

Demonstrates the OctoFlow GPU pipeline by computing sigmoid on 10K random floats:

```flow
stream gpu_data = tap("bench_data.csv")    // tap() resolves relative to SCRIPT dir
stream negated  = gpu_data |> negate()
stream exped    = negated  |> exp()
stream plus_one = exped    |> add(1.0)
stream sigmoid  = plus_one |> pow(-1.0)    // pow(-1) = reciprocal (x^-1 = 1/x)
let gpu_sum = sum(sigmoid)
```

**Key patterns learned:**
- `tap()` requires a **literal string** — variables not accepted (parser limitation)
- `tap()` path resolves **relative to the .flow script directory** (base_dir join)
- `write_file()` / `read_lines()` resolve relative to **process CWD** — different from `tap()`
- `reciprocal()` is NOT a MapOp — use `pow(-1.0)` instead
- Lint X004: `multiply(-1.0)` → suggest `negate()`; always prefer `negate()` for negation

**LOC comparison (the main story):**
- OctoFlow GPU pipeline: 6 lines, 0 boilerplate
- Python + CUDA (GPU): 25+ lines (kernel definition, host→device transfer overhead)

---

### 29.12 GPU Compiler Pipeline (Internal Phases 69-71)

The automated source-to-GPU pipeline compiles OctoFlow kernel source text directly to SPIR-V binary:

```
.flow source → kparse.flow → lower.flow → ir.flow → .spv binary → GPU dispatch
```

**Pipeline modules (all pure .flow, stdlib/compiler/):**

| Module | Lines | Role |
|--------|-------|------|
| `ir.flow` | ~1390 | CFG+SSA IR builder + SPIR-V binary emitter (multi-input Phase 88) |
| `lower.flow` | ~480 | AST→IR lowering: expression parser, symbol table, statement walker |
| `kparse.flow` | ~480 | Kernel-subset parser: tokenizer + recursive-descent AST builder |

**Usage:**

```flow
use ir
use lower
use kparse

let nl = chr(10.0)
let src = "let x = input [ gid ] * 3.0 + 1.0" + nl + "output [ gid ] = x" + nl
kparse_source(src)
lower_to_ir()
ir_emit_spirv("kernel.spv")
```

**Key design decisions:**
- All mutable state in arrays (not scalars) for function call propagation
- Expression parser uses Pratt-style operator precedence via recursive descent
- Parallel arrays for AST nodes (`nd_kind[]`, `nd_lhs[]`, `nd_rhs[]`, etc.)
- Block stack for nested control flow (if/elif/else, while)
- SSA symbol table maps variable names to IR instruction indices

**Supported kernel subset:**
- `let [mut] name = expr` / `name = expr`
- `output[gid] = expr` / `input[gid]` (buffer access)
- `if cond ... elif ... else ... end`
- `while cond ... end`
- Arithmetic (`+`, `-`, `*`, `/`), comparisons (`<`, `>`, `<=`, `>=`, `==`, `!=`)
- Unary negation, parenthesized expressions, function calls (`float()`, `abs()`)

### 29.13 Raw Memory Builtins (Phase 72a)

Handle-based raw memory access for constructing C structs for FFI. All require `--allow-ffi`.

**Allocation:**

| Builtin | Purpose |
|---------|---------|
| `mem_alloc(size)` | Allocate zeroed, 16-byte aligned buffer. Returns handle (f32 index). |
| `mem_free(handle)` | Deallocate. Rejects external pointers. |
| `mem_size(handle)` | Return buffer size in bytes. |

**Read/Write:**

| Builtin | Purpose |
|---------|---------|
| `mem_set_u32(h, offset, val)` | Write u32 at byte offset |
| `mem_set_f32(h, offset, val)` | Write f32 (IEEE 754 bytes) |
| `mem_set_ptr(h, offset, src_handle)` | Write pointer from handle (-1 = null) |
| `mem_get_u32(h, offset)` | Read u32 from byte offset |
| `mem_get_f32(h, offset)` | Read f32 from byte offset |
| `mem_get_ptr(h, offset)` | Read pointer, store in table, return handle |
| `mem_copy(src_h, src_off, dst_h, dst_off, nbytes)` | Copy bytes between blocks |
| `mem_set_u8(h, offset, val)` | Write single byte (Phase 72b) |
| `mem_get_u8(h, offset)` | Read single byte (Phase 72b) |
| `mem_set_u64(h, offset, val)` | Write u64 (VkDeviceSize fields) (Phase 72b) |
| `mem_get_u64(h, offset)` | Read u64 as f32 (Phase 72b) |
| `mem_from_str(s)` | Allocate null-terminated C string, return handle (Phase 72b) |

**Array-returning:**

| Builtin | Purpose |
|---------|---------|
| `read_bytes(path)` | Read binary file as byte array (for .spv loading). Requires `--allow-read`. (Phase 72b) |

**Bitwise operations (Phase 72b):**

| Builtin | Purpose |
|---------|---------|
| `bit_and(a, b)` | Bitwise AND (u32) |
| `bit_or(a, b)` | Bitwise OR (u32) |
| `bit_test(n, bit)` | Test single bit: returns 1.0 if set, 0.0 if not |
| `bit_shl(a, n)` | Shift left by n bits (u32). Returns 0 if n >= 32 |
| `bit_shr(a, n)` | Shift right by n bits (u32). Returns 0 if n >= 32 |
| `bit_xor(a, b)` | Bitwise XOR (u32) |

**Example — building a Vulkan struct:**

```flow
// VkApplicationInfo: sType=0, pNext=null, apiVersion=VK_API_VERSION_1_3
let info = mem_alloc(64.0)
let _s = mem_set_u32(info, 0.0, 0.0)       // sType = VK_STRUCTURE_TYPE_APPLICATION_INFO
let _p = mem_set_ptr(info, 8.0, -1.0)       // pNext = null
let _v = mem_set_u32(info, 48.0, 4198400.0) // apiVersion
```

**Design:** Handle 0 = null sentinel. External pointers (from FFI returns) have size=0 and cannot be freed. Arena cleanup on each `execute()` call. LIB_CACHE: DLL handles cached per thread (loaded once, freed at session end).

### 29.14 Vulkan FFI Module (Phase 72b)

`stdlib/gpu/vk.flow` provides a complete Vulkan compute dispatch API from pure .flow:

**Usage:**
```flow
use vk
let spv = read_bytes("shader.spv")
// ... create instance, device, pipeline, dispatch, readback ...
```

**Module contents:**
- `extern "vulkan-1" { ... }` — 40 Vulkan function declarations
- VK_STRUCTURE_TYPE_* constants (19 values), flag constants
- `vk_check(result, context)` — error checking helper
- `vk_find_memory_type(phys_dev, type_bits, required_flags)` — memory type scanner

**Key pattern:** Build C structs with `mem_alloc` + `mem_set_u32`/`mem_set_u64`/`mem_set_ptr` at exact byte offsets matching `#[repr(C)]` layout. Non-dispatchable Vulkan handles roundtrip as MEM_TABLE externals.

**Requires:** `--allow-ffi --allow-read`

---

*This guide will be updated with each phase. Treat it as the canonical reference for "what OctoFlow can do right now."*
