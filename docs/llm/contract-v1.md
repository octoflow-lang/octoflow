# OctoFlow Contract v2.0

You write OctoFlow code. GPU-native. All blocks end with `end`.

## Syntax

```
let x = 42.0                    // immutable
let mut y = 0.0                 // mutable
y = y + 1                       // reassign (mut only)

if x > 10
    print("big")
elif x > 5
    print("medium")
else
    print("small")
end

for i in range(0, 10)
    print("{i}")
end

for item in items
    print("{item}")
end

let mut i = 0.0
while i < 10
    i = i + 1
end

fn add(a, b)
    return a + b
end

let double = fn(x) x * 2 end

struct Point(x, y)
let p = Point(3.0, 4.0)
print("{p.x}")

use csv                         // stdlib module
use "local_file"                // local .flow file
```

Types: `int` (i64), `float` (f32), `string` (UTF-8), `map`, `array`, `none`. `42`→int, `42.0`→float. Auto-promotes: int+float→float.
Booleans: 1.0=true, 0.0=false. `none` = nothing/null. Conversion: `int(x)` `float(x)` `type_of(x)` `is_none(x)`
Operators: `+ - * / %` `== != < > <= >=` `&& || !`
Strings: `"hello {name}"` interpolates variables. Escapes: `\n \t \\ \"`.
No semicolons. No braces. No type annotations. Comments: `//`

## Built-in Functions

```
// Math
abs(x) sqrt(x) pow(x,n) round(x) floor(x) ceil(x) random()

// String
len(s) trim(s) split(s,d) join(a,d) replace(s,old,new)
contains(s,sub) index_of(s,sub) substring(s,start,len)
to_upper(s) to_lower(s) str(x) ord(c) chr(n)

// Array
len(a) push(a,v) pop(a) slice(a,s,e) sort_array(a) reverse(a) range_array(s,e)
filter(a, fn(x) cond end)
map_each(a, fn(x) expr end)
reduce(a, init, fn(acc,x) expr end)
sort_by(a, fn(a,b) a-b end)

// Map
let mut m = map()
m["key"] = "value"
map_has(m,"key") map_get(m,"key") map_keys(m) map_remove(m,"key")

// I/O (needs --allow-read/--allow-write)
read_file(path) write_file(path,text) read_lines(path) file_exists(path)
read_csv(path) write_csv(path,data) list_dir(path)

// JSON
json_parse(text) json_stringify(obj) json_parse_array(text)

// HTTP Client (needs --allow-net)
http_get(url) http_post(url,body)  // returns .ok .status .body .error

// HTTP Server (needs --allow-net)
http_listen(port) http_accept(srv) http_method(fd) http_path(fd)
http_body(fd) http_respond(fd,status,body) http_respond_json(fd,status,json)
tcp_close(fd)

// Web (needs --allow-net)
web_search(query)     // returns [{title, url, snippet}, ...]
web_read(url)         // returns {title, text, headings, links}

// Error handling
let r = try(expr)                // r.ok r.value r.error

// Stats
mean(a) median(a) stddev(a) variance(a) min_val(a) max_val(a)

// Time
now() now_ms() sleep(ms)

// GPU (data stays on GPU between ops)
gpu_fill(val,n) gpu_range(s,e,step) gpu_random(n)
gpu_add(a,b) gpu_sub(a,b) gpu_mul(a,b) gpu_div(a,b)
gpu_scale(a,s) gpu_sqrt(a) gpu_exp(a) gpu_log(a)
gpu_sum(a) gpu_min(a) gpu_max(a) gpu_mean(a)
gpu_matmul(a,b,m,n,k) gpu_where(cond,a,b) sort(a) gpu_sort(a)

// GUI
window_open(w,h,title) window_close() window_alive() window_poll()
window_draw(r,g,b) window_event_x() window_event_y() window_event_key()

// Encoding
base64_encode(s) base64_decode(s) hex_encode(s)

// Streams (pipeline syntax)
// stream name = tap("file") |> stage(args)
// emit(stream, "output")
```

## Critical Rules

1. `print()` takes ONLY a string literal: `print("x is {x}")`. Never `print(x)` or `print("a" + b)`.
2. GPU calls must be in `let` declarations. Never nest: `let c = gpu_add(gpu_fill(1.0,N), a)` is WRONG. Do `let a = gpu_fill(1.0,N)` then `let c = gpu_add(a, b)`.
3. To update a GPU array in a loop, rebind with `let`: `let arr = gpu_add(arr, delta)` NOT `arr = gpu_add(arr, delta)`.
4. `http_get`/`http_post` return a struct with `.ok`, `.status`, `.body`, `.error` fields.
5. `try(expr)` returns a struct with `.ok`, `.value`, `.error` fields. Check `.ok == 1.0` before using `.value`.
6. Arrays are mutable. Scalars need `let mut` for reassignment.
7. Run with permissions: `octoflow run file.flow --allow-read --allow-write --allow-net`

## DON'T (Common Mistakes)

- DON'T `print(x)` or `print("a" + b)` → DO `print("{x}")` (string literal with {var} only)
- DON'T `if x > 0 { ... }` or use `;` → DO `if x > 0 ... end` (no braces, no semicolons)
- DON'T `let result = gpu_sum(gpu_add(a,b))` → DO `let c = gpu_add(a,b)` then `let s = gpu_sum(c)`
- DON'T `true`/`false`/`null` → DO `1.0`/`0.0`/`none` (use `none` not `null`)
- DON'T `items.append(x)` or `arr.push(x)` → DO `push(arr, x)` (functions, not methods)

## Patterns

**1. Read CSV + Stats**
```
let data = read_csv("data.csv")
let scores = map_each(data, fn(r) r["score"] end)
let avg = mean(scores)
let sd = stddev(scores)
print("Mean: {avg}, StdDev: {sd}")
```

**2. GPU Array Pipeline**
```
let a = gpu_fill(1.0, 1000000)
let b = gpu_fill(2.0, 1000000)
let c = gpu_add(a, b)
let total = gpu_sum(c)
print("Sum: {total}")
```

**3. HTTP API + JSON**
```
let r = http_get("https://api.example.com/data")
if r.ok == 1.0
    let data = json_parse(r.body)
    let name = map_get(data, "name")
    print("Got: {name}")
else
    print("Error: {r.error}")
end
```

**4. File Processing Loop**
```
let files = list_dir("./data")
for f in files
    let text = read_file("./data/" + f)
    let upper = to_upper(text)
    write_file("./output/" + f, upper)
end
```

**5. Error Handling**
```
let r = try(read_file("config.json"))
if r.ok == 1.0
    let config = json_parse(r.value)
    let host = map_get(config, "host")
    print("Host: {host}")
else
    print("Using defaults: {r.error}")
end
```

**6. Build JSON + Write**
```
let mut user = map()
user["name"] = "Alice"
user["score"] = 95.0
let json = json_stringify(user)
write_file("user.json", json)
```

**7. Filter + Transform Data**
```
let data = read_csv("sales.csv")
let big = filter(data, fn(r) r["amount"] > 100 end)
let names = map_each(big, fn(r) r["customer"] end)
let result = join(names, ", ")
print("Big sales: {result}")
```

**8. GPU Matrix Multiply**
```
let m = 64
let k = 128
let n = 32
let a = gpu_random(m * k)
let b = gpu_random(k * n)
let c = gpu_matmul(a, b, m, n, k)
let total = gpu_sum(c)
print("Matmul sum: {total}")
```

**9. HTTP Server**
```
let srv = http_listen(8080)
print("Listening on :8080")
for i in range(0, 10)
    let fd = http_accept(srv)
    let method = http_method(fd)
    let path = http_path(fd)
    let body = http_body(fd)
    if path == "/api"
        http_respond_json(fd, 200.0, body)
    else
        http_respond(fd, 200.0, "Hello from OctoFlow!")
    end
end
tcp_close(srv)
```

**10. Statistical Analysis**
```
let data = [23.1, 45.7, 12.3, 67.8, 34.5, 89.2, 56.4]
let avg = mean(data)
let med = median(data)
let sd = stddev(data)
let lo = min_val(data)
let hi = max_val(data)
print("Mean={avg} Median={med} StdDev={sd} Min={lo} Max={hi}")
```

**11. String Building + Regex**
```
let text = read_file("log.txt")
let lines = split(text, "\n")
let errors = filter(lines, fn(l) contains(l, "ERROR") end)
let count = len(errors)
print("Found {count} errors")
for e in errors
    print("{e}")
end
```

**12. CSV Generation**
```
let mut rows = []
for i in range(0, 100)
    let mut row = map()
    row["id"] = i
    row["value"] = random() * 100
    push(rows, row)
end
write_csv("output.csv", rows)
print("Wrote 100 rows")
```

**13. GPU + Stats Pipeline**
```
let data = gpu_random(1000000)
let scaled = gpu_scale(data, 100.0)
let avg = gpu_mean(scaled)
let total = gpu_sum(scaled)
let lo = gpu_min(scaled)
let hi = gpu_max(scaled)
print("GPU stats: mean={avg} sum={total} min={lo} max={hi}")
```

**14. Struct + Functions**
```
struct Rect(x, y, w, h)

fn area(r)
    return r.w * r.h
end

fn contains_point(r, px, py)
    if px >= r.x && px <= r.x + r.w
        if py >= r.y && py <= r.y + r.h
            return 1.0
        end
    end
    return 0.0
end

let r = Rect(10.0, 20.0, 100.0, 50.0)
let a = area(r)
print("Area: {a}")
```

**15. Streams Pipeline**
```
stream prices = tap("prices.csv")
stream result = prices |> ema(0.1) |> scale(100) |> clamp(0, 100)
emit(result, "smoothed.csv")
```

**16. Map Iteration**
```
let mut scores = map()
scores["alice"] = 95.0
scores["bob"] = 82.0
scores["carol"] = 91.0
let keys = map_keys(scores)
for k in keys
    let v = map_get(scores, k)
    print("{k}: {v}")
end
```

**17. Timer / Benchmark**
```
let t0 = now_ms()
let data = gpu_random(10000000)
let s = gpu_sum(data)
let elapsed = now_ms() - t0
print("GPU sum of 10M elements: {s}")
print("Time: {elapsed}ms")
```

**18. Multi-File Data Pipeline**
```
use csv
let files = list_dir("./reports")
let mut all = []
for f in files
    let rows = read_csv("./reports/" + f)
    for r in rows
        push(all, r)
    end
end
let total = reduce(map_each(all, fn(r) r["revenue"] end), 0.0, fn(a,x) a+x end)
print("Total revenue across all files: {total}")
```

**19. Encoding + Crypto**
```
let text = "Hello, OctoFlow!"
let encoded = base64_encode(text)
let decoded = base64_decode(encoded)
print("Base64: {encoded}")
print("Decoded: {decoded}")
let hex = hex_encode(text)
print("Hex: {hex}")
```

**20. Window/GUI Basics**
```
let win = window_open("My App", 800, 600)
let mut running = 1.0
while running == 1.0
    let ev = window_poll(win)
    // update state based on events
    // render pixels array
    window_draw(win, pixels)
end
window_close(win)
```
