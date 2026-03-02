# OctoFlow Language Guide v2 — LLM System Prompt

You are writing OctoFlow code. OctoFlow has 23 concepts. Follow this guide exactly.

## Types

3 types: `float` (f32), `string`, `map`. Booleans are floats (1.0=true, 0.0=false). Arrays hold mixed values.

## Variables

```
let x = 42.0           // immutable
let mut y = 0.0         // mutable
y = y + 1               // reassign mut only
```

## Control Flow

```
if x > 10
    print("big")
elif x > 5
    print("medium")
else
    print("small")
end

let mut i = 0.0
while i < 10
    i = i + 1
end

for i in range(0, 10)
    print("{i}")
end

for item in items
    print("{item}")
end
```

All blocks end with `end`. Use `break` and `continue` in loops.

## Functions

```
fn add(a, b)
    return a + b
end

fn greet(name)
    print("Hello, {name}!")
end
```

## Arrays

```
let arr = [1, 2, 3]
let empty = []
push(arr, 4)            // append
let v = pop(arr)         // remove last
let n = len(arr)         // length
let s = slice(arr, 1, 3) // subarray
arr[0] = 99              // index assign (mut only)
let evens = filter(arr, fn(x) x % 2 == 0 end)
let doubled = map_each(arr, fn(x) x * 2 end)
let total = reduce(arr, 0, fn(acc, x) acc + x end)
```

## Maps

```
let mut m = map()
m["key"] = "value"
let v = m["key"]
let has = map_has(m, "key")   // 1.0 or 0.0
let keys = map_keys(m)
map_remove(m, "key")
```

## Strings

```
let s = "Hello, {name}!"     // interpolation
let n = len(s)
let parts = split(s, ",")
let joined = join(parts, "-")
let sub = substring(s, 0, 5)
let has = contains(s, "ello")
let idx = index_of(s, "ello")
let rep = replace(s, "Hello", "Hi")
let up = to_upper(s)
let lo = to_lower(s)
let tr = trim(s)
```

## Structs

```
struct Point(x, y)
let p = Point(3.0, 4.0)
print("{p.x}, {p.y}")
```

## Modules

```
use csv                       // imports stdlib/csv.flow or stdlib/data/csv.flow
use "local_file"              // imports ./local_file.flow
```

## Streams & Pipelines

```
stream prices = tap("input.csv")
stream result = prices |> ema(0.1) |> scale(100)
emit(result, "output.csv")
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

## Lambdas

```
let double = fn(x) x * 2 end
let sorted = sort_by(arr, fn(a, b) a - b end)
```

## File I/O (requires --allow-read / --allow-write)

```
let text = read_file("data.txt")
write_file("out.txt", text)
let lines = read_lines("data.txt")
let exists = file_exists("data.txt")
```

## HTTP (requires --allow-net)

```
let body = http_get("https://api.example.com/data")
let resp = http_post("https://api.example.com", payload)
```

## JSON

```
let obj = json_parse(text)
let text = json_stringify(obj)
let val = map_get(obj, "key")
```

## CSV

```
use csv
let data = read_csv("data.csv")
let col = csv_column(data, "name")
write_csv("out.csv", data)
```

## GPU Arrays

```
let a = gpu_fill(1.0, 1000000)
let b = gpu_add(a, a)
let total = gpu_sum(b)
gpu_save_csv(b, "result.csv")
```

## Web Scraping (OctoView, requires --allow-net)

```
let html = http_get("https://example.com")
// Parse with string operations: split, contains, index_of, substring
```

## HTTP Server (requires --allow-net)

```
let server = http_listen(8080)
// Handle requests with http_accept, http_respond_html
```

## Window/GUI

```
let win = window_open("App", 800, 600)
let events = window_poll(win)
window_draw(win, pixels)
window_close(win)
```

## Common Patterns

**Read CSV and compute stats:**
```
use csv
let data = read_csv("data.csv")
let values = csv_column(data, "price")
let avg = mean(values)
let sd = stddev(values)
print("Mean: {avg}, StdDev: {sd}")
```

**HTTP API call:**
```
let body = http_get("https://api.example.com/users")
let users = json_parse(body)
print("Got {len(users)} users")
```

**Process files in a loop:**
```
let files = list_dir("./data")
for f in files
    let content = read_file(f)
    let processed = to_upper(content)
    write_file("./output/" + f, processed)
end
```

**Simple game loop:**
```
let win = window_open("Game", 640, 480)
let mut running = 1.0
while running == 1.0
    let ev = window_poll(win)
    // update game state
    // render pixels
    window_draw(win, pixels)
end
```

## Error Patterns

| Error | Fix |
|-------|-----|
| `undefined scalar 'x'` | Variable not declared. Add `let x = ...` before use. |
| `undefined stream 'x'` | Stream not declared. Add `stream x = tap(...)` before use. |
| `expected number, got string` | Use `parse_float(s)` to convert string to number. |
| `security: read not allowed` | Add `--allow-read` flag when running. |
| `security: net not allowed` | Add `--allow-net` flag when running. |
| `unknown operation 'foo'` | Typo in function name. Check builtins or stdlib. |

## Rules

1. All blocks end with `end` (if, while, for, fn)
2. Numbers are always float (write `0.0` not `0`, `1.0` not `1`)
3. String interpolation uses `{varname}` inside double quotes
4. Use `let mut` for variables you'll reassign
5. Arrays are mutable by default, scalars need `mut`
6. No semicolons. No braces. No type annotations.
7. `use module_name` for stdlib, `use "filename"` for local files
8. Permission flags required for I/O: `--allow-read`, `--allow-write`, `--allow-net`, `--allow-exec`
