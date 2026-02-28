# OctoFlow Language Guide

## Contents

1. [Values & Types](#1-values--types)
2. [Variables](#2-variables)
3. [Operators](#3-operators)
4. [Control Flow](#4-control-flow)
5. [Functions](#5-functions)
6. [Arrays](#6-arrays)
7. [HashMaps](#7-hashmaps)
8. [Strings](#8-strings)
9. [Structs & Vectors](#9-structs--vectors)
10. [Modules & Imports](#10-modules--imports)
11. [Streams & Pipelines](#11-streams--pipelines)
12. [GPU Operations](#12-gpu-operations)
13. [File I/O & Security](#13-file-io--security)
14. [Error Handling](#14-error-handling)
15. [Lambdas & Higher-Order Functions](#15-lambdas--higher-order-functions)

## 1. Values & Types

OctoFlow has three value types:

| Type | Description | Example |
|------|-------------|---------|
| **float** | 32-bit floating point | `42.0`, `3.14`, `-1.0` |
| **string** | UTF-8 text | `"hello"`, `"OctoFlow"` |
| **map** | Key-value pairs | `map()` |

Booleans are floats: `1.0` = true, `0.0` = false.

Arrays are ordered collections that can hold floats, strings, or both.

```
let x = 42.0
let name = "OctoFlow"
let items = [1, 2, 3, 4, 5]
let mut config = map()
```

## 2. Variables

```
let x = 10              // immutable
let mut y = 20          // mutable
y = y + 1               // reassignment (mut only)
```

Immutable by default. Use `let mut` for variables that need reassignment.
Attempting to reassign an immutable variable is a compile error.

## 3. Operators

### Arithmetic
```
+  -  *  /  %
```

### Comparison
```
==  !=  <  >  <=  >=
```

### Logical
```
&&  ||  !
```

### String Concatenation
```
let greeting = "Hello, " + name + "!"
```

### String Interpolation
```
let name = "GPU"
print("Hello, {name}!")     // Hello, GPU!
print("2 + 2 = {2 + 2}")   // variables only, not expressions
```

## 4. Control Flow

### If / Elif / Else

```
if x > 10
    print("big")
elif x > 5
    print("medium")
else
    print("small")
end
```

### While Loop

```
let mut i = 0
while i < 10
    print("{i}")
    i = i + 1
end
```

### For Loop (Range)

```
for i in range(0, 10)
    print("{i}")
end
```

### For-Each (Arrays)

```
let items = [10, 20, 30]
for item in items
    print("{item}")
end
```

### Break / Continue

```
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

## 5. Functions

```
fn add(a, b)
    return a + b
end

fn greet(name)
    print("Hello, {name}!")
end

let result = add(3, 4)    // 7
greet("World")             // Hello, World!
```

Functions can return any type — floats, strings, arrays, or maps.

### Recursive Functions

```
fn fibonacci(n)
    if n <= 1
        return n
    end
    return fibonacci(n - 1) + fibonacci(n - 2)
end
```

### Array Parameters

Arrays are passed by reference. Functions can read and mutate them:

```
fn sum_array(arr)
    let mut total = 0.0
    for x in arr
        total = total + x
    end
    return total
end
```

## 6. Arrays

### Creation

```
let arr = [1, 2, 3, 4, 5]
let empty = []
let mixed = [1.0, "hello", 3.14]
let generated = range_array(0, 100)
```

### Access

```
let first = arr[0]
let last = arr[len(arr) - 1]
let length = len(arr)
```

### Mutation

```
let mut items = [10, 20]
push(items, 30)           // [10, 20, 30]
let last = pop(items)     // 30, items = [10, 20]
items[0] = 99             // [99, 20]
```

### Slicing

```
let sub = slice(arr, 1, 3)    // elements at index 1, 2
```

### Higher-Order Functions

```
let nums = [1, 2, 3, 4, 5]
let evens = filter(nums, fn(x) x % 2 == 0 end)
let doubled = map_each(nums, fn(x) x * 2 end)
let total = reduce(nums, 0, fn(acc, x) acc + x end)
let sorted = sort_by(nums, fn(a, b) b - a end)
```

## 7. HashMaps

```
let mut m = map()
m["name"] = "OctoFlow"
m["version"] = 0.82

let val = m["name"]
let has = map_has(m, "name")     // 1.0
let keys = map_keys(m)           // ["name", "version"]
let vals = map_values(m)
map_remove(m, "version")
```

## 8. Strings

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
```

### Character Operations

```
let code = ord("A")     // 65.0
let char = chr(65)       // "A"
```

## 9. Structs & Vectors

### Structs

```
struct Point(x, y)
struct Color(r, g, b)

let p = Point(3.0, 4.0)
print("{p.x}, {p.y}")     // 3, 4
```

### Built-in Vectors

```
let v2 = vec2(1.0, 2.0)
let v3 = vec3(1.0, 2.0, 3.0)
let v4 = vec4(1.0, 2.0, 3.0, 4.0)

print("{v3.x}, {v3.y}, {v3.z}")
```

## 10. Modules & Imports

```
use csv
use timeseries

let data = read_csv("prices.csv")
let closes = csv_column(data, "close")
let sma_20 = sma(closes, 20)
```

`use` brings all public functions, structs, and constants from a module
into the current scope. Modules are .flow files in the stdlib/ directory
or relative to the current file.

### Module Search Path

1. `stdlib/<name>.flow`
2. `stdlib/<domain>/<name>.flow`
3. `./<name>.flow` (relative to current file)

## 11. Streams & Pipelines

Streams are GPU-dispatched data pipelines:

```
stream prices = tap("input.csv")
stream processed = prices |> ema(0.1) |> scale(100) |> clamp(0, 100)
emit(processed, "output.csv")
```

### Pipeline Functions

Define reusable pipeline transforms:

```
fn normalize: scale(0.01) |> clamp(0, 1)
fn warm_filter: brightness(20) |> contrast(1.2) |> saturate(1.1)

stream result = tap("photo.jpg") |> warm_filter
emit(result, "output.png")
```

See [Streams Guide](streams.md) for the full list of pipe operations.

## 12. GPU Operations

GPU functions create and operate on GPU-resident arrays:

```
let a = gpu_fill(1.0, 10000000)     // 10M elements on GPU
let b = gpu_add(a, a)               // computed on GPU
let total = gpu_sum(b)              // reduction → CPU scalar
```

Data stays on the GPU between operations. Only materializes to CPU
when printed, written to disk, or iterated.

See [GPU Guide](gpu-guide.md) for the complete GPU reference.

## 13. File I/O & Security

### Permission Flags

Scripts need explicit permission for I/O:

```
$ octoflow run script.flow --allow-read --allow-write --allow-net
```

| Flag | Allows |
|------|--------|
| `--allow-read` | File read operations |
| `--allow-write` | File write operations |
| `--allow-net` | Network access (HTTP, TCP, UDP) |
| `--allow-exec` | Shell command execution |
| `--allow-ffi` | Foreign function interface |

### File Operations

```
let text = read_file("data.txt")
write_file("output.txt", text)
append_file("log.txt", "entry")
let lines = read_lines("data.txt")
let exists = file_exists("data.txt")
```

### try() for Error Handling

```
let result = try(read_file("maybe.txt"))
if result.ok == 1.0
    print(result.value)
else
    print("Error: {result.error}")
end
```

## 14. Error Handling

The `try()` function catches errors and returns a decomposed result:

```
let r = try(parse_float("not a number"))
// r.ok    = 0.0
// r.value = 0.0
// r.error = "parse error: ..."
```

Use it to handle file I/O, network calls, or any operation that might fail.

## 15. Lambdas & Higher-Order Functions

Anonymous functions use `fn(...) expr end` syntax:

```
let double = fn(x) x * 2 end
let result = double(21)    // 42

let nums = [3, 1, 4, 1, 5]
let sorted = sort_by(nums, fn(a, b) a - b end)
let big = filter(nums, fn(x) x > 2 end)
```

Lambdas capture outer variables by snapshot (value copy at creation time).
