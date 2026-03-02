# OctoFlow Core (L0)

## Syntax
```
let x = 42.0              // immutable
let mut y = 0.0            // mutable → y = y + 1

fn add(a, b)
    return a + b
end

if x > 0
    print("positive")
elif x == 0
    print("zero")
else
    print("negative")
end

for i in range(0, 10)
    print("{i}")
end

while y < 100
    y = y + 1
end

struct Point(x, y)
let p = Point(3.0, 4.0)

use math                   // import module
```

## Types
float (f32): `42.0` | int (i64): `42` | string: `"hello {name}"` | array: `[]` | map: `map()` | none
Booleans: 1.0=true, 0.0=false. No null — use none. int+float→float.
Operators: + - * / % == != < > <= >= && || !

## Core Builtins
print("text {var}") println("text") input("prompt")
len(x) type_of(x) int(x) float(x) str(x) is_none(x)
abs(x) sqrt(x) pow(x,n) round(x) floor(x) ceil(x) random()

## Arrays & Maps
push(arr, val) pop(arr) slice(arr, s, e) sort_array(arr)
filter(arr, fn(x) cond end) map_each(arr, fn(x) expr end)
reduce(arr, init, fn(acc, x) expr end)
let mut m = map() | m["key"] = val | map_get(m, "key") | map_keys(m)

## Error Handling
let r = try(expr)          // r.ok r.value r.error
if r.ok == 1.0
    print("{r.value}")
end

## Rules
- All blocks end with `end`. No braces. No semicolons.
- print() ONLY takes string literals: `print("{x}")` never `print(x)`
- GPU calls in own `let`: `let c = gpu_add(a, b)` never nested
- Functions not methods: `push(arr, x)` not `arr.push(x)`
- Run: `octoflow run file.flow --allow-read --allow-write --allow-net`

## Domains
Relevant module details will be provided automatically.
Focus on understanding what the user wants.

Available: gpu, web, media, ml, stats, science, gui, data, crypto, db, devops, sys, terminal, string, collections, compiler, ai, loom
