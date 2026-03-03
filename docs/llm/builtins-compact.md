# OctoFlow Builtins — Compact Reference

## Scalar Functions (0 args)
```
random() → f32                    // [0.0, 1.0)
time() → string                   // formatted timestamp
os_name() → string                // "windows" / "linux" / "macos"
now() → f32                       // Unix timestamp (seconds)
now_ms() → f32                    // Unix timestamp (milliseconds)
read_line() → string              // read from stdin
```

## Scalar Functions (1 arg)
```
abs(x) → f32                      sqrt(x) → f32
exp(x) → f32                      log(x) → f32
sin(x) → f32                      cos(x) → f32
floor(x) → f32                    ceil(x) → f32
round(x) → f32                    int(x) → f32 (truncate)
float(x) → f32                    str(x) → string
trim(s) → string                  to_upper(s) → string
to_lower(s) → string              type_of(x) → string
env(name) → string                ord(c) → f32
chr(n) → string                   len(s_or_arr) → f32
mean(arr) → f32                   median(arr) → f32
stddev(arr) → f32                 variance(arr) → f32
dirname(path) → string            basename(path) → string
canonicalize_path(p) → string     is_file(p) → f32
is_dir(p) → f32                   is_symlink(p) → f32
base64_encode(s) → string         base64_decode(s) → string
hex_encode(s) → string            hex_decode(s) → string
timestamp(s) → f32                timestamp_from_unix(n) → string
float_to_bits(f) → f32            bits_to_float(n) → f32
mem_alloc(size) → f32             mem_free(ptr) → f32
mem_size(ptr) → f32               mem_from_str(s) → f32
pop(arr) → value                  first(arr) → value
last(arr) → value                 min_val(arr) → value
max_val(arr) → value
```

## Scalar Functions (2 args)
```
pow(base, exp) → f32              contains(s, sub) → f32
starts_with(s, pre) → f32         ends_with(s, suf) → f32
index_of(s, sub) → f32            char_at(s, idx) → string
repeat(s, n) → string             quantile(arr, q) → f32
correlation(arr1, arr2) → f32     regex_match(text, pat) → f32
is_match(text, pat) → f32         regex_find(text, pat) → string
add_seconds(ts, n) → f32          add_minutes(ts, n) → f32
add_hours(ts, n) → f32            add_days(ts, n) → f32
diff_seconds(ts1, ts2) → f32      diff_days(ts1, ts2) → f32
diff_hours(ts1, ts2) → f32        format_datetime(ts, fmt) → string
float_byte(f, idx) → f32          bit_and(a, b) → f32
bit_or(a, b) → f32                bit_test(val, bit) → f32
mem_get_u8(ptr, off) → f32        mem_get_u32(ptr, off) → f32
mem_get_f32(ptr, off) → f32       mem_get_u64(ptr, off) → f32
mem_get_ptr(ptr, off) → f32       mem_to_str(ptr, len) → string
```

## Scalar Functions (3 args)
```
clamp(x, lo, hi) → f32            substr(s, start, len) → string
replace(s, old, new) → string     regex_replace(text, pat, rep) → string
mem_set_u8(ptr, off, val)          mem_set_u32(ptr, off, val)
mem_set_f32(ptr, off, val)         mem_set_u64(ptr, off, val)
mem_set_ptr(ptr, off, val)
```

## Scalar Functions (5 args)
```
mem_copy(dst, dst_off, src, src_off, len)
```

## Array Functions (LetDecl — must use `let x = fn(...)`)
```
read_lines(path) → [string]       list_dir(path) → [string]
split(s, delim) → [string]        read_csv(path) → [map]
read_bytes(path) → [f32]          json_parse_array(s) → [value]
regex_split(text, pat) → [string] capture_groups(text, pat) → [string]
regex_find_all(text, pat) → [string]
reverse(arr) → [value]            slice(arr, start, end) → [value]
sort_array(arr) → [value]         unique(arr) → [value]
range_array(start, end) → [f32]
filter(arr, lambda) → [value]     map_each(arr, lambda) → [value]
sort_by(arr, lambda) → [value]
```

## Scalar from Array
```
reduce(arr, init, fn(acc, x) expr end) → value
len(arr) → f32
```

## Map Functions
```
map() → empty map (LetDecl)
map_get(m, key) → value           map_set(m, key, value)
map_has(m, key) → f32             map_remove(m, key)
map_keys(m) → [string]            map_values(m) → [value]
map_size(m) → f32
json_parse(s) → map (LetDecl)     json_stringify(m) → string
```

## GPU Array Functions (LetDecl — must use `let x = gpu_fn(...)`)
```
// Create
gpu_fill(val, n)                   gpu_range(start, end, step)
gpu_random(n)

// Binary (two arrays → array)
gpu_add(a, b)     gpu_sub(a, b)    gpu_mul(a, b)     gpu_div(a, b)

// Unary/Scalar (array → array)
gpu_scale(a, s)   gpu_abs(a)       gpu_negate(a)      gpu_sqrt(a)
gpu_exp(a)        gpu_log(a)       gpu_sin(a)         gpu_cos(a)
gpu_pow(a, n)     gpu_floor(a)     gpu_ceil(a)        gpu_round(a)
gpu_clamp(a, lo, hi)               gpu_reverse(a)     gpu_cumsum(a)
gpu_ema(a, alpha)

// Conditional
gpu_where(cond, a, b)

// Matrix
gpu_matmul(a, b, m, n, k)         mat_transpose(a)   normalize(a)

// Data movement
gpu_concat(a, b)                     gpu_gather(data, indices)
gpu_scatter(values, indices, size)

// Reductions (→ scalar)
gpu_sum(a)        gpu_min(a)       gpu_max(a)         gpu_mean(a)
gpu_product(a)    gpu_variance(a)  gpu_stddev(a)      norm(a)
```

## I/O Statements
```
write_file(path, string)           // requires --allow-write
write_csv(path, array_of_maps)     // requires --allow-write
write_bytes(path, byte_array)      // requires --allow-write
```

## Decomposition Functions (LetDecl, produce .field scalars)
```
// HTTP (requires --allow-net)
let r = http_get(url)              // r.status r.body r.ok r.error
let r = http_post(url, body)       // r.status r.body r.ok r.error
let r = http_put(url, body)        // r.status r.body r.ok r.error
let r = http_delete(url)           // r.status r.body r.ok r.error

// Exec (requires --allow-exec)
let r = exec(cmd, arg1, arg2, ...) // r.status r.output r.ok r.error

// Error handling
let r = try(expr)                  // r.value r.ok r.error

// Vectors
let v = vec2(x, y)                 // v.x v.y
let v = vec3(x, y, z)             // v.x v.y v.z
let v = vec4(x, y, z, w)          // v.x v.y v.z v.w

// Structs
struct Foo { a, b }
let f = Foo(1, 2)                  // f.a f.b
```

## Pipeline Operators (for stream |> chains)
```
add(n) subtract(n) multiply(n) divide(n) negate() abs() sqrt()
exp() log() sin() cos() pow(n) clamp(lo,hi) scale(n) offset(n)
normalize() floor() ceil() round()
```
