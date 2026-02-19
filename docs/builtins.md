# Builtins Reference

OctoFlow has 200+ built-in functions available without imports.

## Math

| Function | Description |
|----------|-------------|
| `abs(x)` | Absolute value |
| `sqrt(x)` | Square root |
| `pow(x, n)` | x raised to power n |
| `exp(x)` | e^x |
| `log(x)` | Natural logarithm |
| `sin(x)` | Sine (radians) |
| `cos(x)` | Cosine (radians) |
| `floor(x)` | Round down |
| `ceil(x)` | Round up |
| `round(x)` | Round to nearest |

```
let x = sqrt(144.0)     // 12.0
let y = pow(2.0, 10.0)  // 1024.0
let z = abs(-5.0)       // 5.0
```

## String

| Function | Description |
|----------|-------------|
| `len(s)` | String length |
| `trim(s)` | Remove leading/trailing whitespace |
| `to_upper(s)` | Uppercase |
| `to_lower(s)` | Lowercase |
| `contains(s, sub)` | 1.0 if contains substring |
| `starts_with(s, pre)` | 1.0 if starts with prefix |
| `ends_with(s, suf)` | 1.0 if ends with suffix |
| `index_of(s, sub)` | First index of substring, -1.0 if not found |
| `char_at(s, idx)` | Character at index |
| `repeat(s, n)` | Repeat string n times |
| `substr(s, start, len)` | Substring extraction |
| `replace(s, old, new)` | Replace all occurrences |
| `split(s, delim)` | Split into array |
| `join(arr, delim)` | Join array into string |
| `ord(c)` | Character code point (e.g., ord("A") = 65.0) |
| `chr(n)` | Character from code point (e.g., chr(65) = "A") |
| `str(val)` | Convert value to string |
| `float(val)` | Convert string to float |
| `int(val)` | Convert to integer (truncate) |
| `read_line()` | Read line from stdin |

```
let s = "Hello, World!"
let upper = to_upper(s)        // "HELLO, WORLD!"
let parts = split(s, ", ")     // ["Hello", "World!"]
let joined = join(parts, " - ") // "Hello - World!"
```

## Array

| Function | Description |
|----------|-------------|
| `len(arr)` | Array length |
| `first(arr)` | First element |
| `last(arr)` | Last element |
| `push(arr, val)` | Append element (mutates) |
| `pop(arr)` | Remove and return last |
| `find(arr, val)` | Index of value, -1.0 if not found |
| `reverse(arr)` | Reversed copy |
| `slice(arr, start, end)` | Sub-array |
| `sort_array(arr)` | Sorted copy |
| `unique(arr)` | Remove duplicates |
| `range_array(start, end)` | Generate integer range |

```
let mut arr = [10, 20, 30]
push(arr, 40)
let idx = find(arr, 20)    // 1.0
let sub = slice(arr, 1, 3) // [20, 30]
```

## Higher-Order Functions

| Function | Description |
|----------|-------------|
| `filter(arr, fn(x) cond end)` | Keep elements where cond is truthy |
| `map_each(arr, fn(x) expr end)` | Transform each element |
| `reduce(arr, init, fn(acc, x) expr end)` | Fold/accumulate |
| `sort_by(arr, fn(a, b) expr end)` | Sort by comparison function |

```
let nums = [1, 2, 3, 4, 5]
let evens = filter(nums, fn(x) x % 2 == 0 end)      // [2, 4]
let doubled = map_each(nums, fn(x) x * 2 end)        // [2, 4, 6, 8, 10]
let total = reduce(nums, 0, fn(acc, x) acc + x end)   // 15
```

## Statistics

| Function | Description |
|----------|-------------|
| `mean(arr)` | Arithmetic mean |
| `median(arr)` | Median value |
| `stddev(arr)` | Standard deviation |
| `variance(arr)` | Variance |
| `quantile(arr, q)` | q-th quantile (0.0-1.0) |
| `correlation(a, b)` | Pearson correlation |
| `min_val(arr)` | Minimum value in array |
| `max_val(arr)` | Maximum value in array |
| `dot(a, b)` | Dot product |
| `norm(arr)` | L2 norm |
| `normalize(arr)` | Unit vector |
| `mat_transpose(a, r, c)` | Transpose matrix |

```
let data = [10, 20, 30, 40, 50]
let avg = mean(data)        // 30.0
let std = stddev(data)      // 15.811...
let q75 = quantile(data, 0.75) // 40.0
```

## HashMap

| Function | Description |
|----------|-------------|
| `map()` | Create empty hashmap |
| `map_has(m, key)` | 1.0 if key exists |
| `map_get(m, key)` | Value for key |
| `map_keys(m)` | Array of all keys |
| `map_remove(m, key)` | Remove key |
| `len(m)` | Number of keys |

```
let mut m = map()
m["name"] = "OctoFlow"
m["version"] = 0.82
let has = map_has(m, "name")    // 1.0
let keys = map_keys(m)          // ["name", "version"]
```

## Type

| Function | Description |
|----------|-------------|
| `type_of(val)` | Returns "float", "string", or "map" |
| `float(val)` | Convert to float |
| `int(val)` | Convert to integer float |
| `str(val)` | Convert to string |

## File I/O

Requires `--allow-read` and/or `--allow-write`.

| Function | Description |
|----------|-------------|
| `read_file(path)` | Read entire file as string |
| `read_lines(path)` | Read file as array of lines |
| `read_bytes(path)` | Read file as array of byte values |
| `read_csv(path)` | Read CSV as array of maps |
| `write_file(path, text)` | Write string to file |
| `append_file(path, text)` | Append string to file |
| `write_csv(path, data)` | Write array of maps to CSV |
| `write_bytes(path, arr)` | Write byte array to file |
| `file_exists(path)` | 1.0 if file exists |
| `file_size(path)` | File size in bytes |
| `is_file(path)` | 1.0 if regular file |
| `is_dir(path)` | 1.0 if directory |
| `is_directory(path)` | Alias for is_dir |
| `is_symlink(path)` | 1.0 if symlink |
| `list_dir(path)` | Directory listing as array |

```
let text = read_file("data.txt")
let lines = read_lines("log.txt")
write_file("output.txt", "Hello!")
```

## Path

| Function | Description |
|----------|-------------|
| `join_path(parts...)` | Join path components |
| `dirname(path)` | Parent directory |
| `basename(path)` | File name only |
| `file_dir(path)` | Alias for dirname |
| `file_name(path)` | Alias for basename |
| `file_ext(path)` | File extension |
| `canonicalize_path(path)` | Absolute canonical path |

## JSON

| Function | Description |
|----------|-------------|
| `json_parse(text)` | Parse JSON object to map |
| `json_parse_array(text)` | Parse JSON array |
| `json_stringify(val)` | Convert to JSON string |

```
let data = json_parse("{\"name\": \"OctoFlow\"}")
let name = data["name"]     // "OctoFlow"
let text = json_stringify(data)
```

## Data Persistence

| Function | Description |
|----------|-------------|
| `load_data(path)` | Load .od (OctoData) file to map |
| `save_data(path, map)` | Save map to .od file |

## Date/Time

| Function | Description |
|----------|-------------|
| `now()` | Unix timestamp (seconds) |
| `now_ms()` | Milliseconds since process start |
| `time()` | Alias for now() |
| `timestamp(iso_str)` | Parse ISO 8601 to Unix timestamp |
| `format_datetime(ts, fmt)` | Format timestamp |
| `add_seconds(ts, n)` | Add seconds to timestamp |
| `add_minutes(ts, n)` | Add minutes |
| `add_hours(ts, n)` | Add hours |
| `add_days(ts, n)` | Add days |
| `diff_seconds(a, b)` | Seconds between timestamps |
| `diff_hours(a, b)` | Hours between timestamps |
| `diff_days(a, b)` | Days between timestamps |

## Regex

| Function | Description |
|----------|-------------|
| `regex_match(text, pat)` | 1.0 if matches |
| `is_match(text, pat)` | Alias for regex_match |
| `regex_find(text, pat)` | First match string |
| `regex_find_all(text, pat)` | All matches as array |
| `regex_split(text, pat)` | Split by pattern |
| `regex_replace(text, pat, rep)` | Replace all matches |
| `capture_groups(text, pat)` | Capture groups from first match |

```
let has = regex_match("hello123", "[0-9]+")  // 1.0
let nums = regex_find_all("a1b2c3", "[0-9]") // ["1", "2", "3"]
```

## Encoding

| Function | Description |
|----------|-------------|
| `base64_encode(s)` | Base64 encode |
| `base64_decode(s)` | Base64 decode |
| `hex_encode(s)` | Hex encode |
| `hex_decode(s)` | Hex decode |

## Bitwise

| Function | Description |
|----------|-------------|
| `bit_and(a, b)` | Bitwise AND |
| `bit_or(a, b)` | Bitwise OR |
| `bit_test(n, bit)` | Test if bit is set |
| `float_to_bits(f)` | Float as 32-bit integer representation |
| `bits_to_float(n)` | Integer back to float |

## System

| Function | Description |
|----------|-------------|
| `os_name()` | Operating system name |
| `env(name)` | Environment variable |
| `random()` | Random float [0.0, 1.0) |
| `type_of(val)` | Type name string |

## Network

Requires `--allow-net`.

| Function | Description |
|----------|-------------|
| `tcp_connect(host, port)` | Connect, returns fd |
| `tcp_send(fd, data)` | Send data, returns bytes sent |
| `tcp_recv(fd, max)` | Receive data as string |
| `tcp_close(fd)` | Close connection |
| `tcp_listen(port)` | Listen on port, returns fd |
| `tcp_accept(fd)` | Accept connection, returns client fd |
| `udp_socket()` | Create UDP socket |
| `udp_send_to(fd, host, port, data)` | Send UDP packet |
| `udp_recv_from(fd, max)` | Receive UDP packet |

## HTTP Client

Requires `--allow-net`. Returns decomposed struct: `.status`, `.body`, `.ok`, `.error`.

| Function | Description |
|----------|-------------|
| `http_get(url)` | GET request |
| `http_post(url, body)` | POST request |
| `http_put(url, body)` | PUT request |
| `http_delete(url)` | DELETE request |

```
let r = http_get("https://api.example.com/data")
if r.ok == 1.0
    let data = json_parse(r.body)
end
```

## HTTP Server

Requires `--allow-net`.

| Function | Description |
|----------|-------------|
| `http_listen(port)` | Listen for HTTP connections |
| `http_accept(fd)` | Accept HTTP request |
| `http_method(fd)` | Get request method |
| `http_path(fd)` | Get request path |
| `http_query(fd)` | Get query string |
| `http_body(fd)` | Get request body |
| `http_header(fd, name)` | Get header value |
| `http_respond(fd, status, body)` | Send response |
| `http_respond_json(fd, status, json)` | Send JSON response |

## Shell Execution

Requires `--allow-exec`. Returns decomposed struct: `.status`, `.output`, `.ok`, `.error`.

| Function | Description |
|----------|-------------|
| `exec(cmd, ...args)` | Execute command |

```
let r = exec("git", "status")
if r.ok == 1.0
    print(r.output)
end
```

## FFI / Memory

Requires `--allow-ffi`. Low-level memory operations for foreign function interface.

| Function | Description |
|----------|-------------|
| `mem_alloc(size)` | Allocate memory, returns handle |
| `mem_free(handle)` | Free memory |
| `mem_size(handle)` | Size of allocation |
| `mem_get_u8(h, off)` | Read u8 at offset |
| `mem_get_u32(h, off)` | Read u32 at offset |
| `mem_get_u64(h, off)` | Read u64 at offset |
| `mem_get_f32(h, off)` | Read f32 at offset |
| `mem_get_ptr(h, off)` | Read pointer at offset |
| `mem_set_u8(h, off, val)` | Write u8 |
| `mem_set_u32(h, off, val)` | Write u32 |
| `mem_set_u64(h, off, val)` | Write u64 |
| `mem_set_f32(h, off, val)` | Write f32 |
| `mem_set_ptr(h, off, src)` | Write pointer |
| `mem_copy(src, soff, dst, doff, n)` | Copy memory region |
| `mem_from_str(s)` | String to FFI memory |
| `mem_to_str(handle, len)` | FFI memory to string |

## Error Handling

| Function | Description |
|----------|-------------|
| `try(expr)` | Execute safely, returns `.value`, `.ok`, `.error` |

```
let r = try(read_file("maybe.txt"))
if r.ok == 1.0
    print(r.value)
else
    print("Error: {r.error}")
end
```

## GPU Operations

All GPU functions operate on GPU-resident arrays. Data stays on GPU
between operations. See [GPU Guide](gpu-guide.md) for details.

### Creation

| Function | Description |
|----------|-------------|
| `gpu_fill(val, n)` | Create array of n elements, all val |
| `gpu_range(start, end, step)` | Create arithmetic sequence |

### Element-wise Binary

| Function | Description |
|----------|-------------|
| `gpu_add(a, b)` | a + b |
| `gpu_sub(a, b)` | a - b |
| `gpu_mul(a, b)` | a * b |
| `gpu_div(a, b)` | a / b |

### Element-wise Unary

| Function | Description |
|----------|-------------|
| `gpu_scale(a, s)` | Multiply by scalar |
| `gpu_abs(a)` | Absolute value |
| `gpu_negate(a)` | Negate |
| `gpu_sqrt(a)` | Square root |
| `gpu_exp(a)` | Exponential |
| `gpu_log(a)` | Natural logarithm |
| `gpu_sin(a)` | Sine |
| `gpu_cos(a)` | Cosine |
| `gpu_floor(a)` | Floor |
| `gpu_ceil(a)` | Ceiling |
| `gpu_round(a)` | Round |
| `gpu_pow(a, n)` | Power |
| `gpu_clamp(a, lo, hi)` | Clamp to range |
| `gpu_reverse(a)` | Reverse order |

### Reductions

| Function | Description |
|----------|-------------|
| `gpu_sum(a)` | Sum of all elements |
| `gpu_min(a)` | Minimum element |
| `gpu_max(a)` | Maximum element |
| `gpu_mean(a)` | Arithmetic mean |
| `gpu_cumsum(a)` | Cumulative sum (prefix scan) |

### Conditional

| Function | Description |
|----------|-------------|
| `gpu_where(cond, a, b)` | Select a where cond!=0, else b |

### Matrix

| Function | Description |
|----------|-------------|
| `gpu_matmul(a, b, m, k, n)` | Matrix multiply (m x k) * (k x n) |

### Custom Compute

| Function | Description |
|----------|-------------|
| `gpu_compute(spv_path, arr)` | Load and dispatch custom SPIR-V shader |
| `gpu_info()` | GPU device information string |

## Constructors

| Syntax | Description |
|--------|-------------|
| `vec2(x, y)` | 2D vector (decomposes to .x, .y) |
| `vec3(x, y, z)` | 3D vector (.x, .y, .z) |
| `vec4(x, y, z, w)` | 4D vector (.x, .y, .z, .w) |
| `map()` | Empty hashmap |
| `struct Name(fields)` | Define struct type |
| `try(expr)` | Error-safe execution (.value, .ok, .error) |
