# Builtins Reference

OctoFlow has 440+ built-in functions available without imports.

## Math

| Function | Description |
|----------|-------------|
| `abs(x)` | Absolute value |
| `sqrt(x)` | Square root |
| `pow(x, n)` | x raised to power n |
| `exp(x)` | e^x |
| `log(x)` | Natural logarithm |
| `ln(x)` | Natural logarithm (alias) |
| `sin(x)` | Sine (radians) |
| `cos(x)` | Cosine (radians) |
| `tan(x)` | Tangent (radians) |
| `asin(x)` | Arcsine |
| `acos(x)` | Arccosine |
| `atan(x)` | Arctangent |
| `atan2(y, x)` | Two-argument arctangent |
| `floor(x)` | Round down |
| `ceil(x)` | Round up |
| `round(x)` | Round to nearest |
| `sign(x)` | -1.0, 0.0, or 1.0 |
| `fract(x)` | Fractional part (x - floor(x)) |
| `min(a, b)` | Minimum of two values |
| `max(a, b)` | Maximum of two values |
| `clamp(x, lo, hi)` | Clamp x to [lo, hi] |
| `lerp(a, b, t)` | Linear interpolation: a + (b - a) * t |
| `random()` | Random float [0.0, 1.0) |

```
let x = sqrt(144.0)     // 12.0
let y = pow(2.0, 10.0)  // 1024.0
let angle = atan2(1.0, 1.0) // 0.785...
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
| `to_str(val)` | Convert value to string (alias) |
| `float(val)` | Convert string to float |
| `to_float(s)` | Convert string to float (alias) |
| `int(val)` | Convert to integer (truncate) |
| `from_char_code(n)` | Character from code point (alias for chr) |
| `pad_left(s, width, ch)` | Left-pad string to width with ch |
| `pad_right(s, width, ch)` | Right-pad string to width with ch |
| `tokenize(s)` | Split string into word tokens |
| `format(fmt, ...)` | Format string with arguments |
| `read_line()` | Read line from stdin |

```
let s = "Hello, World!"
let upper = to_upper(s)        // "HELLO, WORLD!"
let padded = pad_left("42", 5, "0") // "00042"
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
| `flatten(arr)` | Flatten nested array |
| `zip(a, b)` | Interleave two arrays |
| `enumerate(arr)` | Array of [index, value] pairs |
| `array_new(size, val)` | Create array of size filled with val |
| `array_copy(dst, doff, src, soff, n)` | Copy n elements between arrays |
| `array_extract(arr, start, len)` | Extract sub-array (like slice) |
| `extend(dst, src)` | Append all elements of src to dst |
| `clone(arr)` | Deep copy of array |
| `min_val(arr)` | Minimum value in array |
| `max_val(arr)` | Maximum value in array |
| `sum(arr)` | Sum of all elements |
| `count(arr)` | Number of elements (alias for len) |

```
let mut arr = [10, 20, 30]
push(arr, 40)
let big = array_new(1000, 0.0)   // 1000 zeros
extend(arr, [50, 60])            // arr is now [10,20,30,40,50,60]
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
| `cosine_similarity(a, b)` | Cosine similarity between vectors |
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
| `map_values(m)` | Array of all values |
| `map_remove(m, key)` | Remove key |
| `len(m)` | Number of keys |

```
let mut m = map()
m["name"] = "OctoFlow"
m["version"] = 0.82
let vals = map_values(m)        // ["OctoFlow", 0.82]
```

## Type Checking

| Function | Description |
|----------|-------------|
| `type_of(val)` | Returns "float", "string", "map", or "array" |
| `is_none(val)` | 1.0 if value is none |
| `is_nan(val)` | 1.0 if NaN |
| `is_inf(val)` | 1.0 if infinite |
| `is_float(val)` | 1.0 if float |
| `is_str(val)` | 1.0 if string |
| `is_map(val)` | 1.0 if map |
| `is_array(val)` | 1.0 if array |
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
| `file_mtime(path)` | File modification time (unix timestamp) |
| `is_file(path)` | 1.0 if regular file |
| `is_dir(path)` | 1.0 if directory |
| `is_directory(path)` | Alias for is_dir |
| `is_symlink(path)` | 1.0 if symlink |
| `list_dir(path)` | Directory listing as array |
| `walk_dir(path)` | Recursive directory listing |
| `read_image(path)` | Read image file |
| `write_image(path, data)` | Write image file |

```
let text = read_file("data.txt")
let lines = read_lines("log.txt")
write_file("output.txt", "Hello!")
```

## Path

| Function | Description |
|----------|-------------|
| `join_path(parts...)` | Join path components |
| `path_join(parts...)` | Alias for join_path |
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
| `now_us()` | Microseconds since process start |
| `time()` | Alias for now() |
| `time_ms()` | Alias for now_ms() |
| `clock()` | High-resolution clock |
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
| `bit_xor(a, b)` | Bitwise XOR |
| `bit_test(n, bit)` | Test if bit is set |
| `bit_shl(a, n)` | Shift left by n bits |
| `bit_shr(a, n)` | Shift right by n bits |
| `float_to_bits(f)` | Float as 32-bit integer representation |
| `bits_to_float(n)` | Integer back to float |
| `float_byte(f)` | Float to byte value |

## System

| Function | Description |
|----------|-------------|
| `print(s)` | Print with interpolation and newline |
| `print_raw(s)` | Print without trailing newline |
| `print_bytes(arr)` | Print byte array as raw output |
| `sleep(seconds)` | Sleep for n seconds |
| `os_name()` | Operating system name |
| `env(name)` | Environment variable |
| `random()` | Random float [0.0, 1.0) |
| `type_of(val)` | Type name string |
| `exit()` | Exit the program |
| `assert(cond)` | Assert condition is truthy, panic if not |
| `panic(msg)` | Abort with error message |

```
print("Hello {name}")    // interpolated print
sleep(0.5)               // pause 500ms
assert(len(arr) > 0)     // abort if empty
```

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
| `socket_close(fd)` | Close socket |

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
| `http_accept(fd)` | Accept HTTP request (blocking) |
| `http_accept_nonblock(fd)` | Accept HTTP request (non-blocking) |
| `http_method(fd)` | Get request method |
| `http_path(fd)` | Get request path |
| `http_query(fd)` | Get query string |
| `http_body(fd)` | Get request body |
| `http_header(fd, name)` | Get header value |
| `http_respond(fd, status, body)` | Send response |
| `http_respond_json(fd, status, json)` | Send JSON response |
| `http_respond_html(fd, status, html)` | Send HTML response |
| `http_respond_image(fd, data)` | Send image response |
| `http_respond_with_headers(fd, status, body, headers)` | Send response with custom headers |

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
| `mem_to_str_at(handle, off, len)` | FFI memory to string at offset |
| `mem_u64_add(h, off, val)` | Atomic u64 add |
| `file_read_into_mem(path)` | Read file directly into FFI memory |
| `file_read_into_mem_u64(path)` | Read file into FFI memory (u64 variant) |

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

## Window / GUI

Requires `--allow-gui` or windowed mode.

| Function | Description |
|----------|-------------|
| `window_open(w, h, title)` | Open a window |
| `window_close()` | Close the window |
| `window_alive()` | 1.0 if window is open |
| `window_poll()` | Poll window events |
| `window_draw(r, g, b)` | Draw framebuffer to window |
| `window_width()` | Window width in pixels |
| `window_height()` | Window height in pixels |
| `window_title(title)` | Set window title |
| `window_dpi()` | Display DPI scale factor |
| `window_event_key()` | Last key event code |
| `window_event_char()` | Last character input |
| `window_event_x()` | Mouse X position |
| `window_event_y()` | Mouse Y position |
| `window_event_scroll()` | Mouse scroll delta |
| `window_event_timer_id()` | Timer event ID |
| `window_key_held(key)` | 1.0 if key is currently held |
| `window_capture_mouse()` | Capture mouse input |
| `window_release_mouse()` | Release mouse capture |
| `window_set_cursor(type)` | Set cursor type |
| `window_set_timer(id, ms)` | Start a repeating timer |
| `window_kill_timer(id)` | Stop a timer |
| `window_create_menu()` | Create a menu bar |
| `window_set_menu(menu)` | Attach menu to window |
| `menu_add_item(menu, label)` | Add menu item |
| `menu_add_submenu(menu, label)` | Add submenu |
| `gui_mouse_down()` | 1.0 if mouse button is down |
| `gui_mouse_buttons()` | Mouse button state bitmask |
| `gui_scroll_y()` | Scroll Y delta |
| `clipboard_get()` | Read clipboard text |
| `clipboard_set(s)` | Write text to clipboard |
| `dialog_open_file(title, filter)` | Open file dialog |
| `dialog_save_file(title, filter)` | Save file dialog |
| `dialog_message(title, msg, type)` | Message dialog |

```
window_open(800, 600, "My App")
while window_alive() == 1.0
    window_poll()
    let key = window_event_key()
end
window_close()
```

## GDI Text

GPU-accelerated text rendering via font atlas.

| Function | Description |
|----------|-------------|
| `gdi_text_begin()` | Start text layout session |
| `gdi_text_add(font, text)` | Add text with font size |
| `gdi_text_w(font)` | Character width for font size |
| `gdi_text_h(font)` | Character height for font size |
| `gdi_text_off()` | Current text cursor offset |
| `gdi_text_height(font)` | Line height for font size |
| `gdi_text_width(font, text)` | Pixel width of text string |
| `gdi_text_atlas()` | Get font atlas data for GPU upload |

```
gdi_text_begin()
gdi_text_add(16, "Hello World")
let atlas = gdi_text_atlas()
loom_set_heap(vm, atlas)
```

## Audio

| Function | Description |
|----------|-------------|
| `audio_play(samples, rate)` | Play PCM samples at sample rate |
| `audio_play_file(path)` | Play audio file (WAV) |
| `audio_stop()` | Stop audio playback |

## Terminal

| Function | Description |
|----------|-------------|
| `term_clear()` | Clear terminal screen |
| `term_move_up(n)` | Move cursor up n lines |
| `term_supports_graphics()` | 1.0 if terminal supports graphics |
| `term_image(data, w, h)` | Display image in terminal |
| `term_image_at(data, w, h, x, y)` | Display image at position |

## System Monitoring

| Function | Description |
|----------|-------------|
| `cpu_util()` | CPU utilization percentage |
| `cpu_count()` | Number of CPU cores |
| `nvml_init()` | Initialize NVIDIA monitoring |
| `nvml_gpu_util()` | GPU utilization percentage |
| `nvml_mem_util()` | GPU memory utilization percentage |
| `nvml_temperature()` | GPU temperature (Celsius) |
| `nvml_vram_used()` | VRAM used (bytes) |
| `nvml_vram_total()` | VRAM total (bytes) |
| `nvml_power()` | GPU power draw (watts) |
| `nvml_gpu_name()` | GPU device name string |
| `nvml_clock_gpu()` | GPU clock speed (MHz) |

```
nvml_init()
let temp = nvml_temperature()
let vram = nvml_vram_used()
print("GPU: {temp}C, VRAM: {vram} bytes")
```

## Web

Requires `--allow-net`.

| Function | Description |
|----------|-------------|
| `web_search(query)` | Search the web, returns results |
| `web_read(url)` | Read webpage content as text |

## Grammar-Constrained Decoding

For LLM inference with structured output.

| Function | Description |
|----------|-------------|
| `grammar_load(path)` | Load GBNF grammar file |
| `grammar_load_str(s)` | Load grammar from string |
| `grammar_mask(handle, logits)` | Apply grammar mask to logit array |
| `grammar_advance(token_id)` | Advance grammar state by token |
| `grammar_reset()` | Reset grammar to initial state |
| `grammar_active()` | 1.0 if grammar is loaded |

```
grammar_load("json.gbnf")
let masked = grammar_mask(handle, logits)
grammar_advance(next_token)
```

## GGUF / LLM Inference

Low-level LLM inference primitives for GGUF model format.

| Function | Description |
|----------|-------------|
| `gguf_cache_file(path)` | Open and cache GGUF model file |
| `gguf_load_tensor(name, arr)` | Load named tensor from GGUF |
| `gguf_matvec(tensor, input)` | Matrix-vector multiply with GGUF tensor |
| `gguf_infer_layer(layer_idx)` | Run inference on a transformer layer |
| `gguf_evict_layer(idx)` | Evict layer from VRAM |
| `gguf_evict_layer_ram(idx)` | Evict layer from RAM |
| `gguf_prefetch_layer(idx)` | Prefetch layer to VRAM |
| `gguf_prefetch_complete()` | 1.0 if prefetch is done |
| `gguf_tokens_per_sec()` | Current inference throughput |
| `gguf_extract_tensor_raw(name)` | Extract raw tensor data |
| `gguf_load_vocab(path)` | Load tokenizer vocabulary |
| `gguf_tokenize(text)` | Tokenize text string |
| `chat_emit_token(token)` | Emit token to chat stream |

## OctoPress

OctoFlow's native compression format for GPU-resident data.

| Function | Description |
|----------|-------------|
| `octopress_init(data)` | Initialize compressor |
| `octopress_analyze(handle)` | Analyze data statistics |
| `octopress_encode(handle)` | Compress data |
| `octopress_decode(handle)` | Decompress data |
| `octopress_gpu_encode(handle)` | GPU-accelerated compression |
| `octopress_save(handle, path)` | Save .ocp file |
| `octopress_load(path)` | Load .ocp file |
| `octopress_info(handle)` | Compression stats |
| `octopress_stream_open(path)` | Open streaming reader |
| `octopress_stream_next(handle)` | Read next chunk |
| `octopress_stream_info(handle)` | Stream metadata |
| `octopress_stream_reset(handle)` | Reset to beginning |
| `octopress_stream_close(handle)` | Close stream |

## GPU Operations

All GPU functions operate on GPU-resident arrays. Data stays on GPU
between operations. See [GPU Guide](gpu-guide.md) for details.

### Creation

| Function | Description |
|----------|-------------|
| `gpu_fill(val, n)` | Create array of n elements, all val |
| `gpu_range(start, end, step)` | Create arithmetic sequence |
| `gpu_random(n, lo, hi)` | Create array of n random values in [lo, hi) |

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
| `gpu_product(a)` | Product of all elements |
| `gpu_variance(a)` | Variance |
| `gpu_stddev(a)` | Standard deviation |
| `gpu_dot(a, b)` | Dot product |
| `gpu_count(a)` | Element count |
| `gpu_cumsum(a)` | Cumulative sum (prefix scan) |

### Transformations

| Function | Description |
|----------|-------------|
| `gpu_where(cond, a, b)` | Select a where cond!=0, else b |
| `gpu_sort(a)` | Parallel radix sort |
| `gpu_concat(a, b)` | Concatenate two arrays |
| `gpu_gather(data, indices)` | Gather elements by index |
| `gpu_scatter(vals, indices, n)` | Scatter values by index into array of size n |
| `gpu_topk(arr, k)` | Top-K values (sorted descending) |
| `gpu_topk_indices(arr, k)` | Indices of top-K values |
| `gpu_ema(arr, alpha)` | Exponential moving average |

### Matrix

| Function | Description |
|----------|-------------|
| `gpu_matmul(a, b, m, n, k)` | Matrix multiply: A is m*k, B is k*n, result is m*n |

### GPU I/O

| Function | Description |
|----------|-------------|
| `gpu_load_csv(path)` | Load CSV file directly to GPU array |
| `gpu_load_binary(path)` | Load raw f32 binary to GPU array |
| `gpu_save_csv(arr, path)` | Save GPU array to CSV file |
| `gpu_save_binary(arr, path)` | Save GPU array to raw f32 binary |

### Custom Compute

| Function | Description |
|----------|-------------|
| `gpu_compute(spv_path, arr)` | Load and dispatch custom SPIR-V shader |
| `gpu_run(spv, arr1, ...)` | Run custom SPIR-V with multiple arrays |
| `gpu_info()` | GPU device information string |
| `gpu_timer_start()` | Start GPU timer |
| `gpu_timer_end()` | End GPU timer, returns microseconds |

## Constructors

| Syntax | Description |
|--------|-------------|
| `vec2(x, y)` | 2D vector (decomposes to .x, .y) |
| `vec3(x, y, z)` | 3D vector (.x, .y, .z) |
| `vec4(x, y, z, w)` | 4D vector (.x, .y, .z, .w) |
| `map()` | Empty hashmap |
| `struct Name(fields)` | Define struct type |
| `try(expr)` | Error-safe execution (.value, .ok, .error) |
