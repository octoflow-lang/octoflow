# Getting Started with OctoFlow

Get from zero to running GPU code in under 5 minutes.

---

## 1. Install

Download from **[GitHub Releases](https://github.com/octoflow-lang/octoflow/releases/latest)**.

### Windows

1. Download `octoflow-v1.2.0-x86_64-windows.zip`
2. Unzip and add the folder to your PATH

### Linux

1. Download `octoflow-v1.2.0-x86_64-linux.tar.gz`
2. Extract and move to PATH:
   ```bash
   tar xzf octoflow-v1.2.0-x86_64-linux.tar.gz
   sudo cp octoflow/octoflow /usr/local/bin/
   ```

### Verify

```bash
octoflow --version
# OctoFlow v1.2.0 (Vulkan backend, GPU detected: NVIDIA RTX 4090)
```

If no GPU is detected, OctoFlow falls back to CPU automatically. Everything still works.

---

## 2. Hello World

Create a file called `hello.flow`:

```flow
let name = "OctoFlow"
print("Hello from {name}!")
```

Run it:

```bash
octoflow run hello.flow
```

Output:

```
Hello from OctoFlow!
```

**Key things to notice:**
- `print()` takes a string literal with `{var}` interpolation — never `print(name)`
- No semicolons, no braces — just clean syntax
- All blocks end with `end`

---

## 3. Your First GPU Pipeline

Create `gpu_demo.flow`:

```flow
// Generate 1 million random values on the GPU
let data = gpu_random(1000000)

// Scale every value by 100
let scaled = gpu_scale(data, 100.0)

// Compute statistics — all on GPU
let avg = gpu_mean(scaled)
let total = gpu_sum(scaled)
let lo = gpu_min(scaled)
let hi = gpu_max(scaled)

print("Mean:  {avg}")
print("Sum:   {total}")
print("Range: [{lo}, {hi}]")
```

Run it:

```bash
octoflow run gpu_demo.flow
```

Output:

```
Mean:  50.0184
Sum:   50018432.0
Range: [0.0012, 99.9987]
```

**Key things to notice:**
- Each GPU operation gets its own `let` — never nest GPU calls
- Data stays on the GPU between operations (no CPU round-trip)
- No setup, no imports, no drivers — it just works

### GPU Matrix Multiply

```flow
let m = 64
let k = 128
let n = 32
let a = gpu_random(m * k)
let b = gpu_random(k * n)
let c = gpu_matmul(a, b, m, n, k)
let total = gpu_sum(c)
print("Matmul ({m}x{k}) * ({k}x{n}) sum: {total}")
```

---

## 4. Chat Mode — Describe It, Build It

This is OctoFlow's signature feature. Instead of writing code, describe what you want.

```bash
octoflow chat
```

You'll see:

```
OctoFlow v1.2 — Chat Mode (type :help for commands)

>
```

Type a description:

```
> Sort an array of numbers and print each one

let data = [42.0, 17.0, 93.0, 5.0, 68.0, 31.0]
let sorted = sort_array(data)
for item in sorted
    print("{item}")
end

[Running...]
5.0
17.0
31.0
42.0
68.0
93.0
```

Try a GPU task:

```
> Compute the dot product of two 1000-element vectors on the GPU

let a = gpu_fill(2.0, 1000)
let b = gpu_fill(3.0, 1000)
let product = gpu_mul(a, b)
let dot = gpu_sum(product)
print("Dot product: {dot}")

[Running...]
Dot product: 6000.0
```

### Multi-Turn Conversations

Chat mode remembers context. Build on previous results:

```
> Now make the vectors random instead of filled

let a = gpu_random(1000)
let b = gpu_random(1000)
let product = gpu_mul(a, b)
let dot = gpu_sum(product)
print("Random dot product: {dot}")

[Running...]
Random dot product: 250.4317
```

### Auto-Repair

If the generated code has an error, OctoFlow automatically feeds the error back to the LLM and retries (up to 3 attempts). You don't need to do anything.

### Chat Commands

| Command | What It Does |
|---------|-------------|
| `:help` | Show available commands |
| `:clear` | Clear conversation history |
| `:undo` | Undo last generation |
| `:diff` | Show what changed from last version |
| `:edit` | Edit generated code before running |
| `:quit` | Exit chat mode |

### Using with an API

```bash
# Local GGUF model (default)
octoflow chat

# Any OpenAI-compatible API
octoflow chat --api http://localhost:8080

# Enable web search tools
octoflow chat --allow-net --web-tools
```

---

## 5. Web Scraping

OctoFlow has built-in web search and page reading. Create `search.flow`:

```flow
// Search the web
let results = web_search("Rust programming language")
print("Top results:")
print("")

// Print first 3 results
for i in range(0, 3)
    let title = results[i].title
    let url = results[i].url
    print("{i}: {title}")
    print("   {url}")
    print("")
end

// Read the first result
let page = web_read(results[0].url)
let words = split(page.text, " ")
let count = len(words)
print("First page has {count} words")
```

Run with network permission:

```bash
octoflow run search.flow --allow-net
```

**Note:** `web_read()` only accepts `http://` and `https://` URLs. For local files, use `read_file()` with `--allow-read`.

---

## 6. File Processing

Create `process.flow`:

```flow
// Read a CSV file
let data = read_csv("./data/sales.csv")
let amounts = map_each(data, fn(r) r["amount"] end)

// Calculate statistics
let total = reduce(amounts, 0.0, fn(acc, x) acc + x end)
let avg = mean(amounts)
let count = len(amounts)

print("Sales Summary")
print("  Total:   {total}")
print("  Average: {avg}")
print("  Count:   {count}")

// Write a report
let report = "Total: " + str(total) + "\nAverage: " + str(avg)
write_file("./output/report.txt", report)
print("Report written to ./output/report.txt")
```

Run with file permissions:

```bash
octoflow run process.flow --allow-read=./data --allow-write=./output
```

Permissions are scoped to specific directories — the program can only read from `./data` and write to `./output`.

---

## 7. Build and Share

Bundle a multi-file program into a single distributable `.flow` file:

```bash
# If your program uses `use csv` or `use "helper"`
octoflow build my_program.flow -o bundle.flow

# List dependencies
octoflow build my_program.flow --list

# Run the bundle
octoflow run bundle.flow --allow-read --allow-write
```

The bundled file contains all imports inlined — anyone can run it with just the `octoflow` binary.

---

## 8. Error Handling

OctoFlow uses `try()` for error handling:

```flow
let r = try(read_file("config.json"))
if r.ok == 1.0
    let config = json_parse(r.value)
    let host = map_get(config, "host")
    print("Host: {host}")
else
    print("Config not found, using defaults")
    print("Error: {r.error}")
end
```

HTTP requests also return result structs:

```flow
let r = http_get("http://api.example.com/data")
if r.ok == 1.0
    let data = json_parse(r.body)
    print("Status: {r.status}")
else
    print("Request failed: {r.error}")
end
```

**Always check `.ok == 1.0`** before using `.value` or `.body`.

---

## 9. Structs and Functions

```flow
struct Point(x, y)

fn distance(a, b)
    let dx = a.x - b.x
    let dy = a.y - b.y
    return sqrt(dx * dx + dy * dy)
end

let p1 = Point(0.0, 0.0)
let p2 = Point(3.0, 4.0)
let d = distance(p1, p2)
print("Distance: {d}")
```

Output:

```
Distance: 5.0
```

---

## Quick Reference

### Types

| Type | Example | Notes |
|------|---------|-------|
| `int` | `42` | 64-bit integer (i64) |
| `float` | `3.14` | 32-bit float (f32) |
| `string` | `"hello"` | UTF-8, `{var}` interpolation |
| `array` | `[1, 2, 3]` | Mutable, mixed types allowed |
| `map` | `map()` | Key-value pairs |
| `none` | `none` | Absence of value. Check with `is_none(x)` |

`int + float` auto-promotes to `float`. Use `int(x)` and `float(x)` for explicit conversion.

### Permissions

| Flag | Grants Access To |
|------|-----------------|
| `--allow-read` | File reading (read_file, read_csv, list_dir) |
| `--allow-read=./data` | File reading, scoped to ./data |
| `--allow-write` | File writing (write_file, write_csv) |
| `--allow-net` | Network (http_get, http_post, web_search) |
| `--allow-exec` | Shell commands |

### Common Functions

```
// Math: abs(x) sqrt(x) pow(x,n) round(x) floor(x) ceil(x) random()
// String: len(s) trim(s) split(s,d) join(a,d) replace(s,old,new) contains(s,sub)
// Array: len(a) push(a,v) pop(a) filter(a,fn) map_each(a,fn) reduce(a,init,fn)
// Map: map() map_get(m,k) map_has(m,k) map_keys(m) map_set(m,k,v)
// I/O: read_file(p) write_file(p,t) read_csv(p) list_dir(p)
// GPU: gpu_random(n) gpu_fill(v,n) gpu_add(a,b) gpu_sum(a) gpu_mean(a)
//      gpu_matmul(a,b,m,n,k) gpu_scale(a,s) gpu_min(a) gpu_max(a)
// JSON: json_parse(s) json_stringify(o)
// Stats: mean(a) median(a) stddev(a) min_val(a) max_val(a)
// Time: now() now_ms() sleep(ms)
```

---

## Next Steps

- **Examples**: See the `examples/` directory for complete programs
- **Docs**: `docs/chat.md` (chat mode), `docs/permissions.md` (security model), `docs/web-builtins.md` (web tools)
- **VS Code**: Install `octoflow-0.1.0.vsix` for syntax highlighting
- **GitHub**: https://github.com/octoflow-lang/octoflow
