# sys — System Utilities

Command-line argument parsing, environment variables, platform detection,
GPU memory info, and timing.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `args` | 4 | CLI argument parsing |
| `env` | 7 | Environment variables, paths |
| `timer` | 7 | Stopwatch, benchmarking |
| `platform` | 8 | OS detection, platform paths |
| `memory` | 4 | GPU memory info |

## args

```
use sys.args
```

| Function | Description |
|----------|-------------|
| `parse_args()` | Parse command-line arguments to map |
| `arg_flag(args, name)` | Get flag value (1.0/0.0) |
| `arg_value(args, name, default)` | Get named argument with default |
| `arg_required(args, name)` | Get required argument (errors if missing) |

```
let args = parse_args()
let verbose = arg_flag(args, "--verbose")
let output = arg_value(args, "--output", "result.csv")
```

## env

```
use sys.env
```

| Function | Description |
|----------|-------------|
| `get_env(name)` | Get environment variable |
| `get_env_or(name, default)` | Get with fallback default |
| `get_home()` | User home directory |
| `get_cwd()` | Current working directory |
| `get_path()` | System PATH |
| `get_user()` | Current username |
| `get_temp_dir()` | Temp directory path |

```
let home = get_home()
let user = get_user()
print("Hello, {user}! Home: {home}")
```

## timer

```
use sys.timer
```

| Function | Description |
|----------|-------------|
| `stopwatch()` | Current time in milliseconds |
| `elapsed(start)` | Milliseconds since start |
| `elapsed_secs(start)` | Seconds since start |
| `benchmark(label, iterations)` | Start benchmark timer |
| `benchmark_end(t0, label, iterations)` | End benchmark and print results |
| `sleep_busy(ms)` | Busy-wait sleep |
| `format_duration(ms)` | Format milliseconds as human-readable |

```
let t0 = stopwatch()
// ... do work ...
let ms = elapsed(t0)
print("Took {ms} ms")
```

## platform

```
use sys.platform
```

| Function | Description |
|----------|-------------|
| `get_os()` | OS name string |
| `is_windows()` | 1.0 if Windows |
| `is_linux()` | 1.0 if Linux |
| `is_mac()` | 1.0 if macOS |
| `home_dir()` | Home directory path |
| `temp_dir()` | Temp directory path |
| `user_name()` | Current username |
| `path_separator()` | Path separator (`/` or `\`) |

## memory

```
use sys.memory
```

| Function | Description |
|----------|-------------|
| `gpu_mem_total()` | Total GPU memory |
| `gpu_mem_used()` | Used GPU memory |
| `gpu_device_name()` | GPU device name string |
| `format_bytes(n)` | Human-readable byte size |
