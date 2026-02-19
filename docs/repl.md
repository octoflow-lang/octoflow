# REPL Reference

## Starting

```
$ octoflow repl
OctoFlow v0.82.0 — GPU-native language
GPU: NVIDIA GeForce GTX 1660 SUPER
143 stdlib modules | :help | :time
```

## Commands

| Command | Short | Description |
|---------|-------|-------------|
| `:help` | `:h` | Show all commands |
| `:vars` | `:v` | List defined variables with types and values |
| `:fns` | `:f` | List defined functions with signatures |
| `:arrays` | `:a` | List GPU arrays with size, MB, and location |
| `:streams` | `:s` | List active streams |
| `:gpu` | | GPU device info and array counts |
| `:type <name>` | `:t` | Show type and value of a variable |
| `:time` | | GPU benchmark smoke test |
| `:time <expr>` | | Benchmark any expression |
| `:load <file>` | `:l` | Load and run a .flow file |
| `:reset` | | Clear all state (variables, functions, arrays) |
| `:clear` | `:cls` | Clear screen |
| `:exit` | `:q` | Exit the REPL |

## Expressions

Bare expressions are evaluated and printed automatically:

```
> 2 + 2
4
> sqrt(144)
12
> "hello" + " world"
"hello world"
```

## Last Result: _

Every expression stores its result in `_`:

```
> 2 + 2
4
> _ * 10
40
> sqrt(_)
6.3245553
```

## Multi-line Blocks

The REPL detects incomplete blocks (`fn`, `if`, `while`, `for`) and
waits for `end`:

```
> fn square(x)
...   return x * x
... end
square defined
> square(7)
49
```

```
> if 1 > 0
...   print("yes")
... end
yes
```

## GPU Benchmark

Type `:time` with no arguments for a quick smoke test:

```
> :time
GPU benchmark (NVIDIA GeForce GTX 1660 SUPER):
  gpu_fill  1M .... 2.3ms
  gpu_add   1M .... 0.5ms
  gpu_sum   1M .... 0.4ms
GPU is ready. Use :time <expr> to benchmark anything.
```

Benchmark any expression:

```
> :time gpu_fill(1.0, 10000000)
  [14.2 ms]
> :time 2 + 2
4
  [0.001 ms]
```

## Loading Files

```
> :load mylib.flow
Loaded mylib.flow [2.3 ms]
> my_function(42)
84
```

All definitions from the loaded file (functions, variables, structs)
become available in the REPL session.

## Arrays and GPU Memory

```
> let a = gpu_fill(1.0, 10000000)
> let b = gpu_add(a, a)
> :arrays
  a    10,000,000  38.1 MB  gpu
  b    10,000,000  38.1 MB  gpu
  total: 2 arrays, 20,000,000 elements, 76.3 MB
```

## Exiting

Any of these work:
- `:exit`, `:quit`, `:q`
- `exit`, `quit`
- Ctrl+D (EOF)
- Ctrl+C (graceful)
