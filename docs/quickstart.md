# Quickstart

Five minutes from download to GPU compute.

## Install

Download the binary for your platform:
- [Windows x64](https://github.com/octoflow-lang/octoflow/releases/latest)
- [Linux x64](https://github.com/octoflow-lang/octoflow/releases/latest)

Unzip. Run. No installer, no dependencies, no SDK.

```
$ octoflow --version
OctoFlow 0.82.0
```

## Hello World

Create `hello.flow`:

```
print("Hello, World!")
```

Run it:

```
$ octoflow run hello.flow
Hello, World!

ok 1ms
```

## Hello GPU

Create `hello_gpu.flow`:

```
let a = gpu_fill(1.0, 1000000.0)
let b = gpu_fill(2.0, 1000000.0)
let c = gpu_add(a, b)
let total = gpu_sum(c)

print("Hello from the GPU!")
print("1,000,000 elements: 1.0 + 2.0 = {total}")
```

Run it:

```
$ octoflow run hello_gpu.flow
Hello from the GPU!
1,000,000 elements: 1.0 + 2.0 = 3000000

ok 12ms
```

1 million elements. GPU-parallel. 12 milliseconds of actual compute.

## REPL

```
$ octoflow repl
OctoFlow v0.82.0 — GPU-native language
GPU: NVIDIA GeForce GTX 1660 SUPER
143 stdlib modules | :help | :time

> 2 + 2
4
> _ * 10
40
> let a = gpu_fill(1.0, 1000000)
> :time gpu_sum(a)
1000000  [0.4 ms]
> :quit
```

## Import Modules

```
use timeseries

let prices = [100.0, 102.0, 101.5, 103.0, 104.5, 103.5, 105.0]
let sma_3 = sma(prices, 3)
let last_sma = sma_3[len(sma_3) - 1]
print("SMA(3): {last_sma}")
```

143 modules across 12 domains. Run `octoflow help <domain>` to explore:

```
$ octoflow help stats
stdlib/stats/ — Statistics & Math

  use descriptive    mean, median, stddev, skewness, kurtosis, describe
  use hypothesis     t_test, paired_t_test, chi_squared, anova, z_test
  use correlation    pearson, spearman, covariance, linear_fit
  ...
```

## Security

OctoFlow uses Deno-style permissions. Scripts can't read files, access
the network, or run commands without explicit flags:

```
$ octoflow run server.flow --allow-read --allow-net
```

## Next Steps

- [Language Guide](language-guide.md) — full syntax reference
- [Builtins Reference](builtins.md) — all built-in functions
- [GPU Guide](gpu-guide.md) — GPU computing in OctoFlow
- [REPL Reference](repl.md) — REPL commands and usage
- [Streams Guide](streams.md) — stream/pipeline syntax
- [Stdlib Reference](stdlib/) — all 12 domain modules
