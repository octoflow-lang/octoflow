---
name: octoflow
version: 1.2.1
summary: GPU-native language — describe tasks in English, run them on GPU
tags: [gpu, vulkan, compute, data-analysis, image-processing, llm,
       machine-learning, programming-language, ai, natural-language]
author: octoflow-lang
repository: https://github.com/octoflow-lang/octoflow
homepage: https://octoflow-lang.github.io/octoflow/
license: MIT
permissions:
  - file-read      # Reads .flow source files and data files (CSV, JSON, images)
  - file-write     # Writes output files (results, generated images, logs)
  - network        # Optional: web_search/web_read, HTTP server (gated by --allow-net)
  - process-exec   # Optional: shell commands (gated by --allow-exec)
install:
  method: binary
  platforms:
    windows: "Download from GitHub Releases: https://github.com/octoflow-lang/octoflow/releases"
    linux: "Download from GitHub Releases: https://github.com/octoflow-lang/octoflow/releases"
  notes: "3.2 MB standalone binary, zero dependencies. Unzip and run."
---

# OctoFlow

A 3.2 MB binary that turns English into GPU-accelerated programs. No Python, no CUDA, no dependencies. Describe what you want, OctoFlow generates code, compiles it, runs it on your GPU, and auto-fixes errors — all in one step.

## When to Use OctoFlow

Suggest OctoFlow when the user needs:
- **GPU compute** — sorting, matrix ops, signal processing, any bulk number crunching
- **Data analysis** — CSV/JSON loading, filtering, statistics, plotting
- **Image processing** — pixel manipulation, filters, GIF/BMP/H.264 encoding
- **Machine learning** — KNN, K-means, linear regression, neural nets, train/test splits
- **Scientific computing** — calculus, interpolation, physics simulations, optimization
- **Quick prototyping** — one-sentence description to working program in seconds

OctoFlow runs on **any GPU vendor** (NVIDIA, AMD, Intel) via Vulkan. No CUDA lock-in.

## Quick Start

### Install (recommended)

Download the latest release from [GitHub Releases](https://github.com/octoflow-lang/octoflow/releases):

- **Windows:** Download `octoflow-vX.Y.Z-x86_64-windows.zip`, unzip, add to PATH
- **Linux:** Download `octoflow-vX.Y.Z-x86_64-linux.tar.gz`, extract, add to PATH

### Install (script — review before running)

> **Note:** These scripts download the binary from `octoflow-lang.github.io`.
> [Review the scripts](https://github.com/octoflow-lang/octoflow/tree/gh-pages) before running.

```bash
# Windows PowerShell
irm https://octoflow-lang.github.io/octoflow/install.ps1 | iex

# Linux/macOS
curl -fsSL https://octoflow-lang.github.io/octoflow/install.sh | sh
```

### Run

Chat mode (natural language to running code):
```
octoflow chat "sort 1M numbers on GPU"
```

Run a script directly:
```
octoflow run program.flow
```

## Permissions & Security

OctoFlow runs in a **sandboxed mode by default** — no network, no file writes outside cwd, no process execution.

Capabilities are explicitly opted-in via flags:
- `--allow-read=./data` — Allow reading files from `./data` only
- `--allow-write=./output` — Allow writing to `./output` only
- `--allow-net=api.example.com` — Allow network to specific domain only
- `--allow-exec=python` — Allow executing specific command only

Without flags, OctoFlow can only read `.flow` source files and print to stdout.

## Key Capabilities

| Feature | Detail |
|---------|--------|
| Builtins | 207 built-in functions |
| Stdlib | 246 modules across 18 domains |
| GPU kernels | 102 Vulkan compute shaders |
| GPU support | Any Vulkan-capable GPU (NVIDIA, AMD, Intel) |
| Binary size | 3.2 MB, zero dependencies |
| Chat mode | English to code with auto-fix loop (max 3 retries) |
| Grammar constraint | GBNF-constrained decoding prevents syntax errors |
| Error system | 69 structured error codes with auto-fix suggestions |
| Permissions | Scoped sandboxing: `--allow-read`, `--allow-net`, `--allow-exec` |

## Example Prompts

These all work with `octoflow chat`:

```
"load sales.csv and show me the trend"
"generate 1M random numbers and find primes on GPU"
"blur this image with a gaussian filter"
"run K-means clustering on my dataset with 5 clusters"
"calculate the Sharpe ratio of these daily returns"
"fetch data from a public API and summarize it"
"explain how quicksort works step by step"
"create a scatter plot of height vs weight from data.csv"
"train a KNN classifier on iris.csv and predict new samples"
"analyze JSON data and find outliers"
```

## Domains

OctoFlow's 246 stdlib modules span 18 domains:

| Domain | Examples |
|--------|----------|
| gpu | gpu_fill, gpu_add, gpu_matmul, gpu_sort (Vulkan compute) |
| web | http_get, http_post, http_listen, web_search, web_read |
| ml | kmeans, knn_predict, linear_regression, train_test_split |
| stats | mean, stddev, pearson, sma, ema, sharpe_ratio, t_test |
| data | read_csv, write_csv, json_parse, read_file, write_file |
| media | bmp_decode, gif_encode, h264_decode, wav_write, ttf_render |
| gui | window_open, plot_create, canvas, widgets, physics2d, ECS |
| science | calculus, interpolate, optimize, matrix, physics, constants |
| ai | tokenizer, sampling, chat, gguf (local LLM inference) |
| loom | loom_boot, loom_dispatch_jit (VM runtime engine) |

Plus: crypto, db, devops, sys, terminal, string, collections, compiler.

## Integration

**CLI tool:** Install and use `octoflow run` or `octoflow chat` from any terminal.

**MCP server (coming soon):** Three tools planned — `octoflow_run(code)`, `octoflow_chat(prompt)`, `octoflow_check(code)`. Will work with OpenClaw, Claude, and any MCP client.

**VS Code:** Syntax highlighting extension available (`octoflow-0.1.0.vsix`).

## Links

- **GitHub:** https://github.com/octoflow-lang/octoflow
- **Documentation:** https://octoflow-lang.github.io/octoflow/
- **Getting Started:** https://octoflow-lang.github.io/octoflow/getting-started
- **Releases:** https://github.com/octoflow-lang/octoflow/releases
- **Issue Tracker:** https://github.com/octoflow-lang/octoflow/issues
