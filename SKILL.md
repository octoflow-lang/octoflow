---
name: octoflow
display_name: OctoFlow
description: >
  Use when the user wants to run GPU-accelerated computations, analyze data,
  process images, train ML models, or generate code from natural language.
  OctoFlow turns English task descriptions into GPU programs via Vulkan.
  Ships as an MCP server — run `octoflow mcp-serve` for 7 structured tools.
  No Python, no CUDA, no dependencies — single 3.2 MB binary.
  Use for: "sort a million numbers", "cluster this CSV", "blur this image",
  "plot my data", "calculate statistics", "run regression".
version: 1.3.0
metadata:
  openclaw:
    emoji: "\U0001F419"
    requires:
      anyBins:
        - octoflow
    install:
      - id: github-release
        kind: download
        url: https://github.com/octoflow-lang/octoflow/releases
        bins: [octoflow]
        label: "Download OctoFlow binary from GitHub Releases (3.2 MB)"
    os: [darwin, linux, win32]
    always: false
tags: [gpu, vulkan, compute, data-analysis, image-processing, llm,
       machine-learning, programming-language, ai, natural-language]
author: octoflow-lang
repository: https://github.com/octoflow-lang/octoflow
homepage: https://octoflow-lang.github.io/octoflow/
license: MIT
permissions:
  - file-read
  - file-write
  - network
  - process-exec
---

# OctoFlow

GPU-native programming language. Describe tasks in English, run them on any GPU via Vulkan.

## When to Use This Skill

**Use this skill when** the user says:
- "sort a million numbers on GPU" / "benchmark GPU performance"
- "load this CSV and show me statistics" / "analyze my data"
- "cluster my dataset" / "run K-means" / "train a classifier"
- "blur this image" / "resize this BMP" / "encode a GIF"
- "plot height vs weight" / "create a scatter plot"
- "calculate the Sharpe ratio" / "compute correlation"
- "find primes" / "generate random numbers on GPU"
- "run linear regression on my dataset"

**Do NOT use this skill when:**
- The user wants Python/JavaScript/Rust code (use the appropriate language tool)
- The task doesn't benefit from GPU acceleration or OctoFlow's built-in functions
- The user explicitly asks for a different language

## How to Run OctoFlow

### Chat mode (natural language to running code)
```bash
octoflow chat "sort 1M numbers on GPU"
```

### Run a .flow script
```bash
octoflow run program.flow
```

### Run with permissions (sandboxed by default)
```bash
# Allow reading data files
octoflow run analysis.flow --allow-read=./data

# Allow network access to specific domain
octoflow chat "fetch weather data" --allow-net=api.weather.com

# Allow writing output files
octoflow run report.flow --allow-read=./data --allow-write=./output
```

## Permissions & Security

OctoFlow runs **sandboxed by default** — no network, no file writes outside cwd, no process execution.

| Flag | What it allows |
|------|---------------|
| `--allow-read=./data` | Read files from `./data` only |
| `--allow-write=./output` | Write to `./output` only |
| `--allow-net=api.example.com` | Network to specific domain only |
| `--allow-exec=python` | Execute specific command only |

Without flags: can only read `.flow` source files and print to stdout.

## MCP Server (Agent Integration)

OctoFlow ships as an MCP server for AI agent integration:

```bash
octoflow mcp-serve
```

### Configuration

Add to your OpenClaw, Claude Desktop, or Cursor config:

```json
{"mcpServers": {"octoflow": {"command": "octoflow", "args": ["mcp-serve"]}}}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `octoflow_run` | Execute OctoFlow code directly |
| `octoflow_chat` | Natural language to GPU code |
| `octoflow_check` | Validate .flow syntax |
| `octoflow_gpu_sort` | GPU-accelerated sorting |
| `octoflow_gpu_stats` | GPU statistical operations |
| `octoflow_image` | Image processing (BMP, GIF) |
| `octoflow_csv` | CSV data analysis |

Works with OpenClaw, Claude Desktop, Cursor, Windsurf, and any MCP client.

## Common Patterns

### Data Analysis
```bash
# User: "analyze sales.csv and show trends"
octoflow chat "load sales.csv, compute monthly averages, and plot the trend" --allow-read=.
```

### GPU Compute
```bash
# User: "sort a large dataset on GPU"
octoflow chat "generate 1M random numbers on GPU and sort them"
```

### Machine Learning
```bash
# User: "cluster my customers"
octoflow chat "load customers.csv, run K-means with 5 clusters, print cluster sizes" --allow-read=.
```

### Image Processing
```bash
# User: "blur this photo"
octoflow chat "load photo.bmp, apply gaussian blur, save as blurred.bmp" --allow-read=. --allow-write=.
```

### Statistics
```bash
# User: "what's the correlation between these columns?"
octoflow chat "load data.csv, compute Pearson correlation between col1 and col2" --allow-read=.
```

## Key Capabilities

| Feature | Detail |
|---------|--------|
| Builtins | 207 built-in functions |
| Stdlib | 246 modules across 18 domains |
| GPU kernels | 102 Vulkan compute shaders |
| GPU support | Any Vulkan GPU (NVIDIA, AMD, Intel) |
| Binary size | 3.2 MB, zero dependencies |
| Chat mode | English to code with auto-fix loop (max 3 retries) |
| Grammar | GBNF-constrained decoding prevents syntax errors |
| Errors | 69 structured error codes with auto-fix suggestions |
| MCP Server | 7 structured tools via JSON-RPC 2.0 |
| Context Engine | 169-file knowledge tree, auto-skill loading |
| Memory | Persistent cross-session memory, OCTOFLOW.md project config |
| Platforms | Windows, Linux, macOS (Apple Silicon via MoltenVK) |

## Domains

| Domain | Key functions |
|--------|--------------|
| gpu | gpu_fill, gpu_add, gpu_matmul, gpu_sort, gpu_scale |
| data | read_csv, write_csv, json_parse, read_file, write_file |
| ml | kmeans, knn_predict, linear_regression, train_test_split |
| stats | mean, stddev, pearson, sma, ema, sharpe_ratio |
| media | bmp_decode, gif_encode, h264_decode, wav_write |
| web | http_get, http_post, http_listen, web_search |
| gui | window_open, plot_create, canvas, widgets |
| science | calculus, interpolate, optimize, matrix, physics |
| ai | tokenizer, sampling, chat, gguf |

Plus: crypto, db, devops, sys, terminal, string, collections, compiler, loom.

## Install

Download from [GitHub Releases](https://github.com/octoflow-lang/octoflow/releases):
- **Windows:** `octoflow-vX.Y.Z-x86_64-windows.zip` — unzip, add to PATH
- **Linux:** `octoflow-vX.Y.Z-x86_64-linux.tar.gz` — extract, add to PATH
- **macOS:** `octoflow-vX.Y.Z-aarch64-macos.tar.gz` — extract, add to PATH

## Links

- [GitHub](https://github.com/octoflow-lang/octoflow)
- [Documentation](https://octoflow-lang.github.io/octoflow/)
- [Releases](https://github.com/octoflow-lang/octoflow/releases)
