# OctoFlow Standard Library

246 modules across 18 domains. All written in `.flow`. Ships with every OctoFlow binary.

```flow
use csv           // use <module> to import
use "ml/nn"       // or use "domain/module" for full path
```

## Domains

| Domain | Modules | What's Inside |
|--------|---------|---------------|
| **ai** | 8 | embed, generate, genetic, inference, sampling, tokenizer, transformer, weight_loader |
| **collections** | 5 | collections, graph, heap, queue, stack |
| **compiler** | 27 | lexer, parser, eval, codegen, ir, spirv_emit, preflight, bootstrap |
| **crypto** | 3 | encoding, hash, random |
| **data** | 6 | csv, io, json, pipeline, transform, validate |
| **db** | 3 | core, query, schema |
| **devops** | 5 | config, fs, log, process, template |
| **formats** | 1 | gguf (11 quantization types) |
| **gui** | 13 | gui_core, widgets, layout, canvas, plot, theme, render, window, buffer_view |
| **llm** | 27 | generate, chat, sampling, tokenizer, transformer, inference, stream, decompose_gguf |
| **media** | 12 | bmp, gif, h264, avi, mp4, wav, ttf, image_filter, image_util |
| **ml** | 9 | nn, regression, classify, cluster, tree, ensemble, metrics, preprocess, linalg |
| **science** | 7 | calculus, physics, signal, matrix, optimize, interpolate, constants |
| **stats** | 7 | descriptive, distribution, correlation, hypothesis, risk, timeseries, math_ext |
| **string** | 3 | string, regex, format |
| **sys** | 6 | args, env, memory, platform, timer, test |
| **terminal** | 5 | halfblock, kitty, sixel, digits, render |
| **web** | 4 | http, server, json_util, url |

## Highlights

- **Self-hosted compiler** (69% .flow) — lexer, parser, eval, codegen, SPIR-V emitter
- **GPU LLM inference** — load GGUF models, run Qwen/LLaMA/Gemma/Phi on Vulkan
- **Native media codecs** — BMP, GIF, H.264, AVI, MP4, WAV, TTF (encode + decode)
- **GUI toolkit** — 16 widgets, layout engine, canvas, charts, themes
- **Terminal graphics** — halfblock, Kitty, Sixel protocols
- **ML primitives** — neural nets, regression, KNN, K-means, decision trees, ensembles

## Usage

```flow
// Import by domain
use csv
let data = read_csv("sales.csv")

// Import specific module
use "ml/nn"
let model = nn_create([784, 128, 10])

// GPU operations are builtins (no import needed)
let a = gpu_random(1000000)
let b = gpu_scale(a, 2.0)
let total = gpu_sum(b)
```

## License

Apache 2.0 — see [LICENSE-STDLIB](../LICENSE-STDLIB).
