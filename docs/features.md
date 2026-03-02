# OctoFlow Feature Status (v1.4.0)

Last updated: February 28, 2026

## Core

| Feature | Status | Notes |
|---|---|---|
| GPU compute (Vulkan) | **Stable** | Any GPU vendor, 102 kernels, CPU fallback |
| Loom Engine (GPU VM) | **Stable** | 5 SSBOs, indirect dispatch, timeline semaphores |
| f16 GPU paths | **Stable** | 50% VRAM savings, auto-detected |
| macOS GPU (MoltenVK) | **Stable** | Apple Silicon (M1-M4) via MoltenVK |
| Stream pipelines | **Stable** | `tap()`, `emit()`, `\|>` operator |
| Integer type (i64) | **Stable** | Auto-promotion: int+float=float |
| `none` value | **Stable** | `is_none()`, JSON null interop |
| Modulo operator (%) | **Stable** | int%int=int, float%float=float |

## AI & Chat

| Feature | Status | Notes |
|---|---|---|
| `octoflow chat` (local) | **Stable** | Qwen3-1.7B Q5_K_M, GPU-accelerated |
| `octoflow chat` (API) | **Stable** | OpenAI-compatible endpoints |
| Auto-repair loop | **Stable** | Structured errors, up to 3 retries |
| Context engine (L0-L3) | **Stable** | 169 files, 18 domains, auto-loaded |
| GBNF grammar constraints | **Stable** | 54 rules, enforced during generation |
| Persistent memory | **Stable** | `~/.octoflow/memory.json`, cross-session |
| Project config (OCTOFLOW.md) | **Stable** | Like `.eslintrc` for LLM instructions |
| Thinking mode (`--think`) | **Stable** | Qwen3 extended reasoning |
| ReAct web tools | **Stable** | `SEARCH:` + `READ:`, max 3 per turn |
| LLM inference (GGUF) | **Stable** | Q4_K/Q5_K dequant, GPU matvec |

## Tools & Integration

| Feature | Status | Notes |
|---|---|---|
| MCP server | **Stable** | 10 tools, JSON-RPC 2.0 over stdio |
| `--output json` (agent mode) | **Stable** | Structured JSON envelope |
| `--stdin-as` (data piping) | **Stable** | Inject stdin as named variable |
| `octoflow build` (bundler) | **Stable** | Single-file .flow bundling |
| `octoflow check` | **Stable** | Preflight validation, 69 error codes |
| `octoflow update` | **Stable** | Self-update from GitHub Releases |
| VS Code extension | **Stable** | Syntax highlighting, ~90 builtins |
| REPL | **Stable** | 13 commands, persistent history (~/.octoflow/repl_history) |
| Permissions (Deno-style) | **Stable** | `--allow-read/write/net/exec=PATH` |

## Media

| Feature | Status | Notes |
|---|---|---|
| Image decode (PNG/JPEG/GIF/BMP) | **Stable** | Pure implementation, no external deps |
| Image encode (BMP) | **Stable** | `write_image()` builtin |
| Audio (WAV PCM) | **Stable** | Read/write |
| TTF font parsing | **Stable** | Glyph extraction |
| Terminal graphics | **Stable** | Kitty/Sixel/halfblock protocols |
| Video (MP4 frame extraction) | **Alpha** | H.264 I-frames only, no playback |
| GIF encode | Planned | v1.4 |
| Audio playback | Planned | v1.4 |

## Network

| Feature | Status | Notes |
|---|---|---|
| HTTP/1.1 client | **Stable** | `http_get()`, `http_post()`, etc. |
| HTTP/1.1 server | **Stable** | `http_serve()`, configurable CORS |
| `web_search()` | **Stable** | DuckDuckGo via curl |
| `web_read()` | **Stable** | Page content extraction via curl |
| HTTPS | **Not shipped** | Silently fails; curl used as bridge for web_search/web_read |
| WebSocket | Planned | v1.4 |
| Domain-scoped permissions | Specced | `--allow-net=api.example.com` |

## GUI

| Feature | Status | Notes |
|---|---|---|
| stdlib/gui toolkit | **Stable** | **Windows only**, 16 widget types, 3 layout systems (vstack/hstack/grid), software-rendered |
| Canvas drawing | **Stable** | Lines, rects, circles, filled circles, pixels |
| Charting (plot.flow) | **Stable** | Line, scatter, bar, heatmap, candlestick; autoscale, grid, crosshair |
| Buffer visualization | **Stable** | Image, heatmap, waveform, histogram display on canvas |
| Themes | **Stable** | Dark, light, ocean presets; custom via `_tc[]` array |
| Cross-platform GUI | Not shipped | macOS/Linux return errors |

## Data

| Feature | Status | Notes |
|---|---|---|
| CSV read/write | **Stable** | Quoted fields, headers |
| JSON parse/encode | **Stable** | Full roundtrip, null=none |
| GGUF tensor files | **Stable** | Read/extract for LLM weights |
| OctoDB + LoomDB | **Beta** | OctoDB (structured data, CRUD, indexing, .odb persistence) + LoomDB (GPU-resident computed-data log, vector similarity search, async flush to OctoDB) |

## Build & Tooling

| Feature | Status | Notes |
|---|---|---|
| `octoflow build` | **Stable** | Bundles .flow imports into single file |
| `octoflow test` | Planned | `fn test_*` pattern matching |
| LSP server | Planned | Diagnostics + go-to-def |
| Formatter | Not planned | |
| Debugger | Planned | v1.5+ |
| Package registry | Planned | v1.5+ |

## Platforms

| Platform | Status | Notes |
|---|---|---|
| Windows x64 | **Stable** | Primary development platform |
| Linux x64 | **Stable** | All features except GUI |
| macOS aarch64 | **Stable** | Apple Silicon via MoltenVK, no GUI |
