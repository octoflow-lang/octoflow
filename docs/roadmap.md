# OctoFlow — Roadmap

**Describe it. Build it. Ship it.** — the complete vibe coding stack in one binary.

---

## What's Shipped

### v0.83 — First Public Release (Feb 19, 2026)

The language works. GPU compute, 51 stdlib modules, sandboxed execution, REPL. Windows + Linux binaries. The foundation.

**What a vibe coder can do:** Write .flow scripts, run GPU compute, process data, make HTTP requests, read/write files, parse JSON/CSV — all sandboxed, all from one binary.

### v0.84 — Terminal Graphics & Native Media

GPU-rendered images in your terminal. Native video decoding — no ffmpeg.

**What's new:** `term_image()` for Kitty/Sixel/halfblock, GIF/AVI/MP4/H.264 decoders written in pure .flow, N-body physics showcase, satellite image processing demo.

### v0.85 — GUI Windows

Draw pixels, handle keyboard and mouse, build interactive apps. Raw Win32 FFI — zero external deps.

**What's new:** `window_open()`, `window_draw()`, `window_poll()` — 11 builtins. Interactive gradient demo at 60 FPS.

### v0.86 — Stdlib Expansion

More building blocks for your AI to compose with.

**What's new:** Base64/hex encoding, A* pathfinding, Entity Component System, genetic algorithms, 2D physics engine, interactive bouncing balls demo.

### v1.0 — Loom Engine + Self-Hosted Compiler (Feb 22, 2026)

The big one. GPU runs autonomously — entire dispatch chains in a single submission. The compiler is 69% written in OctoFlow. LLM inference runs natively on your GPU.

**What a vibe coder can do:** Run Qwen 2.5 1.5B on your GPU from a .flow script. Build GPU database queries. Train neural networks. Compose 246 stdlib modules across 18 domains. Everything in one 2.8 MB binary.

**Under the hood:**
- Loom Engine: 40 GPU kernels, boot-once runtime, indirect dispatch, timeline semaphores, push constants, pipeline composition
- GPU database: SELECT/WHERE/SUM/COUNT/GROUP BY/JOIN — all as dispatch chains
- GPU neural networks: shared-memory GEMV, ReLU, full forward pass
- GPU primitives: prefix scan, histogram, bitonic sort, argmin/argmax, sliding window, SHA-256, BFS, N-body, LZ4 decompression, ECS game logic
- Self-hosted compiler: lexer, parser, preflight, evaluator, IR, 5 SPIR-V emitters (24,212 lines of .flow)
- LLM inference: GGUF loading, Q4_K dequantization on GPU, transformer forward pass

### Post-v1.0 — OctoView Browser (Active)

GPU-native web browser. Vulkan rendering, CSS engine, JavaScript (Boa), multi-tab, inspector. 11,400 lines of Rust, 24 source files. Proving OctoFlow builds real applications.

**Shipped:** HTML/CSS parsing, stylesheet cascade, selector matching, JS engine with DOM mutation, inline images, find-in-page, bookmarks, Flow Mode (syntax-highlighted .flow viewer).

### Post-v1.0 — OctoUI Widget Framework (Active, Native)

OctoFlow's built-in GUI toolkit — like tkinter to Python. GPU-rendered, written entirely in .flow, ships with the language. 21 widget types, reactive state, theming, GDI text rendering.

**Shipped:** Buttons, text input, sliders, checkboxes, dropdowns, tables, tree views, modals, scroll, flexbox layout. 82 tests passing. Ask your AI to "build a GUI app" and OctoUI is just there.

### Post-v1.0 — OctoBrain (Active, Separate Repo)

GPU-native adaptive AI brain framework. Skeleton-free neural architecture — no fixed topology. Data-driven kernel generation via Loom.

**Shipped:** Hierarchical NLP swarms, GPU batch matching, bigram/Markov models, vocabulary scaling to 335 words. 3,200 lines library, 3,500 lines tests.

---

## What's Next

### v1.1 — Reorganization & Rebranding (shipped)

**The cleanup release.** FlowGPU renamed to OctoFlow. Directory structure reorganized. GPU VM renamed to Loom Engine. Professional and consistent naming throughout.

### v1.2 — The Vibe Coder Release

**Goal:** The describe-build-iterate loop works end-to-end.

#### `octoflow chat` — The Core Experience

Talk to OctoFlow. It writes the code. You iterate. Works offline, on your GPU.

| Work Item | Description |
|-----------|-------------|
| `chat` subcommand | New entry in main.rs, prompt loop, streaming output |
| Autoregressive generation | KV cache management + token-by-token sampling (already in stdlib/llm/generate.flow) |
| System prompt + context | Embed Language Guide as system prompt, manage token budget |
| File read/write integration | LLM output -> parse -> write .flow file -> run -> capture result |
| Model packaging | Lazy download of Qwen 2.5 1.5B Q4_K on first `octoflow chat` |

**Architecture:**
- `compiler/src/chat.rs` — chat loop, prompt management, file integration
- `compiler/src/main.rs` — "chat" subcommand
- `stdlib/llm/generate.flow` — autoregressive generation (KV cache, RoPE, sampling — already working)
- `stdlib/llm/generate_streaming.flow` — streaming generation with layer eviction

#### `octoflow new` — Project Scaffolding

Templates for common use cases, embedded in the binary:

| Template | What You Get |
|----------|-------------|
| `dashboard` | CSV -> chart -> PNG |
| `script` | File processing + HTTP calls |
| `scraper` | OctoView web scraping + JSON output |
| `api` | HTTP server with JSON endpoints |
| `game` | Window + game loop + keyboard input |
| `ai` | GGUF model loading + inference |
| `blank` | Empty project |

Usage: `octoflow new dashboard my-project`

#### Error Messages v2

| Improvement | What Changes |
|-------------|-------------|
| Suggestion hints | "Did you mean 'data' instead of 'dta'?" (Levenshtein distance) |
| Source line display | Show the source line alongside the error message |
| JSON error output | `--format json` flag for machine-readable errors |

#### Language Guide v2

Condensed LLM context document (~4K tokens). Every stdlib domain with one example. Web/scraping examples. Error patterns. Becomes the system prompt for `octoflow chat`.

### v1.3 — Batteries++ & Ship It

**Goal:** Stdlib deep enough that vibe coders rarely hit "can't do that." Plus distribution.

| Domain | What to Add |
|--------|-------------|
| **Data** | DataFrame ops (filter, group, join), chart generation, HTML reports |
| **Web** | WebSocket client, cookie jar, form handling, RSS parsing |
| **Crypto** | HMAC, AES-256, JWT tokens, bcrypt |
| **Media** | PNG encode (currently decode only), audio playback, GIF encode |
| **System** | Cron scheduling, process management, env variables |
| **Game** | Simple physics (collision, gravity), sprite sheets, audio mixer |

Also in v1.3:
- `octoflow build` — package .flow app + runtime as standalone binary
- `octoflow test` — run test functions (`fn test_*` pattern)

### v1.4 — octo-llm

**Goal:** Fine-tuned model that generates .flow code better than any general-purpose LLM.

| Work Item | Description |
|-----------|-------------|
| Training data | 85K lines of .flow stdlib + examples, curated prompt/completion pairs |
| LoRA fine-tune | Qwen 2.5 1.5B with .flow specialization |
| Validation | Generated code passes `octoflow check` at >95% rate |
| Error auto-fix | `octoflow chat` detects errors, auto-generates fix, re-runs |
| Bundle weights | Ship fine-tuned model with binary (or lazy download) |

The endgame: `octoflow chat` with octo-llm generates correct .flow code on the first try for common tasks, fixes its own errors for complex tasks, and runs entirely offline on your GPU.

### Beyond v1.4

| Milestone | What You Get |
|-----------|-------------|
| **Package registry** | `octoflow install math-extra` — community modules, versioned, cached |
| **macOS/Metal backend** | OctoFlow runs on Apple Silicon GPUs |
| **ARM/aarch64** | Run on Raspberry Pi, cloud ARM instances |
| **Fractal codec** | Universal compression — GPU-accelerated, works on any structured data |
| **Homeostasis** | GPU self-regulation — VRAM pressure, thermal awareness, dispatch routing |
| **Rust elimination** | Bootstrap loader < 200 lines. OctoFlow compiles itself. |

---

## Architecture

### Codebase (as of v1.0)

| Component | Lines | Language | Purpose |
|-----------|-------|----------|---------|
| Compiler + runtime | 26,745 | Rust | Parser, interpreter, 196 builtins, Vulkan FFI |
| Vulkan runtime | 6,811 | Rust | GPU device, memory, dispatch |
| Parser | 4,077 | Rust | Hand-written recursive descent |
| Self-hosted compiler | 24,212 | .flow | Lexer, parser, preflight, eval, IR, 5 SPIR-V emitters |
| Standard library | 90,844 | .flow | 246 modules across 18 domains |
| Examples | 15,199 | .flow | 134 runnable demos |
| OctoUI (native stdlib) | 15,251 | .flow | Built-in GUI toolkit (like tkinter) |
| OctoBrain | 22,113 | .flow | GPU-native ML brain |
| OctoView Browser | 11,407 | Rust | GPU-native web browser |
| OctoView CLI | 10,205 | Rust | TUI web transpiler |
| **Total** | **202,652** | | **71% .flow, 29% Rust** |

### Tests: 836 Rust + hundreds of .flow integration tests

### Self-Hosting: 69% of the compiler is written in OctoFlow

The compiler chain — lexer, parser, preflight analyzer, evaluator, IR builder, and 5 SPIR-V emitters — is written in .flow files. The Rust layer provides the OS boundary: Vulkan GPU access, file I/O, windowing, network sockets. Zero external Rust crates.

### Loom Engine: 40 GPU Kernels

The GPU compute runtime. Records dispatch chains — sequences of kernel operations — and executes entire chains in a single `vkQueueSubmit`. No CPU round-trips between stages.

**Capabilities proven:**
- Sequential dispatch chains (boot-once)
- GPU self-scheduling (indirect dispatch)
- GPU self-synchronization (timeline semaphores)
- Dynamic shader parameters (push constants)
- Multi-shader composition (pipeline composition)
- Database queries as dispatch chains (SELECT/WHERE/GROUP BY/JOIN)
- Neural network inference (shared-memory GEMV)
- LLM inference (Q4_K dequant, transformer forward pass)
- Cryptographic hashing (SHA-256)
- Graph processing (BFS)
- Scientific simulation (N-body)
- Game logic (ECS)
- Compression (LZ4 block decompression, delta encoding)

---

## Standing Principles

1. **LLM-native first.** Every feature is evaluated by: "Can an LLM generate correct code for this on the first try?"
2. **Zero dependencies.** No new external crates. No pip, no npm, no cargo add. One binary.
3. **Self-hosting direction.** New functionality in .flow, not Rust, unless it's an OS boundary (Vulkan, sockets, filesystem).
4. **Ship the dish.** Users get working solutions. Recipes (stdlib) are there for tinkering. Raw materials (compiler) are there for builders.
5. **GPU is invisible.** Users never think about kernels, VRAM, or dispatch chains. They write simple code. OctoFlow makes it fast.

---

*Previous engineering journal (3,894 lines of phase-by-phase implementation detail) archived at `docs/archive/roadmap-engineering-journal.md`.*
