# Changelog

All notable changes to OctoFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.4.0] - 2026-03-02

Codename: **"The Library"**

### Added

#### Standard Library Expansion (57 new modules)
- **algo/**: sort, search, pathfinding (A*, manhattan), geometry, graph algorithms
- **math/**: matrix, vector, complex numbers, probability, noise, fractals, interpolation
- **collections/**: stack, queue, heap, trie, skip list, ring buffer, LRU cache, bitset
- **string/**: regex, format, diff, fuzzy match
- **formats/**: GGUF parser, JSON
- **data/**: CSV, pipeline, transform, validate

#### Multimedia Stack (18 modules)
- **Audio DSP**: PCM buffers, oscillator (sine/square/saw/tri/noise/FM), ADSR envelope, FFT (Cooley-Tukey radix-2, STFT)
- **Audio FX**: biquad filters (LPF/HPF/BPF/notch), delay line, Schroeder reverb, distortion, bitcrush, tremolo, chorus, compressor
- **Audio Mixer**: multi-track, per-track volume/pan, mute/solo, stereo mixdown, master gain
- **WAV Encoder**: RIFF PCM writer (8/16-bit, mono/stereo)
- **Image Transform**: nearest/bilinear resize, rotation (90/180/270 + arbitrary), scale, translate
- **Color Spaces**: RGB to/from YUV, YCbCr, XYZ, LAB, CMYK + Delta-E distance
- **Video Timeline**: tracks, clips, transitions (cut/fade/dissolve/wipe), keyframes (lerp/step/smooth)
- **Players**: audio_player, image_viewer, video_player, playlist, media_player
- **Editors**: audio_editor, image_editor, video_editor

#### LoomView v0.1
- GPU visualization toolkit: 10 renderers, data fingerprinting, palette system
- 3 recipes: heat diffusion, wave propagation, N-body gravity

#### OctoSearch
- GPU-first full-text search with BM25 scoring
- Index persistence via OctoPress `.ocp` files
- Self-hosted BM25 kernel via IR builder

### Fixed
- 5 compiler language fixes (R-01 through R-05): implicit return, auto-promote arrays on push, inline array literal args, nested import scope, mutable scalar writeback
- 14 edge-case hardening fixes across 9 files
- probability.flow normal_cdf wrong erf argument
- matrix_ext.flow eigenvalue Rayleigh quotient

## [1.3.0] - 2026-03-01

Codename: **"The Foundation"**

### Added

#### Loom Engine Phase 4A — Adaptive Computing Fabric
- JIT Adaptive Kernels: IR builder with 80+ ops, runtime SPIR-V emission
- On-Demand Main Looms: park/unpark, auto_spawn, VM reuse
- Mailbox IPC: ring buffer inter-VM communication
- OctoPress: analysis + raw/delta/fractal encode/decode + `.ocp` persistence + GPU fractal compression
- CPU Thread Pool: `loom_threads`, `async_read`, `await` builtins

#### Algorithm Space Exploration Engine (ASE)
- Genome create/randomize/read, evolution (selection + crossover + mutation), fitness evaluation
- 3 demos: sort discovery, bitwise circuit evolution, hash parameter evolution
- Swarm Sort: multi-VM cooperative sorting (721 lines, 8-gene genome)

#### GPU ML
- 17 GPU-accelerated ML functions in `stdlib/ml/gpu_ml.flow`

### Changed
- Support Loom threading: SPIR-V file cache, non-blocking fence, queue mutex, batch pacing, async present
- Array builtins: `extend(dest, src)`, `array_copy()`, `array_extract()`

## [1.2.0] - 2026-02-28

Codename: **"The Ship"**

### Added

#### Chat Mode (`octoflow chat`)
- LLM-powered code generation from natural language descriptions
- Auto-repair loop: structured errors fed back to LLM, up to 3 fix attempts
- Multi-turn conversation with 8-message memory window
- Streaming token output during generation
- Local GGUF model support and OpenAI-compatible API mode
- ReAct tool-use: LLM can issue `SEARCH:` and `READ:` commands with `--web-tools`
- System prompt (CONTRACT.md) covering full syntax, 200+ builtins, 20 patterns
- Execution sandbox: chat-generated code scoped to cwd, no network by default, 1M iteration limit
- Chat commands: `:undo`, `:diff`, `:edit`, `:clear`, `:help`

#### Integer Type
- `42` is now `int` (i64), `42.0` is `float` (f32)
- Arithmetic auto-promotion: `int + float = float`
- Integer division is exact: `7 / 2 = 3`
- `int()` and `float()` conversion builtins
- `type_of()` returns `"int"`, `"float"`, `"string"`, `"array"`, `"map"`, `"none"`

#### `none` Value
- `none` keyword for first-class null representation
- `is_none(x)` builtin
- JSON `null` converts to `none` and back
- Map lookups return `none` for missing keys

#### Modulo Operator
- `%` operator at multiplicative precedence
- `int % int = int`, `float % float = float`

#### Web Builtins
- `web_search(query)` — returns array of `{title, url, snippet}` results
- `web_read(url)` — returns `{title, text, headings, links}` page content
- Gated on `--allow-net` permission
- Uses DDG HTML search and curl for HTTPS

#### Scoped Permissions
- `--allow-read=path` — restrict file reads to specific directory
- `--allow-write=path` — restrict file writes to specific directory
- `--allow-net=host` — restrict network access to specific host
- `--allow-exec=path` — restrict process execution to specific binary
- Bare flags (`--allow-read`) allow unrestricted access for that category

#### `octoflow build` — Single-File Bundler
- `octoflow build main.flow -o bundle.flow` bundles all imports
- Recursive import tracing with topological sort
- Circular import detection
- `use` declaration stripping in output
- `--list` flag shows dependency tree

#### Structured Error Output
- `--format json` emits machine-readable JSON errors
- 69 error codes (E001-E092) with per-code fix suggestions
- Fields: `code`, `message`, `line`, `suggestion`, `context`
- Preflight errors also emit structured JSON

#### CPU Fallback
- `gpu_matmul` and all GPU operations fall back to CPU when no GPU is available
- Startup message: `[note] No GPU detected — using CPU fallback`
- `sort()` and `gpu_sort()` builtins added

#### VS Code Extension
- TextMate grammar for `.flow` syntax highlighting
- 90+ builtin keywords, string interpolation, code folding
- All operators and control flow keywords
- Install: `code --install-extension octoflow-0.1.0.vsix`

### Changed

- `print()` parameter validation now reports E011 instead of generic parse error
- Loom VM error messages improved across 16 call sites with clearer diagnostics
- GPU matmul validates matrix dimensions before dispatch
- Chat mode uses `max_tokens=512` (was unbounded)
- Multiline input support in chat and REPL

### Fixed

- GPU matmul dimension mismatch no longer panics (returns descriptive error)
- Regex evaluation step limit prevents ReDoS on pathological patterns
- JSON parser depth limit prevents stack overflow on deeply nested input
- String escape sequences `\n`, `\t`, `\r`, `\\`, `\"`, `\0` fully validated

### Security

- Recursion depth guard: `MAX_RECURSION_DEPTH = 50` prevents stack overflow
- Regex ReDoS prevention with step limit on evaluation
- JSON nesting depth limit on parsed input
- Execution sandbox for chat-generated code (scoped I/O, no network, iteration limit)
- 28 hardening tests with 0 failures

## [1.1.0] - 2026-02-25

### Added
- Loom Engine: GPU compute runtime (40+ kernels, async dispatch, JIT)
- Self-hosting compiler (69% .flow — lexer, parser, eval, preflight, codegen, IR)
- GUI toolkit (16 widget types)
- Media codecs (BMP, GIF, H.264, AVI, MP4, TTF, WAV)
- LLM inference stack (GGUF parser, transformer, sampling, tokenizer)
- Terminal graphics (halfblock, kitty, sixel, digits)
- 246 stdlib modules across 18 domains

## [1.0.0] - 2026-02-23

### Added
- GPU VM: general-purpose virtual machine with 5-SSBO memory model
- GPU VM: register-based inter-instance message passing
- GPU VM: homeostasis regulator, indirect dispatch, HOST_VISIBLE polling
- GPU VM: dormant VM activation, I/O streaming
- DB engine: columnar storage with fused query kernels
- DB compression: delta encoding, dictionary lookup, Q4_K dequantization
- 51 stdlib modules
- Security sandbox with --allow-read, --allow-write, --allow-net, --allow-exec
- REPL with GPU support
- 658 tests passing

### Fixed
- Uninitialized vector UB in device.rs (P0)
- Missing read permission checks in load_file() (P1)

## [0.83.1] - 2026-02-19
- Initial public release
- GPU-native language with Vulkan compute backend
- 51 stdlib modules
- Windows x64 and Linux x64 binaries
