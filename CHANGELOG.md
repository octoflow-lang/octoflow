# Changelog

All notable changes to OctoFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
