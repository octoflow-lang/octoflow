# Changelog

## [1.2.0] - 2026-02-28

### Added
- **Integer type**: `42` is `int` (i64), `42.0` is `float` (f32), auto-promotion on mixed ops
- **Modulo operator**: `x % 2` works with both int and float at multiplicative precedence
- **`none` value**: First-class null representation with `is_none()` builtin
- **Web builtins**: `web_search(query)` and `web_read(url)` for web-connected programs
- **ReAct tool-use in chat**: LLM issues SEARCH/READ commands to gather context before generating code
- **`octoflow build`**: Bundle multi-file projects into a single self-contained `.flow` file
- **`octoflow new`**: Project scaffolding with 7 built-in templates (dashboard, api, gpu-compute, etc.)
- **Scoped permissions**: `--allow-read=path`, `--allow-write=path`, `--allow-net=host`, `--allow-exec=path`
- **VS Code extension**: TextMate grammar with ~90 builtins, string interpolation, folding
- **CPU fallback**: All GPU operations work on CPU-only machines, including `gpu_matmul`
- **`sort` / `gpu_sort` builtins**: Array sorting with NaN handling
- **Structured error output**: `--format json` emits machine-readable errors with 69 error codes (E001-E092)
- **Error catalog**: Per-code fix suggestions for all 69 error codes
- **GPU status message**: Startup note when running in CPU-only mode
- **Improved Loom VM errors**: 16 call sites with clearer diagnostics and recovery suggestions
- Chat UX: streaming output, 8-message memory, multiline input, `:undo`/`:diff`/`:edit` commands

### Security
- Recursion depth guard: `MAX_RECURSION_DEPTH = 50`
- Regex ReDoS prevention: step limit on regex evaluation
- JSON nesting depth limit: prevents stack overflow on deeply nested input
- String escape sequence validation
- Execution sandbox: chat mode scopes I/O to cwd, denies network by default
- FFI warning on `--allow-ffi`
- GPU memory quota: `--gpu-max-mb` flag

### Stats
- Builtins: 196 → 201
- Tests: 926 → 1,017+
- Error codes: ~20 → 69

## [1.1.0] - 2026-02-25

### Added
- Loom Engine: GPU compute runtime rebrand (vm_* → loom_* aliases)
- OctoView Browser: GPU-native web browser with Vulkan rendering
- `octoflow chat`: AI coding assistant with GGUF LLM inference
- Self-hosted compiler at 69% .flow (24,212 lines in stdlib/compiler/)
- 246 stdlib modules across 18 domains

## [1.0.0] - 2026-02-23

### Added
- GPU VM: general-purpose virtual machine with 5-SSBO memory model
- GPU VM: register-based inter-instance message passing (R30→R31)
- GPU VM: homeostasis regulator (activation stability, throughput tracking)
- GPU VM: indirect dispatch — GPU self-programs workgroup counts
- GPU VM: CPU polling with HOST_VISIBLE Metrics/Control for zero-copy reads
- GPU VM: dormant VM activation without command buffer rebuild
- GPU VM: I/O streaming — CPU streams data to Globals, GPU processes without restart
- DB engine: columnar storage in GPU memory with fused query kernels
- DB compression: delta encoding, dictionary lookup, Q4_K dequantization
- DB compression: compressed query chains (decompress → WHERE → aggregate), bit-exact
- 51 stdlib modules (collections, data, db, ml, science, stats, string, sys, time, web, core)
- Security sandbox with --allow-read, --allow-write, --allow-net, --allow-exec
- REPL with GPU support
- 658 tests passing (648 Rust + 10 .flow integration)

### Fixed
- Uninitialized vector UB in device.rs (P0)
- Missing read permission checks in load_file() (P1)
- 12 manual div_ceil reimplementations replaced with .div_ceil() (P1)
- Deduplicated video_open/video_frame handlers (-172 lines) (P2)

## [0.83.1] - 2026-02-19
- Initial public release
- GPU-native language with Vulkan compute backend
- 51 stdlib modules
- Windows x64 and Linux x64 binaries
