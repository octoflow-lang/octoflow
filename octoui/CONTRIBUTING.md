# Contributing to OctoUI

OctoUI is the GPU-native UI rendering engine for OctoFlow.

## Development Setup

1. Install OctoFlow compiler from [octoflow-lang/octoflow](https://github.com/octoflow-lang/octoflow)
2. Ensure Vulkan SDK is installed
3. Clone this repo

## Building Kernels

```bash
octoflow run kernels/emit_ui_clear.flow --allow-read --allow-write
octoflow run kernels/emit_ui_rect.flow --allow-read --allow-write
octoflow run kernels/emit_ui_text.flow --allow-read --allow-write
```

## Running Tests

```bash
octoflow run tests/test_tree.flow
octoflow run tests/test_kernels.flow --allow-read --allow-write --allow-ffi
octoflow run tests/test_pipeline.flow --allow-read --allow-write --allow-ffi
```

## Code Style

- All code in `.flow` files (no Rust changes)
- Follow OctoFlow conventions: parallel arrays for widget state, snapshot-safe module patterns
- Every function must have explicit `return` (even void functions return 0.0)
- Use `let _var = fn(...)` not bare `fn(...)` for function calls

## License

Apache 2.0
