# Contributing to OctoFlow

## Reporting Issues

Open an issue on GitHub with:
- OctoFlow version (`octoflow --version`)
- OS and GPU info
- Minimal `.flow` script that reproduces the problem
- Expected vs actual behavior

## Contributing Code

The compiler source is currently private. Contributions are welcome for:
- **stdlib modules** (`.flow` files in `stdlib/`)
- **examples** (`.flow` files in `examples/`)
- **documentation** (`docs/`)
- **bug reports and feature requests**

## Sustainability

OctoFlow is looking for contributors who want to help build a GPU-native language from the ground up. The compiler will be open-sourced once a sustainable development team is in place.

If you're interested in GPU computing, language design, or runtime engineering — [open an issue](https://github.com/octoflow-lang/octoflow/issues) or reach out.

## Running Tests

```bash
octoflow test stdlib/       # run stdlib tests
octoflow test examples/     # run example tests
```

## Style

- `.flow` files: 2-space indent, snake_case for functions and variables
- Keep examples self-contained and runnable
- Include comments explaining non-obvious GPU operations
