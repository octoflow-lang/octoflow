# Contributing to OctoFlow

## Reporting Issues

Open an issue on GitHub with:
- OctoFlow version (`octoflow --version`)
- OS and GPU info
- Minimal `.flow` script that reproduces the problem
- Expected vs actual behavior

## Contributing Code

The compiler source is not currently public. Contributions are welcome for:
- **stdlib modules** (`.flow` files in `stdlib/`)
- **examples** (`.flow` files in `examples/`)
- **documentation** (`docs/`)
- **bug reports and feature requests**

## Running Tests

```bash
octoflow test stdlib/       # run stdlib tests
octoflow test examples/     # run example tests
```

## Style

- `.flow` files: 2-space indent, snake_case for functions and variables
- Keep examples self-contained and runnable
- Include comments explaining non-obvious GPU operations
