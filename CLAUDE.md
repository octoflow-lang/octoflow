# CLAUDE.md

OctoFlow is a GPU-native programming language. GPU is the primary execution target. Zero external dependencies.

## Quick Reference

- **Language guide**: `docs/language-guide.md`
- **Builtins**: `docs/builtins.md`
- **GPU guide**: `docs/gpu-guide.md`
- **Loom Engine**: `docs/loom-engine.md`
- **Stdlib**: `docs/stdlib.md`

## Running

```bash
octoflow run file.flow              # run a program
octoflow repl                       # interactive REPL
octoflow chat                       # AI code generation
octoflow test stdlib/               # run stdlib tests
```

## Style

- `.flow` files: 2-space indent, snake_case
- Keep examples self-contained and runnable
- Security: `--allow-read`, `--allow-write`, `--allow-net`, `--allow-exec`
