# Permissions

OctoFlow uses a Deno-style security model: all I/O is denied by default.
Scripts cannot read files, access the network, or run shell commands
unless you explicitly grant permission with flags.

## Default Behavior

With no flags, a script can only:
- Compute (CPU and GPU)
- Print to stdout
- Read from stdin

Everything else is blocked.

## Permission Flags

### File Read

```bash
octoflow run script.flow --allow-read           # read any file
octoflow run script.flow --allow-read=./data     # read only from ./data/
```

Grants access to `read_file()`, `read_csv()`, `read_image()`, `file_exists()`,
`walk_dir()`, and all file-reading builtins.

### File Write

```bash
octoflow run script.flow --allow-write           # write any file
octoflow run script.flow --allow-write=./output   # write only to ./output/
```

Grants access to `write_file()`, `write_csv()`, `write_image()`, `append_file()`,
and all file-writing builtins.

Path traversal is blocked: `--allow-write=./output` rejects writes to
`./output/../../etc/passwd`. All paths are canonicalized before checking.

### Network

```bash
octoflow run script.flow --allow-net             # all network access
```

Grants access to `http_get()`, `http_post()`, `http_serve()`, `web_search()`,
`web_read()`, and all network builtins.

**Note:** OctoFlow's built-in HTTP client supports HTTP/1.1 only (no HTTPS).
The `web_search()` and `web_read()` builtins use curl as a bridge for HTTPS.

### Shell Execution

```bash
octoflow run script.flow --allow-exec            # run shell commands
```

Grants access to `run_shell()` and `pipe()`. Use with caution — these
execute arbitrary system commands.

## Combining Flags

```bash
octoflow run pipeline.flow \
  --allow-read=./data \
  --allow-write=./output \
  --allow-net
```

## Chat Mode

Chat-generated code runs in a sandbox with these defaults:
- File I/O scoped to current working directory
- Network denied
- Max 1,000,000 loop iterations
- No shell execution

Override with the same flags:

```bash
octoflow chat --allow-read=./data --allow-write=./output --allow-net
```

## What Happens on Denial

When a script tries an operation without permission, OctoFlow returns
an error with the denied operation and the flag needed to grant it:

```
Error: Permission denied: file read requires --allow-read
```

The script continues running — denied operations return error values,
they don't crash the program.

## Security Model

- **Principle of least privilege:** Grant only what the script needs
- **Path scoping:** `=path` restricts to a directory subtree
- **No ambient authority:** Environment variables and config files
  cannot grant permissions — only CLI flags
- **Canonicalization:** All paths resolved to absolute before checking
  (blocks `../` traversal)
