# Permissions

OctoFlow uses a Deno-style permission model. By default, programs have no access to
the filesystem, network, or process execution. Permissions must be granted explicitly
via command-line flags.

## Permission Flags

| Flag | Grants | Example |
|------|--------|---------|
| `--allow-read` | File read access | `--allow-read` or `--allow-read=./data` |
| `--allow-write` | File write access | `--allow-write` or `--allow-write=./output` |
| `--allow-net` | Network access | `--allow-net` or `--allow-net=api.example.com` |
| `--allow-exec` | Process execution | `--allow-exec` or `--allow-exec=/usr/bin/git` |

## Bare vs Scoped

**Bare flag** — grants unrestricted access for that category:
```bash
octoflow run app.flow --allow-read --allow-write
```

**Scoped flag** — restricts access to a specific path or host:
```bash
octoflow run app.flow --allow-read=./data --allow-write=./output
```

With `--allow-read=./data`, the program can read files inside `./data/` but not
from `./secrets/` or `/etc/`. Attempts to read outside the scope produce error E051.

## Examples

### Read a CSV, write results

```bash
octoflow run analysis.flow --allow-read=./data --allow-write=./output
```

```flow
let csv = csv_read("./data/sales.csv")
let total = arr_sum(csv_column(csv, "amount"))
write_file("./output/report.txt", "Total: {total}")
```

### Fetch data from an API

```bash
octoflow run fetch.flow --allow-net=api.example.com
```

```flow
let data = http_get_json("https://api.example.com/data")
print("{data}")
```

### Run an external tool

```bash
octoflow run deploy.flow --allow-exec=/usr/bin/git --allow-net
```

```flow
let status = run("git status")
print("{status}")
```

### Multiple scoped permissions

```bash
octoflow run pipeline.flow \
  --allow-read=./input \
  --allow-write=./output \
  --allow-net=api.weather.com \
  --allow-net=api.stocks.com
```

## Chat Mode Permissions

Chat-generated code runs in a sandbox by default. To grant permissions:

```bash
octoflow chat --allow-read=./data --allow-net
```

The sandbox also enforces:
- Maximum 1,000,000 loop iterations
- I/O scoped to current working directory (unless explicitly widened)
- No process execution (unless `--allow-exec` granted)

## Error Codes

| Code | Error | Example |
|------|-------|---------|
| E051 | Network permission denied | `web_search()` without `--allow-net` |
| E052 | File read permission denied | `read_file()` without `--allow-read` |
| E053 | File write permission denied | `write_file()` without `--allow-write` |
| E054 | Exec permission denied | `run()` without `--allow-exec` |
| E055 | Path outside allowed scope | `read_file("/etc/passwd")` with `--allow-read=./data` |

## Why Permissions Matter

OctoFlow is designed for LLM-generated code. When `octoflow chat` generates and
runs code, permissions ensure that generated code cannot:

- Read sensitive files (SSH keys, credentials, system configs)
- Write to arbitrary locations (overwrite system files)
- Phone home to unknown servers (exfiltrate data)
- Execute arbitrary commands (privilege escalation)

Start with minimal permissions. Add more as needed.
