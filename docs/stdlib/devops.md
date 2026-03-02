# devops — DevOps & Scripting

File system operations, process management, logging, configuration,
and templating.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `fs` | 9 | File finding, copying, path manipulation |
| `process` | 7 | Command execution, environment |
| `log` | 6 | Leveled logging with timestamps |
| `config` | 3 | INI and .env file parsing |
| `template` | 3 | String template rendering |

## fs

```
use devops.fs
```

| Function | Description |
|----------|-------------|
| `find_files(dir, extension)` | Find files with given extension |
| `walk_dir(dir)` | Recursive directory listing |
| `file_info(path)` | File metadata (exists, is_file, is_dir, name, ext) |
| `glob_files(dir, pattern)` | Simple glob matching (*.ext) |
| `copy_file(src, dst)` | Copy file |
| `path_join(parts)` | Join path parts with `/` |
| `path_parent(path)` | Parent directory |
| `path_name(path)` | Base filename |
| `path_ext(path)` | File extension |

```
let flows = find_files("stdlib/", ".flow")
for f in flows
    let info = file_info(f)
    print("{info}")
end
```

## process

```
use devops.process
```

| Function | Description |
|----------|-------------|
| `run(cmd)` | Run command and return output |
| `run_shell(cmd)` | Run via system shell (cross-platform) |
| `run_status(cmd)` | Run command, return exit status |
| `run_ok(cmd)` | 1.0 if command succeeds |
| `pipe(cmd1, cmd2)` | Chain two commands sequentially |
| `env_get(name)` | Get environment variable |
| `which(program)` | Find program in PATH |

Requires `--allow-exec`.

```
let status = run_shell("git status")
let node = which("node")
if node != ""
    print("Node.js found at {node}")
end
```

## log

```
use devops.log
```

| Function | Description |
|----------|-------------|
| `log_set_level(level)` | Set minimum level: DEBUG, INFO, WARN, ERROR |
| `log_debug(msg)` | Log at DEBUG level |
| `log_info(msg)` | Log at INFO level |
| `log_warn(msg)` | Log at WARN level |
| `log_error(msg)` | Log at ERROR level |
| `log_timed(msg)` | Log with timestamp |

```
log_set_level("INFO")
log_info("Starting process")
log_warn("Disk usage at 90%")
```

## config

```
use devops.config
```

| Function | Description |
|----------|-------------|
| `parse_ini(content)` | Parse INI-style config to map |
| `parse_env_file(path)` | Parse .env file (KEY=VALUE) |
| `config_get(config, key, default_val)` | Get value with fallback default |

## template

```
use devops.template
```

| Function | Description |
|----------|-------------|
| `render(template, vars)` | Replace `{key}` placeholders with map values |
| `load_template(path)` | Load template from file |
| `render_file(template_path, vars)` | Load and render in one call |

```
let mut vars = map()
vars["name"] = "OctoFlow"
vars["version"] = "0.82"
let html = render_file("template.html", vars)
```
