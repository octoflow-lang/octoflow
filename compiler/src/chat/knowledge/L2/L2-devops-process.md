# process (L2)
devops/process — Shell and process utilities.

## Functions
run(cmd: string) → string
  Run command, return stdout
run_shell(cmd: string) → string
  Run via shell interpreter
run_status(cmd: string) → int
  Run command, return exit code
run_ok(cmd: string) → int
  Check if command succeeds (exit 0)
pipe(cmds: array) → string
  Pipe multiple commands together
env_get(name: string) → string
  Get environment variable
which(name: string) → string
  Find executable path
