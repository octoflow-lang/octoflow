# log (L2)
devops/log — Level-based logging with timestamps.

## Functions
log_set_level(level: string) → int
  Set minimum log level (DEBUG/INFO/WARN/ERROR)
log_debug(msg: string) → int
  Log debug message
log_info(msg: string) → int
  Log info message
log_warn(msg: string) → int
  Log warning message
log_error(msg: string) → int
  Log error message
log_timed(label: string, fn: any) → any
  Log execution time of function
