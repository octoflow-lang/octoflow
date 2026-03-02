# env (L2)
sys/env — Environment variable access.

## Functions
get_env(name: string) → string
  Get environment variable
get_env_or(name: string, default: string) → string
  Get env var with fallback default
get_home() → string
  Get user home directory
get_cwd() → string
  Get current working directory
get_path() → string
  Get PATH variable
get_user() → string
  Get current username
get_temp_dir() → string
  Get system temp directory
