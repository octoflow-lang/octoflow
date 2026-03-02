# platform (L2)
sys/platform — Platform detection and OS paths.

## Functions
get_os() → string
  Get OS name (windows/linux/macos)
is_windows() → int
  Check if running on Windows
is_linux() → int
  Check if running on Linux
is_mac() → int
  Check if running on macOS
home_dir() → string
  Get user home directory
temp_dir() → string
  Get system temp directory
path_separator() → string
  Get OS path separator (/ or \\)
