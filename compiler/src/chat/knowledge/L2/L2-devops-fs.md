# fs (L2)
devops/fs — Filesystem utilities (find, walk, glob, copy, path ops).

## Functions
find_files(dir: string, pattern: string) → array
  Find files matching pattern
walk_dir(dir: string) → array
  Recursively list all files
file_info(path: string) → map
  Get file metadata (size, modified)
glob_files(pattern: string) → array
  Glob-match file paths
copy_file(src: string, dst: string) → int
  Copy file from src to dst
path_join(a: string, b: string) → string
  Join two path segments
path_parent(path: string) → string
  Get parent directory
path_name(path: string) → string
  Get filename from path
path_ext(path: string) → string
  Get file extension
