# io (L2)
data/io — Data I/O convenience wrappers (load/save JSON, text, lines, file size formatting)

## Functions
load_json(path: string) → map
  Load and parse JSON file
save_json(path: string, obj: map) → float
  Serialize and write JSON to file
load_text(path: string) → string
  Read entire file as string
save_text(path: string, s: string) → float
  Write string to file
load_lines(path: string) → array
  Read file as array of lines
save_lines(path: string, arr: array) → float
  Write array of lines to file
file_size_str(path: string) → string
  Human-readable file size (e.g. "1.2 MB")
