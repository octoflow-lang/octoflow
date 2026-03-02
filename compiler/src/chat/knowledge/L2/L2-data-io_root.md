# io (L2)
io — File I/O via FFI. Requires --allow-ffi.

## Functions
io_read_file(path: string) → string
  Read entire file as string
io_read_bytes(path: string) → array
  Read file as byte array
io_write_file(path: string, data: string) → int
  Write string to file
io_append_file(path: string, data: string) → int
  Append string to file
io_write_bytes(path: string, data: array) → int
  Write byte array to file
io_puts(msg: string) → int
  Print to stdout with newline
io_remove(path: string) → int
  Delete file
io_file_size(path: string) → int
  Get file size in bytes
