# format (L2)
string/format — String formatting (pad left/right, format number, format bytes, format duration)

## Functions
pad_left(s: string, n: int, s: string) → string
  Pad string on left to width n with fill char
pad_right(s: string, n: int, s: string) → string
  Pad string on right to width n with fill char
format_number(x: float, n: int) → string
  Format number with n decimal places
format_bytes(x: float) → string
  Human-readable byte size (e.g. "1.2 KB")
format_duration(x: float) → string
  Human-readable duration from seconds
