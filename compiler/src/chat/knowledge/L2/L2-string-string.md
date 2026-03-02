# string (L2)
string/string — String utilities (center, justify, count, reverse, is_digit, is_alpha, strip, title, zfill)

## Functions
str_center(s: string, n: int) → string
  Center string in field of width n
str_ljust(s: string, n: int) → string
  Left-justify string in field of width n
str_rjust(s: string, n: int) → string
  Right-justify string in field of width n
str_count(s: string, s: string) → int
  Count occurrences of substring
str_reverse(s: string) → string
  Reverse string
str_is_digit(s: string) → float
  Test if all chars are digits (1.0/0.0)
str_is_alpha(s: string) → float
  Test if all chars are alphabetic (1.0/0.0)
str_lstrip(s: string) → string
  Strip leading whitespace
str_rstrip(s: string) → string
  Strip trailing whitespace
str_title(s: string) → string
  Title-case string
str_zfill(s: string, n: int) → string
  Zero-pad string to width n
