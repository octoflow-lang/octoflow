# string — String Utilities

Extended string manipulation, formatting, and pattern matching.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `string` | 15 | Padding, centering, case, character checks |
| `format` | 5 | Number formatting, byte sizes, durations |
| `regex` | 3 | Glob matching, escaping, word count |

## string

```
use string.string
```

| Function | Description |
|----------|-------------|
| `str_center(s, width, fill)` | Center string with fill character |
| `str_ljust(s, width, fill)` | Left-justify with fill |
| `str_rjust(s, width, fill)` | Right-justify with fill |
| `str_count(s, sub)` | Count substring occurrences |
| `str_repeat(s, n)` | Repeat string n times |
| `str_reverse(s)` | Reverse string |
| `str_is_digit(s)` | 1.0 if all digits |
| `str_is_alpha(s)` | 1.0 if all alphabetic |
| `str_is_empty(s)` | 1.0 if empty after trim |
| `str_lstrip(s)` | Strip left whitespace |
| `str_rstrip(s)` | Strip right whitespace |
| `str_title(s)` | Title Case |
| `str_zfill(s, width)` | Zero-pad to width |
| `str_join(arr, sep)` | Join array with separator |
| `str_split(s, sep)` | Split by separator |

```
let s = str_center("hello", 20, "-")    // "-------hello--------"
let title = str_title("hello world")     // "Hello World"
let padded = str_zfill("42", 5)          // "00042"
```

## format

```
use string.format
```

| Function | Description |
|----------|-------------|
| `pad_left(s, width, ch)` | Pad string on left |
| `pad_right(s, width, ch)` | Pad string on right |
| `format_number(n, decimals)` | Format float to N decimal places |
| `format_bytes(n)` | Human-readable byte size (B, KB, MB, GB, TB) |
| `format_duration(seconds)` | Human-readable duration (d, h, m, s) |

```
let size = format_bytes(1536000)         // "1.46 MB"
let time = format_duration(3661)         // "1h 1m 1s"
let pi = format_number(3.14159, 2)       // "3.14"
```

## regex

```
use string.regex
```

| Function | Description |
|----------|-------------|
| `glob_match(pattern, text)` | Shell-style glob matching |
| `str_escape(s)` | Escape special regex characters |
| `word_count(s)` | Count words in string |

## Built-in String Functions

These are always available without `use`:

| Function | Description |
|----------|-------------|
| `len(s)` | String length |
| `trim(s)` | Remove whitespace |
| `to_upper(s)` | Uppercase |
| `to_lower(s)` | Lowercase |
| `contains(s, sub)` | 1.0 if contains substring |
| `starts_with(s, pre)` | 1.0 if starts with prefix |
| `ends_with(s, suf)` | 1.0 if ends with suffix |
| `index_of(s, sub)` | First index, -1 if not found |
| `char_at(s, idx)` | Character at index |
| `substr(s, start, len)` | Substring extraction |
| `replace(s, old, new)` | Replace all occurrences |
| `split(s, delim)` | Split into array |
| `join(arr, delim)` | Join array into string |
| `ord(c)` | Character code point |
| `chr(n)` | Character from code point |
| `regex_match(text, pat)` | 1.0 if matches |
| `regex_find(text, pat)` | First match string |
| `regex_find_all(text, pat)` | All matches as array |
| `regex_replace(text, pat, rep)` | Replace all matches |
