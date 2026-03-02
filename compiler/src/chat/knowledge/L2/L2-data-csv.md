# csv (L2)
data/csv — CSV parsing/formatting (RFC 4180 quoted fields, read/write, filter, sort, columns)

## Functions
csv_parse_line(s: string, delim: string) → array
  Split one CSV line into fields
csv_parse_text(text: string, delim: string) → array
  Parse multi-line CSV text into array of rows
csv_format_row(arr: array, delim: string) → string
  Format one row as CSV string
csv_format(data: array, delim: string) → string
  Format array of rows as CSV text
csv_read(path: string) → array
  Read CSV file into array of rows
csv_write(path: string, data: array) → float
  Write rows to CSV file
csv_headers(data: array) → array
  Extract header row from parsed CSV
csv_column(data: array, n: int) → array
  Extract single column by index
csv_filter(data: array, fn: string) → array
  Filter rows by predicate
csv_sort(data: array, n: int) → array
  Sort rows by column index
