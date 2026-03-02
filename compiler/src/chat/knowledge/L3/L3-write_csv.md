# write_csv (L3)

## Working Example
```flow
use data/csv

let mut rows = []
let header = ["name", "score", "grade"]
push(rows, header)

let r1 = ["Alice", "95", "A"]
let r2 = ["Bob", "82", "B"]
let r3 = ["Carol", "78", "C"]
push(rows, r1)
push(rows, r2)
push(rows, r3)

csv_write("grades.csv", rows)
print("Wrote {len(rows)} rows to grades.csv")
```

## Expected Output
```
Wrote 4 rows to grades.csv
```

## Common Mistakes
- DON'T: `csv_write(rows, "path")` --> DO: `csv_write("path", rows)` (path first)
- DON'T: run without flag --> DO: run with `--allow-write`
- DON'T: `rows.push(r1)` --> DO: `push(rows, r1)`
- DON'T: pass maps to csv_write --> DO: pass array of arrays (each row is an array)

## Edge Cases
- Include header row as the first element of the rows array
- Numeric values must be strings in the row arrays (e.g., "95" not 95)
- Fields containing commas or quotes are automatically escaped
