# read_csv (L3)

## Working Example
```flow
use data/csv

let rows = csv_read("sales.csv")
let headers = csv_headers(rows)
print("Columns: {headers}")

let amounts = csv_column(rows, 2)
let mut total = 0.0
for val in amounts
    total = total + float(val)
end

let count = len(amounts)
let avg = total / count
print("Total: {total}")
print("Average: {avg}")
print("Rows: {count}")
```

## Expected Output
```
Columns: ["date", "product", "amount"]
Total: 4250.0
Average: 850.0
Rows: 5
```

## Common Mistakes
- DON'T: `csv_read(path)` with no flag --> DO: run with `--allow-read`
- DON'T: `rows.headers()` --> DO: `csv_headers(rows)`
- DON'T: `rows[0]["amount"]` on raw CSV --> DO: `csv_column(rows, 2)` for column extraction
- DON'T: `print(total)` --> DO: `print("{total}")`

## Edge Cases
- csv_read returns raw string arrays; column 0 is the first field after headers
- Empty CSV file returns an empty array; check `len(rows)` before accessing columns
- Quoted fields with commas inside are handled per RFC 4180
