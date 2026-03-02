# csv_filter (L3)

## Working Example
```flow
use data/csv

let data = csv_read("sensors.csv")
let total = len(data)
print("loaded {total} rows")

let hot = csv_filter(data, fn(row)
  return row.temperature > 30.0
end)

let hot_count = len(hot)
print("rows above 30C: {hot_count}")

for i in range(0, hot_count)
  let r = get(hot, i)
  let id = r.sensor_id
  let temp = r.temperature
  print("  sensor {id}: {temp}C")
end

let critical = csv_filter(hot, fn(row)
  return row.temperature > 50.0
end)

let crit_count = len(critical)
print("critical (>50C): {crit_count}")
```

## Expected Output
```
loaded 1000 rows
rows above 30C: 342
  sensor S-001: 31.5C
  sensor S-003: 45.2C
  sensor S-007: 33.8C
critical (>50C): 18
```

*(Output varies with CSV content. Only first few filtered rows shown.)*

## Common Mistakes
- DON'T: `data.filter(fn)` → DO: `csv_filter(data, fn(row) ... end)`
- DON'T: `csv_read("file.csv");` → DO: no semicolons
- DON'T: `fn(row) { return row.x > 5.0 }` → DO: `fn(row) return row.x > 5.0 end`

## Edge Cases
- csv_filter preserves column types inferred by csv_read
- Chaining filters (filter the result of a filter) is supported
- Empty filter result returns an empty list, not an error
