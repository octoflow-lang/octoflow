# data — Data Loading & Transformation

Modules for reading, writing, transforming, and validating structured data.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `csv` | 9 | CSV read/write/filter/sort |
| `io` | 7 | JSON/text file helpers |
| `transform` | 8 | Normalize, standardize, clip, resample |
| `validate` | 5 | Input validation utilities |
| `pipeline` | 3 | Batch processing and chaining |

## csv

```
use data.csv
```

| Function | Description |
|----------|-------------|
| `csv_read(path)` | Read CSV file into array of maps |
| `csv_write(rows, path, headers)` | Write array of maps to CSV file |
| `csv_headers(rows)` | Extract column header names |
| `csv_column(rows, col)` | Get column as numeric array |
| `csv_column_str(rows, col)` | Get column as string array |
| `csv_filter(rows, col, op, val)` | Filter rows by numeric condition (`>`, `<`, `>=`, `<=`, `==`, `!=`) |
| `csv_filter_str(rows, col, op, val)` | Filter rows by string condition (`==`, `!=`, `contains`, `starts_with`) |
| `csv_select(rows, cols)` | Select subset of columns |
| `csv_sort(rows, col, ascending)` | Sort rows by column |

```
let rows = csv_read("prices.csv")
let closes = csv_column(rows, "close")
let expensive = csv_filter(rows, "price", ">", 100.0)
csv_write(expensive, "filtered.csv", csv_headers(rows))
```

## io

```
use data.io
```

| Function | Description |
|----------|-------------|
| `load_json(path)` | Load and parse JSON file to map |
| `save_json(data, path)` | Save map as JSON file |
| `load_text(path)` | Load file as single string |
| `save_text(content, path)` | Write string to file |
| `load_lines(path)` | Load file as array of lines |
| `save_lines(lines, path)` | Write array of lines to file |
| `file_size_str(path)` | Human-readable file size (e.g., "1.5 MB") |

## transform

```
use data.transform
```

| Function | Description |
|----------|-------------|
| `normalize(arr)` | Min-max normalization to [0, 1] |
| `standardize(arr)` | Z-score standardization (mean=0, std=1) |
| `one_hot(labels, num_classes)` | One-hot encoding |
| `clip(arr, lo, hi)` | Clip values to range |
| `interpolate_missing(arr, missing_val)` | Linear interpolation for missing values |
| `resample(arr, factor)` | Downsample by taking every Nth element |
| `bin_data(arr, num_bins)` | Histogram binning |
| `scale_to_range(arr, new_min, new_max)` | Scale array to target range |

```
let raw = csv_column(csv_read("data.csv"), "value")
let norm = normalize(raw)           // [0, 1]
let std = standardize(raw)          // mean=0, std=1
let clipped = clip(raw, 0, 100)     // cap outliers
```

## validate

```
use data.validate
```

| Function | Description |
|----------|-------------|
| `is_numeric(s)` | 1.0 if string is a valid number |
| `is_email(s)` | 1.0 if basic email format |
| `is_url(s)` | 1.0 if basic URL format |
| `in_range(val, low, high)` | 1.0 if value is within range |
| `clamp_value(val, low, high)` | Clamp value to [low, high] |

## pipeline

```
use data.pipeline
```

| Function | Description |
|----------|-------------|
| `chain_apply(arr, fn_names)` | Apply named transforms in sequence (abs, floor, ceil, round, sqrt, exp, log, negate, double, halve) |
| `batch_process(arr, batch_size, fn_name)` | Process array in batches with aggregate (sum, mean, min, max, count) |
| `parallel_merge(arr1, arr2)` | Interleave two arrays |
