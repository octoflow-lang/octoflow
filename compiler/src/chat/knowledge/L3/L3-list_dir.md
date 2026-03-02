# list_dir (L3)

## Working Example
```flow
use devops/fs

let files = find_files("./data", "*")
print("Found {len(files)} items in ./data")

for f in files
    let ext = path_ext(f)
    let name = path_name(f)
    print("  {name} (ext: {ext})")
end

let csvs = find_files("./data", "*.csv")
print("CSV files: {len(csvs)}")
```

## Expected Output
```
Found 5 items in ./data
  report.csv (ext: .csv)
  config.json (ext: .json)
  notes.txt (ext: .txt)
  image.png (ext: .png)
  summary.csv (ext: .csv)
CSV files: 2
```

## Common Mistakes
- DON'T: `list_dir("path")` --> DO: `find_files("path", "*")` (devops/fs function)
- DON'T: `f.ext` --> DO: `path_ext(f)` (functions not methods)
- DON'T: `print(name)` --> DO: `print("{name}")`
- DON'T: run without flag --> DO: run with `--allow-read`

## Edge Cases
- find_files returns full paths; use path_name to extract filenames
- Use walk_dir for recursive listing of all subdirectories
- Pattern "*" matches all files; use "*.csv" to filter by extension
