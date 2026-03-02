# walk_dir (L3)

## Working Example
```flow
use devops/fs

let files = walk_dir("./src")
let count = len(files)
print("found {count} files in ./src")

let mut flow_count = 0.0
for i in range(0, count)
  let path = get(files, i)
  let ext = str_ends_with(path, ".flow")
  if ext == 1.0
    flow_count = flow_count + 1.0
    print("  source: {path}")
  end
end

print("total .flow files: {flow_count}")

let root_files = walk_dir(".")
let root_count = len(root_files)
print("project root has {root_count} total files")
```

## Expected Output
```
found 12 files in ./src
  source: ./src/main.flow
  source: ./src/utils.flow
  source: ./src/parser.flow
total .flow files: 3
project root has 47 total files
```

## Common Mistakes
- DON'T: `walk_dir()` with no argument → DO: `walk_dir("./src")` (path required)
- DON'T: `files.length` → DO: `len(files)`
- DON'T: `for file in files` → DO: `for i in range(0, count)` then `get(files, i)`

## Edge Cases
- walk_dir on a nonexistent path returns an empty list
- Symbolic links are followed by default
- Hidden files (dotfiles) are included in results
