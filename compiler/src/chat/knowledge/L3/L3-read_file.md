# read_file (L3)

## Working Example
```flow
use data/io

let content = load_text("notes.txt")
let lines = split(content, "\n")
let count = len(lines)
print("File has {count} lines")

for i in range(0, count)
    let line = lines[i]
    if len(line) > 0
        print("Line {i}: {line}")
    end
end
```

## Expected Output
```
File has 4 lines
Line 0: Buy groceries
Line 1: Finish report
Line 2: Call dentist
Line 3: Read chapter 5
```

## Common Mistakes
- DON'T: `read_file("path")` --> DO: `load_text("path")` (data/io wrapper)
- DON'T: run without flag --> DO: run with `--allow-read`
- DON'T: `content.split("\n")` --> DO: `split(content, "\n")`
- DON'T: `print(line)` --> DO: `print("{line}")`

## Edge Cases
- load_text reads the entire file as a single string; use load_lines for an array
- Non-existent files cause a runtime error; wrap with try() for safe access
- Large files are read fully into memory; consider load_lines for line-by-line processing
