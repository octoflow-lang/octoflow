# write_file (L3)

## Working Example
```flow
use data/io

let mut lines = []
push(lines, "Title: Weekly Report")
push(lines, "Date: 2026-02-28")
push(lines, "Status: Complete")
push(lines, "")
push(lines, "All tasks finished on schedule.")

save_lines("report.txt", lines)
print("Wrote {len(lines)} lines to report.txt")

let single = "Quick note: remember to backup."
save_text("memo.txt", single)
print("Saved memo.txt")
```

## Expected Output
```
Wrote 5 lines to report.txt
Saved memo.txt
```

## Common Mistakes
- DON'T: `write_file("path", text)` --> DO: `save_text("path", text)` (data/io wrapper)
- DON'T: run without flag --> DO: run with `--allow-write`
- DON'T: `lines.push("text")` --> DO: `push(lines, "text")`
- DON'T: `save_text(content, "path")` --> DO: `save_text("path", content)` (path first)

## Edge Cases
- save_text overwrites existing files; use io_append_file from data/io_root to append
- save_lines joins the array with newline separators automatically
- Empty string writes create an empty file (not an error)
