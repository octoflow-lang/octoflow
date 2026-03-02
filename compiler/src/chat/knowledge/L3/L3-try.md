# try (L3)

## Working Example
```flow
use data/io

let result = try(load_text("config.txt"))
if result.ok == 1.0
    let content = result.value
    print("Config loaded: {content}")
else
    let err = result.error
    print("Could not load config: {err}")
    print("Using defaults")
end

let num_result = try(int("not_a_number"))
if num_result.ok == 0.0
    print("Parse failed: {num_result.error}")
end
```

## Expected Output
```
Could not load config: file not found: config.txt
Using defaults
Parse failed: invalid integer literal
```

## Common Mistakes
- DON'T: `if result.ok == true` --> DO: `if result.ok == 1.0`
- DON'T: `try { expr }` --> DO: `try(expr)` (function call syntax)
- DON'T: `result.value` without checking .ok --> DO: check `result.ok == 1.0` first
- DON'T: `catch(e)` --> DO: use `result.error` field

## Edge Cases
- try() wraps any expression; the inner expression is evaluated once
- On success: .ok is 1.0, .value holds the result, .error is none
- On failure: .ok is 0.0, .value is none, .error holds the error message string
