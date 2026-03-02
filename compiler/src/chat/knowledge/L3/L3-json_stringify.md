# json_stringify (L3)

## Working Example
```flow
use data/json

let mut user = map()
user["name"] = "Bob"
user["age"] = 25
user["scores"] = [90, 85, 92]

let text = json_encode(user)
print("JSON: {text}")

use web/json_util
let pretty = json_pretty(user)
print("Pretty:")
print("{pretty}")
```

## Expected Output
```
JSON: {"name":"Bob","age":25,"scores":[90,85,92]}
Pretty:
{
  "name": "Bob",
  "age": 25,
  "scores": [90, 85, 92]
}
```

## Common Mistakes
- DON'T: `json_stringify(obj)` --> DO: `json_encode(obj)` (correct function name)
- DON'T: `print(text)` --> DO: `print("{text}")`
- DON'T: `map("name", "Bob")` --> DO: `let mut m = map()` then `m["name"] = "Bob"`

## Edge Cases
- Arrays and nested maps are serialized recursively
- none values are serialized as JSON null
- Use json_pretty from web/json_util for human-readable output
