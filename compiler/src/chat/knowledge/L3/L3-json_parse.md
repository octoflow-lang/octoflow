# json_parse (L3)

## Working Example
```flow
use data/json

let text = "{\"name\": \"Alice\", \"age\": 30, \"active\": true}"
let obj = json_decode(text)

let name = map_get(obj, "name")
let age = map_get(obj, "age")
print("Name: {name}")
print("Age: {age}")

let has_email = map_has(obj, "email")
if has_email == 0.0
    print("No email field found")
end
```

## Expected Output
```
Name: Alice
Age: 30
No email field found
```

## Common Mistakes
- DON'T: `json_parse(text)` --> DO: `json_decode(text)` (correct function name)
- DON'T: `obj["name"]` for reading --> DO: `map_get(obj, "name")`
- DON'T: `obj.name` --> DO: `map_get(obj, "name")`
- DON'T: `if has_email == false` --> DO: `if has_email == 0.0`

## Edge Cases
- json_decode returns a map; for JSON arrays use json_get_array()
- Nested objects become nested maps; use map_get twice to drill down
- JSON true/false become 1.0/0.0 in OctoFlow
