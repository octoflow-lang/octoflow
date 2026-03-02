# map (L3)

## Working Example
```flow
let mut config = map()
config["host"] = "localhost"
config["port"] = 8080
config["debug"] = 1.0

let keys = map_keys(config)
print("Keys: {keys}")

for k in keys
    let val = map_get(config, k)
    print("  {k} = {val}")
end

let has_host = map_has(config, "host")
let has_ssl = map_has(config, "ssl")
print("Has host: {has_host}")
print("Has ssl: {has_ssl}")
```

## Expected Output
```
Keys: ["host", "port", "debug"]
  host = localhost
  port = 8080
  debug = 1.0
Has host: 1.0
Has ssl: 0.0
```

## Common Mistakes
- DON'T: `let m = {}` --> DO: `let mut m = map()`
- DON'T: `m.get("key")` --> DO: `map_get(m, "key")`
- DON'T: `"key" in m` --> DO: `map_has(m, "key")`
- DON'T: `let m = map()` then `m["k"] = v` --> DO: `let mut m = map()` (must be mutable to set keys)

## Edge Cases
- map_get on missing key returns none; check with map_has first
- map_keys returns an array of strings; iteration order is insertion order
- Maps can hold any value type including nested maps and arrays
