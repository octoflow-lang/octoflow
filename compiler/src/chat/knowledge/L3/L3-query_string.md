# query_string (L3)

## Working Example
```flow
use web/url

let mut params = map()
params["q"] = "octoflow lang"
params["page"] = "1"
params["sort"] = "date desc"

let qs = query_string(params)
print("Query: {qs}")

let base = "https://search.example.com/api"
let full_url = base + "?" + qs
print("URL: {full_url}")

let parsed = url_parse(full_url)
let host = map_get(parsed, "host")
let path = map_get(parsed, "path")
print("Host: {host}")
print("Path: {path}")
```

## Expected Output
```
Query: q=octoflow+lang&page=1&sort=date+desc
URL: https://search.example.com/api?q=octoflow+lang&page=1&sort=date+desc
Host: search.example.com
Path: /api
```

## Common Mistakes
- DON'T: manually build `"key=" + val + "&"` --> DO: `query_string(params)`
- DON'T: `params.set("key", val)` --> DO: `params["key"] = val`
- DON'T: forget to use `let mut` for the map --> DO: `let mut params = map()`

## Edge Cases
- query_string auto-encodes spaces and special characters
- All values should be strings; numeric values are converted automatically
- Empty map produces an empty string
