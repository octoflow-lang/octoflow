# http_post (L3)

## Working Example
```flow
use data/json

let mut payload = map()
payload["title"] = "New Task"
payload["priority"] = "high"
payload["done"] = 0.0

let body = json_encode(payload)
let resp = http_post("https://api.example.com/tasks", body)

if resp.ok == 1.0
    if resp.status == 201
        let result = json_decode(resp.body)
        let id = map_get(result, "id")
        print("Created task {id}")
    else
        print("Server error: {resp.status}")
    end
else
    print("Request failed: {resp.error}")
end
```

## Expected Output
```
Created task 42
```

## Common Mistakes
- DON'T: `http_post(url, map)` --> DO: `http_post(url, json_encode(map))` (body must be a string)
- DON'T: run without flag --> DO: run with `--allow-net`
- DON'T: `resp["body"]` --> DO: `resp.body` (struct field access)
- DON'T: `if resp.status == "201"` --> DO: `if resp.status == 201` (numeric comparison)

## Edge Cases
- http_post sends the body as-is; serialize to JSON first if needed
- Check resp.ok before resp.status; network failure means no status code
- Status 201 (Created) is typical for successful POST; 200 is also valid
