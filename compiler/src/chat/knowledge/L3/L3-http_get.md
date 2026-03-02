# http_get (L3)

## Working Example
```flow
use data/json

let resp = http_get("https://api.example.com/users/1")
if resp.ok == 1.0
    if resp.status == 200
        let user = json_decode(resp.body)
        let name = map_get(user, "name")
        let email = map_get(user, "email")
        print("User: {name}")
        print("Email: {email}")
    else
        print("HTTP error: {resp.status}")
    end
else
    print("Request failed: {resp.error}")
end
```

## Expected Output
```
User: Alice Smith
Email: alice@example.com
```

## Common Mistakes
- DON'T: run without flag --> DO: run with `--allow-net`
- DON'T: `resp.json()` --> DO: `json_decode(resp.body)`
- DON'T: `if resp.ok` --> DO: `if resp.ok == 1.0`
- DON'T: `http_get(url, headers)` --> DO: `http_get(url)` (single argument)

## Edge Cases
- Always check resp.ok before accessing resp.status or resp.body
- resp.body is always a string; parse with json_decode if JSON
- Network errors set resp.ok to 0.0 and resp.error to the message
