# send_response_cors (L3)

## Working Example
```flow
use web/server
use data/json

http_serve(8080)
print("CORS-enabled API on port 8080")

while 1.0
    let req = http_accept()
    let path = map_get(req, "path")
    let method = map_get(req, "method")
    let origin = map_get(req, "origin")

    if path == "/api/data"
        let mut body = map()
        body["items"] = [1, 2, 3]
        body["total"] = 3
        let json_body = json_encode(body)
        send_response_cors(req, 200, json_body, origin)
    elif method == "OPTIONS"
        send_response_cors(req, 204, "", origin)
    else
        send_response_cors(req, 404, "not found", origin)
    end
end
```

## Expected Output
```
CORS-enabled API on port 8080
```

## Common Mistakes
- DON'T: manually set CORS headers --> DO: `send_response_cors(fd, status, body, origin)`
- DON'T: forget OPTIONS preflight --> DO: handle OPTIONS with 204 and empty body
- DON'T: hardcode origin "*" in production --> DO: pass the request origin for reflection
- DON'T: run without flag --> DO: run with `--allow-net`

## Edge Cases
- send_response_cors sets Access-Control-Allow-Origin, Methods, and Headers
- OPTIONS preflight must return before the actual request is processed
- Pass empty string as origin to allow all origins (equivalent to "*")
