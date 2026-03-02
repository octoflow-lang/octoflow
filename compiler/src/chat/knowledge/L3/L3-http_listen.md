# http_listen (L3)

## Working Example
```flow
use web/server
use data/json

http_serve(3000)
print("Server running on port 3000")

while 1.0
    let req = http_accept()
    let path = map_get(req, "path")
    let method = map_get(req, "method")

    if path == "/hello"
        let resp = text_response("Hello, OctoFlow!")
        http_respond(req, resp)
    elif path == "/status"
        let mut data = map()
        data["status"] = "ok"
        data["uptime"] = 120
        let resp = json_response(data)
        http_respond(req, resp)
    else
        let resp = not_found()
        http_respond(req, resp)
    end
end
```

## Expected Output
```
Server running on port 3000
```

## Common Mistakes
- DON'T: `http_listen(3000)` --> DO: `http_serve(3000)` (web/server wrapper)
- DON'T: `req.path` --> DO: `map_get(req, "path")`
- DON'T: run without flag --> DO: run with `--allow-net`
- DON'T: forget the accept loop --> DO: use `while 1.0` with `http_accept()`

## Edge Cases
- http_accept blocks until a request arrives
- Use text_response, json_response, or html_response to build reply objects
- Server runs until the process is killed; no graceful shutdown built in
