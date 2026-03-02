# web — Web & HTTP

HTTP client helpers, JSON utilities, and URL manipulation.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `http` | 3 | High-level HTTP client |
| `json_util` | 4 | JSON manipulation helpers |
| `url` | 3 | URL parsing and building |

## http

```
use web.http
```

| Function | Description |
|----------|-------------|
| `http_get_json(url)` | GET request, parse response as JSON |
| `http_post_json(url, data)` | POST JSON data, parse response |
| `api_call(method, url, headers, body)` | Generic API call with headers map |

Requires `--allow-net`.

```
let data = http_get_json("https://api.example.com/users")
print("Got {len(data)} users")

let mut payload = map()
payload["name"] = "test"
let result = http_post_json("https://api.example.com/users", payload)
```

## json_util

```
use web.json_util
```

| Function | Description |
|----------|-------------|
| `json_pretty(data)` | Pretty-print JSON |
| `json_merge(a, b)` | Merge two maps (b overrides a) |
| `json_get(data, key, default_val)` | Get value with default |
| `json_flatten(data, prefix)` | Flatten nested map to dot-separated keys |

## url

```
use web.url
```

| Function | Description |
|----------|-------------|
| `url_parse(url)` | Parse URL into components (protocol, host, path, query) |
| `build_url(protocol, host, path, query)` | Build URL from components |
| `query_string(params)` | Build query string from map |

```
let parts = url_parse("https://api.example.com/v1/users?page=2")
// parts["host"] = "api.example.com"
// parts["path"] = "/v1/users"
// parts["query"] = "page=2"
```

## Built-in HTTP

These are always available without `use`:

| Function | Description |
|----------|-------------|
| `http_get(url)` | GET request -> `.status`, `.body`, `.ok`, `.error` |
| `http_post(url, body)` | POST request |
| `http_put(url, body)` | PUT request |
| `http_delete(url)` | DELETE request |
| `json_parse(text)` | Parse JSON string to map |
| `json_parse_array(text)` | Parse JSON array |
| `json_stringify(val)` | Convert to JSON string |

## Built-in HTTP Server

| Function | Description |
|----------|-------------|
| `http_listen(port)` | Listen for HTTP connections |
| `http_accept(fd)` | Accept request |
| `http_method(fd)` | Get request method |
| `http_path(fd)` | Get request path |
| `http_body(fd)` | Get request body |
| `http_respond(fd, status, body)` | Send response |
| `http_respond_json(fd, status, json)` | Send JSON response |
