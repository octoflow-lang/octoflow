# http (L2)
web/http — HTTP client (GET/POST JSON, generic API calls)

## Functions
http_get_json(url: string) → map
  GET request, parse JSON response
http_post_json(url: string, obj: map) → map
  POST JSON body, parse JSON response
api_call(url: string, method: string, obj: map) → map
  Generic HTTP request with method and optional body
