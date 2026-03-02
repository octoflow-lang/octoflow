# server (L2)
web/server — HTTP server (listen, accept, respond with text/JSON/HTML/redirect/error)

## Functions
http_serve(n: int) → float
  Start HTTP server on given port
text_response(s: string) → map
  Build plain text response
json_response(obj: map) → map
  Build JSON response
html_response(s: string) → map
  Build HTML response
redirect_response(url: string) → map
  Build redirect response
ok_json(obj: map) → map
  200 OK with JSON body shorthand
ok_html(s: string) → map
  200 OK with HTML body shorthand
not_found() → map
  404 Not Found response
