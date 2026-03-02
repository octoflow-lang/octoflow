# url (L2)
web/url — URL parsing and building (parse to components, build from parts, query strings)

## Functions
url_parse(url: string) → map
  Parse URL into components (scheme, host, path, query, fragment)
build_url(obj: map) → string
  Build URL string from component map
query_string(obj: map) → string
  Encode map as URL query string
