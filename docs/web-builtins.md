# Web Builtins

OctoFlow v1.2 includes two builtins for web-connected programs:
`web_search()` and `web_read()`. Both require the `--allow-net` permission.

## `web_search(query)`

Search the web and return structured results.

**Parameters:**
- `query` (string) — The search query

**Returns:** Array of maps, each containing:
- `title` (string) — Page title
- `url` (string) — Page URL
- `snippet` (string) — Text excerpt

**Example:**

```flow
let results = web_search("OctoFlow GPU language")
for r in results
    print("{r.title}")
    print("  {r.url}")
    print("  {r.snippet}")
    print("")
end
```

```bash
octoflow run search.flow --allow-net
```

## `web_read(url)`

Fetch a web page and extract its content.

**Parameters:**
- `url` (string) — The URL to fetch (HTTPS supported)

**Returns:** Map containing:
- `title` (string) — Page title
- `text` (string) — Extracted text content (HTML stripped)
- `headings` (array) — List of heading strings
- `links` (array) — List of link URLs found on the page

**Example:**

```flow
let page = web_read("https://example.com")
print("Title: {page.title}")
print("Text: {page.text}")

// Extract headings
let headings = page.headings
for h in headings
    print("  - {h}")
end
```

## Combining Search and Read

```flow
// Search for a topic, then read the top result
let results = web_search("Rust programming language features")
if len(results) > 0
    let url = results[0].url
    print("Reading: {url}")
    let page = web_read(url)
    print("{page.title}")
    print("{page.text}")
end
```

## Permission Requirements

Both builtins require `--allow-net`. You can scope to specific hosts:

```bash
# Allow all network access
octoflow run app.flow --allow-net

# Allow only specific hosts
octoflow run app.flow --allow-net=api.example.com
octoflow run app.flow --allow-net=example.com --allow-net=api.github.com
```

Without `--allow-net`, calling `web_search()` or `web_read()` produces
error E051 (network permission denied).

## Implementation Notes

- Search uses DuckDuckGo HTML (no API key required)
- HTTPS requests use system curl
- Page content is extracted as plain text (HTML tags stripped)
- Large pages are truncated to prevent memory issues
- Timeouts apply to prevent hanging on slow servers
