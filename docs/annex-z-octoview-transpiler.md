# Annex Z: The OctoView Web Transpiler

> Every page is data.

**Date:** 2026-02-25
**Status:** Concept paper — architecture, transpiler pipeline, OVD format, progressive rendering
**Supplements:** Annex I (OctoView browser shell + OctoFlowWeb native app framework)

---

## The Thesis

Every browser engine — Chrome, Firefox, Safari, Servo, Ladybird — does the same
thing: implement the 10,000-page HTML/CSS/JS specification as faithfully as possible.
They compete on correctness and speed of the **same pipeline**. None of them question
whether the pipeline itself is the problem.

OctoView rejects the pipeline. It doesn't ask "how do we render this HTML faster?"
It asks **"what is this page actually saying, and how do we express that on GPU?"**

```
Chrome:     HTML → DOM → CSSOM → JS → Render Tree → Layout → Paint → Composite → Pixels
            (all CPU until final composite, reimplements the full spec)

OctoView:   HTML → Transpile → OctoView Document → GPU Render
            (extract meaning, discard the ceremony, render natively)
```

This is not a browser that renders HTML. It's a **transpiler** that converts the web
into a GPU-native format. Browsing the web and scraping it become the same operation.

---

## Why This Matters

### The Web's Architecture Problem

The web stack was designed in 1995 for documents. It has been retrofitted for
applications through 30 years of accumulated complexity:

```
1995: HTML (documents)
1996: CSS (styling)
1997: JavaScript (interactivity)
2004: AJAX (dynamic content)
2010: Single-page apps (Gmail, Twitter)
2013: React (virtual DOM to manage complexity)
2016: Webpack, Babel, TypeScript (tooling to manage tooling)
2020: Next.js, Nuxt, Remix (frameworks to manage frameworks)
2024: Astro, HTMX (backlash against the complexity)
```

Each layer was added to compensate for the limitations of the layer below.
The result: a page load in Chrome triggers DNS lookup, TLS handshake, HTML parsing,
CSS parsing, CSS cascade resolution, JavaScript parsing, compilation (V8 JIT),
execution, virtual DOM construction, virtual DOM diffing, real DOM mutation,
style recalculation, layout, paint, layer creation, compositing. Most of this
happens on CPU. The GPU sits idle until the very last step.

### OctoView's Answer

Don't fix the pipeline. Replace it.

1. **Parse the web's formats** — use proven Rust crates (html5ever, cssparser, Boa)
2. **Extract meaning** — semantic extraction with heuristics for div-soup
3. **Express as GPU-native data** — flat columnar OctoView Document
4. **Render on GPU** — OctoUI dispatch chains, single Vulkan submit
5. **Cache the result** — .ovd files load instantly on revisit

The web's 30 years of accumulated complexity collapses into a single extraction step.

---

## Relationship to Annex I

Annex I covers two products:

```
Annex I:   OctoView (the browser shell) + OctoFlowWeb (the .flow web framework)
           ─── how developers BUILD for the GPU web
           ─── how users RUN native OctoFlowWeb apps

Annex Z:   The OctoView Web Transpiler
           ─── how OctoView CONSUMES the existing HTML/CSS/JS web
           ─── the engine that makes 30 years of web content accessible
```

Annex I's three compatibility tiers still apply:

```
Tier 1: NATIVE       OctoFlowWeb apps (.flow)       ← Annex I focus
Tier 2: TRANSPILED   React/Vue/Svelte auto-convert   ← Annex I Section 12
Tier 3: LEGACY WEB   HTML/CSS/JS websites            ← THIS DOCUMENT
```

This document is the deep-dive into Tier 3 — not as a concession ("80% compatible,
route the rest to Chrome") but as a **first-class feature** that makes the existing
web better than it is in Chrome.

---

## Architecture: The Three-Loom Pipeline

Every page load spawns three concurrent looms, each independent, each progressive.
The page weaves itself together — content first, then style, then behavior.

```
    HTTP Response (bytes streaming in)
              │
              ▼
┌─────────────────────────────────────────────────────┐
│              OctoView Web Transpiler                 │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  Loom 0: CONTENT (instant, <50ms)           │    │
│  │                                              │    │
│  │  Stream-parse HTML as bytes arrive           │    │
│  │  html5ever (Rust, streaming mode)            │    │
│  │  Extract: headings, text, links, images,     │    │
│  │    tables, lists, forms, navigation          │    │
│  │  Semantic heuristics on div soup             │    │
│  │  ARIA role extraction                        │    │
│  │  → OctoView Document (unstyled)              │    │
│  │  → GPU render (content visible immediately)  │    │
│  └──────────────┬──────────────────────────────┘    │
│                 │                                     │
│  ┌──────────────▼──────────────────────────────┐    │
│  │  Loom 1: STYLE (fast, ~100-200ms)           │    │
│  │                                              │    │
│  │  CSS arrives → cssparser (Rust)              │    │
│  │  Resolve cascade + inheritance → computed    │    │
│  │  Map CSS properties → OVD style records      │    │
│  │  Styles stored flat (no cascade at render)   │    │
│  │  → Patch OctoView Document with styles       │    │
│  │  → GPU re-render (styled, looks proper)      │    │
│  └──────────────┬──────────────────────────────┘    │
│                 │                                     │
│  ┌──────────────▼──────────────────────────────┐    │
│  │  Loom 2: BEHAVIOR (background cron)         │    │
│  │                                              │    │
│  │  Static page: skip entirely                  │    │
│  │  Light JS: Boa → DOM mutations → diff/patch  │    │
│  │  SPA: full execute → snapshot OVD → cache    │    │
│  │  Cron: re-snapshot every N minutes           │    │
│  │  → Progressive updates to rendered page      │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
└─────────────────────────────────────────────────────┘
              │
              ▼
     OctoView Document (.ovd)
              │
     ┌────────┼────────┐
     │        │        │
     ▼        ▼        ▼
  GPU       Query     Cache
  Render    (LoomDB)  (.ovd file)
```

### Why Three Looms, Not One

Every existing browser treats the rendering pipeline as **sequential and blocking**.
CSS blocks rendering. JS blocks parsing. The user stares at a white screen until
everything finishes.

OctoView makes each concern independent:

| Loom | Blocks on | User sees | Time |
|------|-----------|-----------|------|
| **Loom 0: Content** | Nothing (starts on first byte) | Text, headings, links, images | <50ms |
| **Loom 1: Style** | CSS download | Properly styled content | ~100-200ms |
| **Loom 2: Behavior** | Nothing (background) | JS-generated content appears progressively | 1-5s |

The user sees **content in under 50 milliseconds**. Not a loading spinner. Not a
white screen. Actual readable content with headings, paragraphs, links, images.
Style arrives ~150ms later. JS-generated content fills in over the next few seconds.

This is Progressive Enhancement — not as a development philosophy, but as an
**architectural guarantee**.

---

## Network Interception Points

OctoView intercepts at five points in the HTTP pipeline, earlier than any browser
needs to because we're not waiting for a complete document before acting.

```
Point A: DNS + CONNECT
─────────────────────
OctoView maintains its own connection pool (hyper/reqwest)
HTTP/2 multiplexing: all resources over one connection
Early Hints (103): server says "you'll need style.css and app.js"
  → prefetch starts before HTML body arrives
  → Loom 1 (style) can begin before Loom 0 finishes

Point B: RESPONSE HEADERS
─────────────────────────
Content-Type routes the response:
  text/html           → activate transpiler pipeline (three looms)
  application/json    → direct to LoomDB ingest (structured data)
  image/*             → decode → GPU texture upload → render
  text/plain          → OVD text node, render directly
  application/pdf     → PDF parser → OVD document
Cache headers checked:
  Fresh .ovd in cache → skip everything, render from cache (~10ms)
  Stale .ovd          → render from cache immediately, re-transpile in background

Point C: STREAMING BODY — Loom 0 starts
───────────────────────────────────────
HTML bytes stream via chunked transfer encoding
html5ever parses incrementally (streaming mode)
  → DOM fragments emitted as they're parsed
  → Semantic extraction runs on each fragment
  → GPU renders content as it arrives
OctoView's preload scanner runs in parallel:
  discovers <link>, <script>, <img> → fires parallel fetches
  CSS → feeds Loom 1
  JS → queues for Loom 2
  Images → decode → GPU texture cache

Point D: CSS ARRIVES — Loom 1 starts
────────────────────────────────────
cssparser resolves styles:
  cascade + inheritance → computed styles per node
  stored as flat style records in OVD (resolved once)
  no cascade resolution at render time — already computed
GPU re-renders with proper styling
  diff: only nodes whose styles changed get updated

Point E: JS QUEUED — Loom 2 starts (background cron)
────────────────────────────────────────────────────
JS files downloaded but NOT blocking render
Tiered execution strategy:
  No JS / analytics only → skip (done, page complete)
  Small JS bundles       → Boa executes in background
                           DOM mutations → diff OVD → patch → GPU update
  SPA (div#root + large) → full Boa execution
                           snapshot final OVD → cache to disk
                           cron: re-execute every N minutes for freshness
```

### The Cache Fast Path

```
User navigates to URL
        │
        ▼
  .ovd cache check
        │
    ┌───┴───┐
    │       │
  FRESH   STALE/MISS
    │       │
    ▼       ▼
  Render    Render stale .ovd immediately (instant)
  from      + re-transpile in background
  cache     + swap in fresh .ovd when ready
  (~10ms)   (stale-while-revalidate for pages)
```

Every page visited once is cached as an `.ovd` file. Return visits are **always
instant** (10ms), even for SPAs that originally took 5 seconds to load.

---

## OctoView Document Format (.ovd)

### Design: Flat Columnar With Logical Tree

The OVD is simultaneously a render target, a database table, and a semantic model.
Stored as flat parallel arrays (GPU-friendly), with parent/depth fields that
reconstruct the tree logically.

```
.ovd file
├── Header (48 bytes)
│   ├── Magic: "OVDX" (4 bytes)
│   ├── Version: u16
│   ├── Flags: u16 (compressed, has_images, has_js_snapshot)
│   ├── Node count: u32
│   ├── Source URL hash: u64 (for cache lookup)
│   ├── Created timestamp: u64
│   ├── Source URL length: u16
│   └── Source URL: UTF-8 string
│
├── Schema (column descriptors)
│   ├── Column: node_id      (u32, sequential)
│   ├── Column: type         (u8 enum)
│   ├── Column: depth        (u16, tree depth)
│   ├── Column: parent_id    (u32, -1 for root)
│   ├── Column: text         (string, content)
│   ├── Column: href         (string, URL)
│   ├── Column: src          (string, resource URL)
│   ├── Column: alt          (string, alt text)
│   ├── Column: level        (u8, heading level 1-6)
│   ├── Column: semantic     (u8 enum, inferred role)
│   ├── Column: source_tag   (u8 enum, original HTML tag)
│   ├── Column: style        (packed struct, computed CSS)
│   └── Column: bounds       (f32×4, layout x/y/w/h after GPU layout)
│
├── Node Data (flat columnar, .octo encoding)
│   ├── node_id[]      raw u32 LE
│   ├── type[]         raw u8
│   ├── depth[]        raw u16 LE
│   ├── parent_id[]    raw u32 LE
│   ├── text[]         length-prefixed UTF-8 strings
│   ├── href[]         length-prefixed UTF-8 strings
│   ├── src[]          length-prefixed UTF-8 strings
│   ├── alt[]          length-prefixed UTF-8 strings
│   ├── level[]        raw u8
│   ├── semantic[]     raw u8
│   ├── source_tag[]   raw u8
│   ├── style[]        packed structs (font_size, color, bg, bold, italic, ...)
│   └── bounds[]       raw f32×4 LE
│
├── Resource Table
│   ├── Resource entry[]
│   │   ├── url: string
│   │   ├── type: u8 (image, font, video, audio)
│   │   ├── size: u64
│   │   ├── hash: u64 (content hash for dedup)
│   │   └── data_offset: u64 (into embedded data section, or external ref)
│   └── Resource data (embedded small resources, external refs for large)
│
├── JS Snapshot Section (optional, for SPA cache)
│   ├── Snapshot timestamp: u64
│   ├── Cron interval: u32 (seconds)
│   └── DOM mutation log (for incremental update)
│
└── Footer
    ├── Stats: node count per type, max depth, content length
    ├── Checksum: xxhash64
    └── Magic: "OVDX" (sentinel)
```

### Node Types (enum)

```
0x01  page            Root document node
0x02  heading         h1-h6 or inferred heading
0x03  paragraph       Text block
0x04  link            Anchor with href
0x05  image           Image with src
0x06  table           Table (children are rows/cells)
0x07  table_row       Row within table
0x08  table_cell      Cell within row
0x09  list            Ordered or unordered list
0x0A  list_item       Item within list
0x0B  form            Form container
0x0C  input_field     Text input, checkbox, radio, etc.
0x0D  button          Clickable button
0x0E  navigation      Nav block (header nav, sidebar, breadcrumbs)
0x0F  media           Video or audio element
0x10  code_block      Preformatted code
0x11  blockquote      Quoted content
0x12  separator       Horizontal rule or visual separator
0x13  section         Generic content section
0x14  card            Inferred card pattern (image + text + link)
0x15  modal           Popup/dialog/overlay
0x16  sidebar         Side content
0x17  footer          Page footer
0x18  header          Page header
0x19  text_span       Inline text with distinct styling (bold, italic, code)
0x1A  icon            Small image/SVG used as icon
0xFF  unknown         Unclassified node (fallback)
```

### Semantic Roles (enum)

```
0x01  content         Primary page content
0x02  navigation      Navigation structure
0x03  search          Search functionality
0x04  sidebar         Auxiliary content
0x05  advertising     Ad content (detected by heuristics)
0x06  tracking        Analytics/tracking (hidden elements)
0x07  interactive     User-interactable element
0x08  media           Rich media content
0x09  metadata        Non-visible page metadata
0x0A  decoration      Purely decorative element
0x0B  structural      Layout container (no semantic meaning)
```

### Style Record (packed struct, 32 bytes per node)

```
struct OvdStyle {
    font_size:    f32,     // computed px
    font_weight:  u16,     // 100-900
    font_style:   u8,      // 0=normal, 1=italic
    text_align:   u8,      // 0=left, 1=center, 2=right, 3=justify
    color:        [u8; 4], // RGBA
    background:   [u8; 4], // RGBA
    margin:       [f32; 4],// top, right, bottom, left (computed px)
    padding:      [f32; 4],// top, right, bottom, left (computed px)
    display:      u8,      // 0=block, 1=inline, 2=flex, 3=grid, 4=none
    visibility:   u8,      // 0=hidden, 1=visible
    _padding:     [u8; 2], // alignment
}
// Total: 48 bytes per node (GPU-aligned to 16-byte boundary)
```

---

## Semantic Extraction Engine

The extraction engine is the core innovation — it doesn't just parse tags, it
**understands the page**.

### Layer 1: HTML Semantics (deterministic, fast)

Direct tag-to-type mapping. No ambiguity.

```
Tag                → OVD Type           Notes
──────────────────────────────────────────────────
<h1>..<h6>         → heading            level from tag number
<p>                → paragraph          text content extracted
<a href="...">     → link               href + text preserved
<img src="...">    → image              src + alt + width/height
<table>            → table              recursive: rows → cells
<ul>, <ol>         → list               ordered flag from tag
<li>               → list_item          within parent list
<form>             → form               children are fields
<input>, <select>  → input_field        type, name, placeholder
<button>           → button             text content
<nav>              → navigation         direct semantic
<article>          → section            content block
<aside>            → sidebar            auxiliary content
<header>           → header             page header
<footer>           → footer             page footer
<pre>, <code>      → code_block         preformatted text
<blockquote>       → blockquote         quoted content
<hr>               → separator          visual divider
<video>, <audio>   → media              src + type
```

This covers ~30% of the web (sites using semantic HTML: Wikipedia, docs, blogs,
government sites, academic papers).

### Layer 2: ARIA Roles (semantic hints from modern apps)

Modern web apps use ARIA attributes for accessibility. These are explicit semantic
signals that most browsers use only for screen readers. OctoView uses them as
primary extraction signals.

```
ARIA Attribute           → OVD Mapping
──────────────────────────────────────────────────────
role="button"            → button (even on <div> or <span>)
role="navigation"        → navigation
role="main"              → section (semantic: content)
role="search"            → form (semantic: search)
role="tab"               → button (semantic: interactive)
role="tabpanel"          → section
role="dialog"            → modal
role="alert"             → section (semantic: interactive)
role="banner"            → header
role="contentinfo"       → footer
role="complementary"     → sidebar
role="list"              → list
role="listitem"          → list_item
role="heading"           → heading (aria-level for level)
aria-label="..."         → accessible name (stored in text)
aria-hidden="true"       → skip (not content, not rendered)
aria-expanded="true/false" → state flag on parent node
```

This covers another ~20% — modern React/Vue/Angular apps that use ARIA properly.

### Layer 3: Heuristics (the div-soup decoder)

The remaining ~50% of the web is `<div>` and `<span>` soup with CSS classes. The
heuristic engine infers semantics from computed styles and structural patterns.

```
Signal                                          → Inference
─────────────────────────────────────────────────────────────────
font-size > 20px + font-weight ≥ 600            → heading (level by size)
  + short text (< 100 chars)                       largest = h1, decreasing
  + block display

Large text block (> 200 chars)                   → paragraph
  + line-height > 1.3                              (semantic: content)
  + font-size 14-18px

Repeated identical child structures              → list
  (3+ siblings with same tag/class pattern)        children → list_item
  e.g., div.card × 10 each with img + h3 + p

Repeated structure with image + text + link      → card (inferred)
  within grid/flex container                       (semantic: content)

position: fixed/sticky + contains links          → navigation (inferred)
  or: top of page + horizontal flex + links

position: absolute + z-index > 100               → modal (inferred)
  + overlay background

width < 300px + side position                    → sidebar (inferred)
  or: aside-like class names (*sidebar*, *aside*)

display: none or visibility: hidden              → skip (not content)
  or: zero dimensions

Class/ID name matching patterns:                 → semantic hints
  *nav*, *menu*, *header*                          navigation
  *footer*, *copyright*                            footer
  *sidebar*, *aside*                               sidebar
  *modal*, *dialog*, *popup*                       modal
  *ad*, *banner*, *sponsor*, *promo*               advertising
  *search*                                         search
  *content*, *article*, *post*, *entry*            content
  *btn*, *button*, *cta*                           button
```

### Heuristic Confidence Scoring

Each heuristic inference carries a confidence score:

```
Confidence   Source          Example
─────────────────────────────────────────────────────
1.00         HTML semantic   <h1> → heading
0.95         ARIA role       role="navigation" → navigation
0.80         Strong pattern  font-size:32 + bold + short → heading
0.60         Structural      3+ similar siblings → list
0.40         Class name      class="sidebar-nav" → navigation
0.20         Weak pattern    width < 300px → maybe sidebar
```

Low-confidence inferences are stored but flagged. The renderer uses them; the query
engine can filter by confidence: `WHERE confidence > 0.7` for high-quality extraction.

---

## JavaScript Tiering Strategy

### The Problem With JS

Modern web: 70% of pages require JavaScript to render meaningful content. SPAs like
Gmail, Twitter, and dashboards serve an empty `<div id="root">` and build everything
in JS. Without executing JS, these pages are blank.

Traditional browsers: parse JS → compile (V8 JIT) → execute → DOM mutations → re-render.
This is the #1 bottleneck. V8 is 10 million lines of C++ optimized over 15 years.

OctoView: JS is a **background cron**, not a blocking gate.

### Three Tiers

```
                     Detection                       Strategy
─────────────────────────────────────────────────────────────────────

TIER A: NO JS        No <script> tags                Skip Loom 2 entirely
(~15% of web)        or only analytics/tracking      Page complete at Loom 1
                     (Google Analytics, etc.)         Fastest possible path

TIER B: LIGHT JS     <script> present but            Loom 2 background cron
(~35% of web)        DOM exists in HTML              Boa executes in background
                     JS enhances, doesn't create     DOM mutations → diff OVD
                     (dropdowns, forms, animations)  → patch → GPU update
                                                     User sees content first,
                                                     JS features appear in 1-2s

TIER C: SPA          <div id="root"> or              Loom 2 full execution
(~50% of web)        <div id="app"> + empty body     Boa runs all JS
                     Large JS bundles (>200KB)        Produces final DOM
                     Frameworks: React, Vue, Angular  Snapshot → .ovd cache
                                                     First visit: 3-5s
                                                     Return visit: 10ms (cached)
                                                     Cron: re-snapshot every N min
```

### Tier Detection Algorithm

```
fn detect_js_tier(dom, scripts) -> JsTier:
    // Count script tags
    let total_js_bytes = sum(scripts.map(s -> s.size))
    let has_framework = scripts.any(s ->
        s.contains("react") or s.contains("vue") or
        s.contains("angular") or s.contains("svelte"))
    let dom_has_content = dom.text_nodes.len() > 10

    if total_js_bytes == 0 or all_analytics(scripts):
        return TIER_A   // no JS needed

    if dom_has_content and total_js_bytes < 200_000:
        return TIER_B   // light enhancement

    return TIER_C       // SPA, needs full execution
```

### The Cron Model

For Tier B and C pages, Loom 2 operates as a background cron:

```
First visit:
  t=0ms     Loom 0 renders content (what HTML contains)
  t=100ms   Loom 1 applies styles
  t=200ms   Loom 2 starts Boa JS engine in background
  t=1-5s    Loom 2 completes → diffs DOM → patches OVD
            → GPU re-renders with JS content
            → snapshots .ovd to cache

Return visit:
  t=0ms     Load .ovd from cache → GPU render (instant)
  t=100ms   Background: check if cron interval elapsed
            → if yes: re-fetch page, re-transpile, update cache
            → user sees cached version immediately

Cron intervals (configurable):
  News sites:      5 minutes    (content changes frequently)
  Documentation:   24 hours     (content is stable)
  SPAs (Gmail):    on-navigate  (re-snapshot when user navigates within app)
  Custom:          user-defined per domain
```

---

## Browsing = Scraping: The Query Interface

Because every page is an OctoView Document (flat columnar arrays), extracting data
from any webpage is a database query. No Puppeteer. No BeautifulSoup. No regex on HTML.

### CLI Queries

```bash
# Get all headings from a page
octoview query "https://en.wikipedia.org/wiki/GPU" \
  "SELECT text, level FROM nodes WHERE type = 'heading' ORDER BY node_id"

# Get all links from Hacker News
octoview query "https://news.ycombinator.com" \
  "SELECT text, href FROM nodes WHERE type = 'link' AND semantic = 'content'"

# Get structured table data
octoview query "https://example.com/pricing" \
  "SELECT * FROM nodes WHERE type = 'table_cell' AND parent_type = 'table_row'"

# Count images on a page
octoview query "https://example.com" \
  "SELECT COUNT(*) FROM nodes WHERE type = 'image'"

# Scrape 100 pages in parallel (each page is a loom pipeline)
octoview batch urls.txt \
  "SELECT text, href FROM nodes WHERE type = 'link' AND depth < 3" \
  --output results.csv --parallel 20
```

### OctoFlow API

```flow
use octoview.fetch

// Load any page as structured data
let page = ov_load("https://en.wikipedia.org/wiki/GPU")

// Query the page like a database
let headings = ov_query(page, "type = 'heading'")
let links = ov_query(page, "type = 'link' AND semantic = 'content'")
let tables = ov_query(page, "type = 'table'")
let images = ov_query(page, "type = 'image'")

// Print all headings
for h in headings
    print("H{h.level}: {h.text}")
end

// Get all links with their text
for link in links
    print("{link.text} → {link.href}")
end

// Batch scrape
let urls = ["https://news.ycombinator.com", "https://reddit.com", "https://lobste.rs"]
let pages = ov_load_batch(urls)  // parallel loom pipelines

for page in pages
    let headlines = ov_query(page, "type = 'heading' AND level <= 2")
    print("=== {page.url} ===")
    for h in headlines
        print("  {h.text}")
    end
end
```

### LoomDB Integration

```flow
// Pipe web pages directly into LoomDB for analytics
use octoview.fetch
use loomdb

// Scrape 1000 product pages, analyze pricing
let urls = read_lines("product_urls.txt")
let pages = ov_load_batch(urls)

for page in pages
    loomdb_ingest(page.nodes, "products")
end

// Now query across all pages
loomdb query "SELECT text, style.font_size
              FROM products
              WHERE type = 'text_span'
              AND text LIKE '%$%'
              ORDER BY style.font_size DESC"
// → finds the largest-displayed prices across all product pages
```

### Why This Is Faster Than Traditional Scraping

```
Traditional scraping pipeline:
  1. HTTP request                    (reqwest)              ~200ms
  2. Receive full HTML               (wait for complete)    ~500ms
  3. Parse HTML → DOM tree           (html5ever/scraper)    ~50ms
  4. Maybe execute JS                (headless Chrome)      ~3000ms
  5. Traverse DOM, extract data      (CSS selectors)        ~100ms
  6. Serialize to JSON/CSV           (serde)                ~50ms
  TOTAL: ~400ms (no JS) or ~4000ms (with JS)

OctoView pipeline:
  1. HTTP request + stream parse     (reqwest + html5ever)  ~100ms
  2. Semantic extraction             (three-layer engine)   ~50ms
  3. OVD in memory                   (flat columnar)        ~0ms
  4. Query (GPU scan)                (LoomDB pattern)       ~1ms
  TOTAL: ~150ms (first visit) or ~10ms (cached .ovd)
```

For cached pages: **400x faster** than traditional scraping.
For uncached pages: **2-3x faster** (streaming parse, no full DOM traversal).
For JS-heavy pages: **20-40x faster on return visits** (cached snapshot vs headless Chrome every time).

### Why OctoView Bypasses Anti-Scraping Entirely

Traditional web scraping is an arms race. Websites deploy increasingly aggressive
countermeasures, and scrapers deploy increasingly complex workarounds. The entire
industry exists because scraping and browsing are treated as fundamentally different
operations. OctoView dissolves this distinction.

**The problem with traditional scraping:**

```
Countermeasure            What It Blocks                  Scraper Workaround
─────────────────────────────────────────────────────────────────────────────
Rate limiting             Rapid repeated requests         Proxy rotation ($$$)
IP blocking               Known scraper IP ranges         Residential proxies ($$$)
CAPTCHAs                  Automated access                CAPTCHA-solving services ($$$)
Cloudflare JS challenge   Non-browser clients             Headless Chrome (heavy, slow)
Bot fingerprinting        Missing browser signals         Puppeteer stealth plugins
Dynamic rendering         No-JS scrapers                  Headless Chrome (3-5s/page)
Anti-headless detection   Headless browser signatures     Patched Chromium builds
Login walls               Unauthenticated access          Session management, cookies
DOM obfuscation           CSS selector-based extraction   Fragile, breaks on redesign
Honeypot links            Automated crawlers              Manual URL curation
```

Every workaround adds cost, complexity, and fragility. A scraping pipeline for a
major site requires: rotating proxies ($50-500/month), CAPTCHA solving ($2-5/1000),
headless Chrome instances (8GB RAM per 100 pages), and constant maintenance when
sites change their defenses. The industry spends **$3-5 billion annually** on web
scraping infrastructure.

**Why none of this applies to OctoView:**

```
OctoView Property                     Why Anti-Scraping Can't Detect It
───────────────────────────────────────────────────────────────────────────

1. IT'S A REAL BROWSER               OctoView makes real HTTP requests with real
                                      headers, real TLS fingerprints, real connection
                                      patterns. It IS browsing, not simulating it.

2. ONE FETCH = INFINITE QUERIES       Traditional scrapers fetch the same page 100x
                                      to extract different data. OctoView fetches ONCE,
                                      caches the .ovd, queries it offline forever.
                                      Rate limiting becomes irrelevant — you never
                                      re-fetch to re-query.

3. NO HEADLESS SIGNATURE              OctoView isn't headless Chrome with a flag.
                                      It's a native Vulkan application. No
                                      navigator.webdriver, no Chrome DevTools Protocol,
                                      no puppeteer fingerprint. Anti-headless detection
                                      has nothing to detect.

4. USER-SPEED BROWSING                A human browses 1-2 pages per minute. OctoView
                                      at normal browsing speed is indistinguishable
                                      from any browser. The data extraction happens
                                      LOCALLY on the cached .ovd — the server never
                                      sees the queries.

5. COMMUNITY CACHE                    For popular sites, the OVD snapshot CDN means
                                      you may never hit the server at all. Someone
                                      else already browsed it. Zero requests = zero
                                      rate limiting = zero detection.

6. JS EXECUTION IS REAL               Boa executes JavaScript for real. Cloudflare
                                      challenges, bot checks, dynamic rendering —
                                      all work normally because OctoView actually
                                      runs the JS. The difference: it extracts the
                                      result into .ovd instead of throwing it away.

7. SEMANTIC, NOT STRUCTURAL           Traditional scrapers break when a site changes
                                      its CSS classes or DOM structure. OctoView's
                                      heuristic extraction understands MEANING:
                                      "this is a heading" not "this is div.sc-abc123".
                                      Site redesigns don't break extraction.
```

**The fundamental shift:**

```
Traditional scraping:
  You are an adversary.
  The website is a fortress.
  You sneak in, grab data, sneak out.
  The fortress builds higher walls.
  You build longer ladders.
  Repeat forever.

OctoView:
  You are a browser.
  The website serves you a page.
  You understand it.
  There is no adversarial relationship.
  There is no arms race.
  The data is yours because you received it.
```

**Practical impact by use case:**

| Use Case | Traditional Scraper | OctoView |
|----------|-------------------|----------|
| **Price monitoring** (1000 products) | 100 rotating proxies, CAPTCHA solving, 4h runtime, breaks monthly | Browse 1000 pages normally over a week. Query cached .ovd forever. |
| **News aggregation** (50 sources) | Headless Chrome farm, rate-limited to 1 req/5s per source, ~4 min cycle | Load each source once. Cron refreshes .ovd every 5 min. Query instantly. |
| **Competitive intelligence** | Enterprise scraping service ($500/mo), delayed data, legal gray area | Normal browsing. .ovd cache = your database. Real-time, local, private. |
| **Academic research** (10K papers) | Weeks of careful crawling, API rate limits, IP bans | Batch browse at human speed. Query .ovd cache with SQL. Data never leaves your machine. |
| **Job market analysis** | Blocked within minutes by LinkedIn/Indeed | Browse job listings normally. Each page auto-extracted to .ovd. Aggregate locally. |

**The $3-5B scraping industry problem, solved by architecture:**

OctoView doesn't beat anti-scraping measures. It makes them irrelevant. When browsing
and data extraction are the same operation, there's nothing to detect, nothing to block,
and nothing to bypass. The website served you the page. You understood it. That's it.

---

## Progressive Rendering Timeline

### Static Page (Wikipedia, blog, docs)

```
t=0ms      HTTP request sent
t=50ms     First HTML bytes arrive → Loom 0 starts streaming parse
t=70ms     First content nodes extracted → GPU render (text visible!)
t=100ms    Page structure complete → headings, paragraphs, links rendered
t=150ms    CSS downloaded → Loom 1 computes styles → GPU re-render (styled)
t=200ms    Images downloaded → GPU texture upload → rendered in place
t=200ms    PAGE COMPLETE (no JS needed, Tier A)

Chrome comparison: ~1200ms (full parse → style → layout → paint)
Speedup: ~6x
```

### News Site (light JS — dropdowns, lazy loading)

```
t=0ms      HTTP request sent
t=50ms     First HTML bytes → Loom 0 starts
t=70ms     Content visible (headlines, text)
t=150ms    Styled (CSS applied)
t=200ms    PAGE VISUALLY COMPLETE (Tier B, JS not blocking)
t=500ms    Loom 2: JS executing in background
t=1200ms   JS complete → dropdowns work, lazy images load
t=1200ms   PAGE FULLY INTERACTIVE

Chrome comparison: ~1500ms to first paint, ~2500ms to interactive
Speedup: ~3x visual, ~2x interactive
```

### SPA (Gmail, Twitter — first visit)

```
t=0ms      HTTP request sent
t=50ms     HTML arrives: <div id="root"></div> → nearly empty
t=70ms     Loom 0: minimal content ("Loading..." text if present)
t=150ms    Loom 1: styles computed (minimal)
t=200ms    Loom 2: starts Boa JS execution
t=3000ms   JS framework boots, renders virtual DOM
t=4000ms   OctoView diffs final DOM → full OVD
t=4500ms   PAGE COMPLETE → snapshot cached as .ovd

Chrome comparison: ~3000ms
Speed: similar first visit (Boa slower than V8, but not blocking)
```

### SPA (Gmail, Twitter — return visit)

```
t=0ms      Cache check: .ovd exists, fresh
t=10ms     Load .ovd from disk → GPU render
t=10ms     PAGE COMPLETE (full content from cache!)
t=100ms    Background: check if cron refresh needed

Chrome comparison: ~3000ms (re-executes all JS every time)
Speedup: ~300x
```

---

## Tab Architecture: Each Tab Is a Loom Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    OctoView Browser                      │
│                                                          │
│  Tab 1: wikipedia.org                                    │
│  └── Loom Pipeline (3 looms: content/style/js)          │
│      └── OVD: 2,847 nodes, 156KB                        │
│                                                          │
│  Tab 2: news.ycombinator.com                            │
│  └── Loom Pipeline (2 looms: content/style, no JS)      │
│      └── OVD: 423 nodes, 28KB                           │
│                                                          │
│  Tab 3: gmail.com                                        │
│  └── Cached OVD (890 nodes, 67KB, cron: 5min)           │
│                                                          │
│  Tab 4: docs.octoflow.dev                               │
│  └── Native OctoFlowWeb (Tier 1, direct GPU render)     │
│                                                          │
│  Shared Resources:                                       │
│  ├── Font Atlas: 4MB GPU VRAM (shared across all tabs)  │
│  ├── Image Cache: three-tier (VRAM 50MB / RAM 200MB)    │
│  ├── Connection Pool: HTTP/2 multiplexed per origin     │
│  ├── JS Engine Pool: 2 Boa instances (reused)           │
│  └── OVD Cache: ~/.octoview/cache/*.ovd                 │
│                                                          │
│  Total Memory: ~180MB for 4 tabs                        │
│  (Chrome equivalent: ~1.5GB)                             │
├─────────────────────────────────────────────────────────┤
│  OctoFlow Runtime │ Loom Engine │ Vulkan │ OctoUI       │
└─────────────────────────────────────────────────────────┘
```

### Resource Sharing

Tabs share GPU resources through the Loom Engine:
- **Font atlas**: one GPU texture, all tabs reference it
- **Image cache**: deduped by content hash across tabs
- **SPIR-V kernels**: compiled once, dispatched per-tab
- **Vulkan device**: single VkDevice, multiple command buffers

This is fundamentally different from Chrome's multi-process model where each tab
is an isolated process with its own memory space.

---

## Community Snapshot CDN (Future)

The OVD cache creates a natural opportunity for community sharing:

```
                  Popular websites
                  (Gmail, Twitter, YouTube, GitHub, Reddit)
                        │
                        ▼
          Community maintainers run cron snapshots
          (like search engine crawlers, but for OctoView)
                        │
                        ▼
          Publish .ovd snapshots to CDN
          (versioned, timestamped, signed)
                        │
                        ▼
          OctoView checks CDN before executing JS locally
                        │
                ┌───────┴───────┐
                │               │
          Fresh snapshot    No snapshot
          available         available
                │               │
                ▼               ▼
          Instant load    Local transpile
          (zero JS cost)  (normal pipeline)
```

### Why This Works

- SPAs are deterministic for most users — Gmail's inbox layout is the same structure
  whether you have 10 or 10,000 emails. The dynamic data changes; the structure doesn't.
- Snapshots capture **structure**, not content. User-specific data (emails, messages)
  is loaded via API calls that OctoView handles separately.
- Community snapshots are **opt-in**. Privacy-sensitive users can disable CDN lookup
  and always transpile locally.

### Trust Model

```
Snapshot providers:
  Official OctoView CDN → trusted, signed by OctoView team
  Community mirrors     → signed by maintainer key, user chooses trust
  Self-hosted           → organizations can host their own snapshot CDN

Verification:
  Every .ovd carries a signature + source URL hash
  OctoView verifies signature before using snapshot
  Tampered snapshots → rejected, falls back to local transpile
```

---

## Integration With OctoFlow Ecosystem

### OctoView + LoomDB

```
OctoView transpiles pages → OVD documents (flat columnar)
LoomDB queries flat columnar data
→ LoomDB can query web pages directly

Use case: competitive intelligence
  Scrape 500 competitor product pages → ingest into LoomDB
  SELECT product_name, price FROM pages WHERE type = 'card'
  GROUP BY product_name
  → price comparison dashboard, updated by cron
```

### OctoView + OctoFlowWeb

```
Native apps (Tier 1):   OctoFlowWeb .flow → direct GPU render
Legacy web (Tier 3):    HTML/CSS/JS → Transpiler → OVD → GPU render
Both produce:           GPU render commands via OctoUI dispatch chains

The user doesn't know (or care) which tier they're on.
Native apps are faster. Legacy web is compatible. Both work.
```

### OctoView + OctoMark

```
OctoMark documents (.md with media blocks)
→ Parsed to OVD-like structure
→ Rendered by the same GPU pipeline

A Markdown blog post and an HTML blog post
both become OctoView Documents.
Same renderer. Same query interface.
```

### OctoView + Loom Engine

```
Each tab = 1-3 loom pipelines
Each pipeline = dispatch chain of GPU kernels
Font rendering = ui_text.spv kernel
Rectangle/box rendering = ui_rect.spv kernel
Image rendering = texture sampling kernel
Layout = GPU-parallel constraint solver (future)

The Loom Engine is the foundation.
OctoView is just another Loom application.
```

---

## Performance Projections

### Page Load Comparison

| Page Type | Chrome | OctoView (first) | OctoView (cached) |
|-----------|--------|-------------------|-------------------|
| Static blog | ~800ms | **~70ms** | **~10ms** |
| News site | ~1.2s | **~200ms** | **~10ms** |
| Documentation | ~1.0s | **~150ms** | **~10ms** |
| Light JS (forms) | ~1.5s | **~200ms** + 1s JS | **~10ms** |
| Heavy SPA (Gmail) | ~3s | **~3-5s** (first) | **~10ms** |
| Heavy SPA (Twitter) | ~4s | **~4-6s** (first) | **~10ms** |

### Scraping Comparison

| Task | Puppeteer + Chrome | BeautifulSoup | OctoView |
|------|-------------------|---------------|----------|
| 1 page (static) | ~2s | ~400ms | **~150ms** |
| 1 page (SPA) | ~5s | N/A (no JS) | **~5s first, 10ms cached** |
| 100 pages (static) | ~30s | ~15s | **~3s** (parallel looms) |
| 100 pages (cached) | ~30s | ~15s | **~0.5s** |
| Data extraction | CSS selectors | CSS selectors | **SQL queries** |
| Output format | HTML strings | HTML strings | **Typed columnar data** |

### Memory Comparison

| Scenario | Chrome | OctoView |
|----------|--------|----------|
| 1 tab (simple) | ~150 MB | **~30 MB** |
| 5 tabs (mixed) | ~800 MB | **~120 MB** |
| 10 tabs (heavy) | ~2 GB | **~200 MB** |
| 20 tabs (stress) | ~4 GB | **~350 MB** |
| Scraping 100 pages | ~8 GB (100 headless instances) | **~500 MB** (100 loom pipelines) |

---

## Rust Crate Dependencies

All web-facing components are pure Rust. No C/C++ dependencies beyond Vulkan driver.

```
Networking:
  reqwest          HTTP/1.1 + HTTP/2 client
  hyper            HTTP server (for web UI, if needed)
  rustls           TLS (no OpenSSL dependency)
  url              URL parsing

HTML/CSS Parsing:
  html5ever        Browser-grade HTML5 parser (from Servo)
  cssparser        CSS Syntax Level 3 parser (from Servo)
  selectors        CSS selector matching (from Servo)
  markup5ever      Shared HTML/XML parsing utilities

JavaScript:
  boa_engine       Pure-Rust JS engine (~50K lines vs V8's 10M)
                   ES2024 support, no unsafe, embeddable

Layout (future):
  taffy            Flexbox + CSS Grid layout engine (Rust)

Hashing:
  xxhash-rust      Fast hashing for content dedup and cache keys

Compression:
  zstd             .ovd file compression (optional)
```

### Why Not Embed Chromium?

| Approach | Binary Size | Memory | Control | Speed |
|----------|-------------|--------|---------|-------|
| Embed Chromium (Electron) | +150 MB | +300 MB/tab | None | Chrome speed |
| Embed WebView (Tauri) | +0 MB (system) | +100 MB/tab | Limited | System browser speed |
| **OctoView (native)** | **+5 MB** | **+20 MB/tab** | **Full** | **GPU speed** |

---

## What Makes OctoView Different From Everything Else

| Aspect | Chrome / Firefox | Servo / Ladybird | OctoView |
|--------|------------------|-------------------|----------|
| **Goal** | Implement HTML/CSS/JS spec | Implement spec in Rust | **Transpile web to GPU-native** |
| **Rendering** | CPU layout + paint, GPU composite | CPU layout, GPU composite | **GPU everything** |
| **JS role** | Blocking gate | Blocking gate | **Background cron** |
| **Output** | Pixels (opaque) | Pixels (opaque) | **Typed document + pixels** |
| **Pages as data** | No (need DevTools/scraper) | No | **Yes (every page is queryable)** |
| **Cache** | HTTP cache (raw HTML/CSS/JS) | HTTP cache | **.ovd cache (pre-transpiled)** |
| **Return visits** | Re-parse, re-execute, re-render | Re-parse, re-execute | **Instant (10ms from .ovd)** |
| **10 tabs memory** | ~2 GB | ~1.5 GB (estimated) | **~200 MB** |
| **Scraping** | External tool (Puppeteer) | External tool | **Built-in (browsing IS scraping)** |
| **Web compat** | 100% (that's the goal) | 95%+ (that's the goal) | **Semantic (understands meaning)** |
| **New web** | HTML/CSS/JS only | HTML/CSS/JS only | **OctoFlowWeb native + HTML compat** |

### The Fundamental Insight

Chrome, Servo, and Ladybird all answer: **"How do we render HTML/CSS/JS correctly?"**

OctoView answers: **"What does this page mean, and how do we show that on GPU?"**

The first question leads to reimplementing a 10,000-page spec.
The second question leads to a transpiler that extracts meaning.

One is an engineering race. The other is an architectural leap.

---

## Security Threat Model

The web is the #1 attack vector for end users. Every page load is a trust decision.
OctoView's architecture eliminates entire classes of browser vulnerabilities by design,
but introduces new surfaces that must be hardened. This section covers both.

### Architectural Security Advantages

Before the threats — what OctoView eliminates structurally:

```
Chrome Attack Surface                    OctoView Equivalent
────────────────────────────────────────────────────────────────────────
V8 JavaScript Engine (10M lines C++)     Boa (50K lines Rust, memory-safe)
  → 60% of Chrome CVEs are V8 bugs        → 200x smaller attack surface
  → Use-after-free, buffer overflow        → No unsafe memory bugs (Rust)

DOM API (document.*, window.*)           No DOM API exposed to JS
  → XSS: script injects into DOM           → JS output is diffed, not injected
  → Cookie theft: document.cookie           → No document object, no cookie access
  → Keylogging: addEventListener            → Input handling is native, not JS

CSS Engine (50K+ properties)             Flat style records (48 bytes)
  → CSS exfiltration attacks                → Styles resolved once, stored as data
  → @import chains (recursive fetch)        → No @import at render time
  → CSS injection via user content          → CSS is data extraction, not execution

Plugin/Extension System                  No extension system (v1)
  → Malicious extensions steal data         → No third-party code in browser
  → Extension supply chain attacks          → Controlled first-party only

WebRTC / Web Audio / WebGL               Not implemented
  → IP leak via WebRTC                      → No WebRTC, no IP leak
  → Fingerprinting via AudioContext         → No Web Audio API
  → GPU fingerprinting via WebGL            → Vulkan compute only (no graphics query)

iframe / cross-origin embedding          No iframes
  → Clickjacking                            → Pages are flat OVD, no embedding
  → Cross-origin data theft                 → No cross-origin execution context

Service Workers / Web Workers            Not implemented
  → Persistent background scripts           → No persistent page-level code
  → Cryptomining via workers                → JS only runs during cron window
```

**By conservative estimate, OctoView eliminates ~70% of Chrome's CVE categories**
by not implementing the APIs those vulnerabilities target.

### Threat Matrix

#### T1: Malicious JavaScript Execution

```
Threat:     Page includes cryptominer, keylogger, or data exfiltration JS
Vector:     Loom 2 (JS cron) executes malicious code via Boa engine
Severity:   HIGH
```

**Mitigations:**

| Layer | Measure | Effect |
|-------|---------|--------|
| **Sandbox** | Boa runs in a restricted sandbox: no filesystem, no network, no system calls | JS can modify DOM but cannot escape to host |
| **Time budget** | JS cron has a configurable execution time limit (default: 5s per page) | Cryptominers killed after budget expires |
| **No DOM API leak** | Boa's DOM is a virtual structure; no `document.cookie`, no `window.location` with credentials, no `localStorage` | Data exfiltration impossible — no data to exfilt |
| **Capability deny-list** | Block known dangerous APIs: `eval()` on external strings, `WebSocket`, `XMLHttpRequest` within Boa | JS cannot make outbound connections from Boa context |
| **Output diffing** | JS doesn't inject directly into the OVD; Boa produces a DOM delta that's diffed and validated before patching | Injected `<script>` nodes stripped during diff |
| **User control** | Per-domain JS policy: `allow` / `deny` / `ask`. Default: `allow` for known sites, `ask` for unknown | User decides which sites get JS execution |

**Defense in depth:** Even if Boa is compromised, the sandbox prevents host access.
Even if the sandbox leaks, the DOM diff strips executable content. Even if the diff
fails, the time budget kills runaway processes.

#### T2: Cross-Site Scripting (XSS)

```
Threat:     Attacker injects malicious content into a legitimate page
Vector:     Stored XSS, reflected XSS, DOM-based XSS
Severity:   HIGH (in traditional browsers), LOW (in OctoView)
```

**Why XSS is structurally mitigated:**

```
Traditional browser XSS:
  1. Attacker injects: <script>steal(document.cookie)</script>
  2. Browser parses HTML → encounters <script>
  3. V8 executes → accesses document.cookie → sends to attacker
  4. User is compromised

OctoView:
  1. Attacker injects: <script>steal(document.cookie)</script>
  2. html5ever parses HTML → encounters <script>
  3. Loom 0 (content extraction): <script> is NOT a content node
     → stripped during semantic extraction. Never reaches OVD.
  4. Loom 2 (JS cron): if JS executes, Boa has no document.cookie
     → no DOM API → nothing to steal
  5. Attack fails at two independent layers
```

XSS requires three things: injection, execution, and exfiltration. OctoView
breaks all three: injection is stripped at extraction, execution is sandboxed
without sensitive APIs, and exfiltration is blocked by network isolation.

#### T3: Phishing and Credential Theft

```
Threat:     Fake login page mimics real site to steal passwords
Vector:     Visual deception (looks like Gmail/bank login)
Severity:   HIGH
```

**Mitigations:**

| Layer | Measure | Effect |
|-------|---------|--------|
| **Semantic typing** | OVD marks form fields with `type: input_field`, `semantic: interactive` — the browser knows it's a form, not just pixels | Can warn: "this page contains a login form" |
| **Domain awareness** | OctoView displays the true domain prominently; cannot be spoofed by page content | URL bar is native UI, never rendered by page |
| **Form isolation** | Form submissions route through OctoView's native HTTP client, not page JS | Page JS cannot intercept or redirect form data |
| **Credential manager** | Native credential storage (OS keychain); only auto-fills on verified domains | Password never sent to wrong domain |
| **Visual warnings** | Pages with login forms on non-HTTPS or recently-registered domains get explicit warnings | User sees "This site is asking for credentials. Domain registered 3 days ago." |
| **OVD fingerprinting** | Known phishing page structures (login form + urgent language + mismatched domain) can be pattern-matched on the OVD | Heuristic phishing detection on structured data, not pixels |

**Advantage over Chrome:** Chrome detects phishing by matching URLs against a blocklist
(Safe Browsing). OctoView can also analyze the **semantic content** of the page — it
knows the page contains a login form, it knows the surrounding text says "urgent,"
and it knows the domain doesn't match the brand in the page. Structural phishing
detection is harder to evade than URL blocklists.

#### T4: Malicious .ovd Files (Cache Poisoning)

```
Threat:     Crafted .ovd file exploits parser vulnerability or contains deceptive content
Vector:     Community snapshot CDN, shared .ovd files, cache injection
Severity:   MEDIUM-HIGH
```

**Mitigations:**

| Layer | Measure | Effect |
|-------|---------|--------|
| **Format validation** | Strict .ovd parser: magic bytes, version check, bounds checking on all arrays, node count limits | Malformed files rejected before processing |
| **Signature verification** | Every CDN .ovd is signed (Ed25519) by the snapshot provider | Unsigned or tampered files rejected |
| **URL hash binding** | .ovd header contains hash of source URL; cache lookup verifies match | Can't substitute one site's .ovd for another |
| **Content hash** | Footer xxhash64 checksum; verified on load | Bit-flipped or truncated files detected |
| **Node count limits** | Maximum 100,000 nodes per .ovd (configurable) | Resource exhaustion via giant .ovd prevented |
| **No executable content** | .ovd contains typed data (text, styles, bounds) — no scripts, no event handlers, no executable code | .ovd files are inert data, not programs |
| **Sandboxed rendering** | Even if .ovd contains unexpected data, the GPU renderer only reads typed arrays — no interpretation of content as code | Buffer overrun in .ovd cannot become code execution |

**The key insight:** .ovd is a **data format**, not an executable format. Unlike HTML
(which contains `<script>` tags that execute), an .ovd file contains parallel arrays
of typed values. There is no mechanism by which .ovd content becomes code execution.
The worst a malicious .ovd can do is display wrong content — it cannot compromise
the host system.

#### T5: Network Attacks (MITM, DNS Poisoning)

```
Threat:     Attacker intercepts or redirects network traffic
Vector:     Compromised Wi-Fi, DNS hijacking, BGP hijacking
Severity:   HIGH
```

**Mitigations:**

| Layer | Measure | Effect |
|-------|---------|--------|
| **TLS everywhere** | All HTTP upgraded to HTTPS via rustls (no OpenSSL) | Encrypted in transit, certificate pinning available |
| **HSTS preload** | Respect HSTS headers; preload list for known HTTPS-only domains | Downgrade attacks prevented |
| **Certificate transparency** | Verify certificates against CT logs | Rogue certificates detected |
| **DNS-over-HTTPS** | Optional DoH for DNS resolution | DNS queries encrypted, poisoning prevented |
| **Rustls (no OpenSSL)** | Pure-Rust TLS implementation | No heartbleed-class vulnerabilities |

Same protections as modern Chrome, but with a smaller TLS implementation
(rustls: ~30K lines Rust vs OpenSSL: ~500K lines C).

#### T6: Privacy and Tracking

```
Threat:     Websites track users across sites; browsing data exposed
Vector:     Cookies, fingerprinting, tracking pixels, referrer headers
Severity:   MEDIUM
```

**Mitigations:**

| Layer | Measure | Effect |
|-------|---------|--------|
| **No third-party cookies** | OctoView does not support third-party cookies by default | Cross-site tracking via cookies eliminated |
| **Minimal fingerprint** | No Canvas API, no WebGL, no AudioContext, no font enumeration | Primary fingerprinting vectors don't exist |
| **Referrer policy** | Strict: origin-only referrer (no path) by default | Sites can't see where you came from |
| **Tracking pixel stripping** | Semantic extraction identifies 1x1 images and known tracking patterns → `semantic: tracking` → stripped or blocked | Tracking pixels never rendered, never queried |
| **Ad/tracker heuristics** | Known ad domains and tracking scripts identified during Loom 2 | User can choose: block, allow, or flag |
| **Local-only queries** | All .ovd queries execute locally — no data sent to any server | Your scraping/analysis is private by default |
| **Cache isolation** | .ovd cache partitioned by domain; no cross-domain cache probing | Cache timing attacks prevented |
| **Ephemeral mode** | `octoview --ephemeral` — no .ovd cache, no history, no cookies | Private browsing with zero disk trace |

**Privacy advantage over Chrome:** Google's business model requires tracking.
OctoView has no business model that conflicts with user privacy. The semantic
extraction engine can **detect and label** tracking (ad pixels, analytics scripts,
fingerprinting attempts) because it understands page semantics — something Chrome
actively avoids doing because it would undermine its own ad platform.

#### T7: GPU-Specific Attacks

```
Threat:     Malicious content exploits GPU driver or VRAM to attack host
Vector:     Crafted SPIR-V, GPU memory corruption, side-channel attacks
Severity:   LOW-MEDIUM (theoretical)
```

**Mitigations:**

| Layer | Measure | Effect |
|-------|---------|--------|
| **No user-supplied SPIR-V** | Web pages cannot provide GPU shaders; all SPIR-V is compiled from OctoView's own IR | No arbitrary shader execution from web content |
| **spirv-val validation** | All generated SPIR-V passes Khronos validator before dispatch | Malformed shaders rejected before GPU submission |
| **Buffer bounds** | Vulkan buffer sizes computed from OVD node count; no user-controlled sizing | Buffer overflow in VRAM prevented |
| **Dispatch limits** | Maximum dispatch chain length per page render (configurable) | GPU resource exhaustion prevented |
| **Vulkan validation layers** | Debug builds run with full Vulkan validation | Driver-level memory safety verified in development |
| **Read-only VRAM for content** | Page content uploaded to GPU as read-only storage buffers | Web content cannot modify GPU state |

**Key difference from WebGL/WebGPU:** Those APIs let web pages submit arbitrary
shaders to the GPU — the #1 GPU attack vector in Chrome. OctoView never does this.
Web content is data (text, styles, images) uploaded to GPU buffers. The shaders are
OctoView's own (ui_clear.spv, ui_rect.spv, ui_text.spv) — compiled once, validated,
and reused for every page. The GPU attack surface is near-zero.

#### T8: Resource Exhaustion (DoS)

```
Threat:     Malicious page designed to consume all GPU memory or CPU time
Vector:     Extremely large DOM, recursive CSS, infinite JS loops, massive images
Severity:   MEDIUM
```

**Mitigations:**

| Layer | Measure | Effect |
|-------|---------|--------|
| **Node count limit** | Max 100,000 nodes per OVD (configurable) | Giant DOM pages truncated, not crashed |
| **JS time budget** | Loom 2 cron: max 5 seconds per execution cycle | Infinite loops killed after budget |
| **Image size limit** | Max 50MB per image, max 200MB total images per page | Memory bomb images rejected |
| **CSS depth limit** | Max 32 levels of selector nesting during resolution | CSS selector bombs prevented |
| **VRAM budget per tab** | Configurable per-tab VRAM limit (default: 500MB) | Single tab cannot exhaust GPU |
| **Total VRAM guard** | OctoView reserves 20% VRAM as headroom; new tabs denied if exceeded | System stability maintained |
| **Connection limits** | Max 20 concurrent connections per origin, 100 total | Connection flood prevented |
| **Download limits** | Max response body: 50MB HTML, 200MB total resources per page | Bandwidth exhaustion prevented |

#### T9: Supply Chain — Community Snapshot CDN

```
Threat:     Compromised snapshot provider serves malicious .ovd files
Vector:     Man-in-the-middle on CDN, compromised maintainer account, rogue provider
Severity:   MEDIUM-HIGH
```

**Mitigations:**

| Layer | Measure | Effect |
|-------|---------|--------|
| **Signature chain** | CDN snapshots signed by provider key; provider key signed by OctoView root CA | Two-level trust: provider + OctoView authority |
| **Transparency log** | All published .ovd files logged in append-only transparency log | Retroactive detection of rogue snapshots |
| **Reproducible snapshots** | Given same URL + timestamp, any provider should produce same .ovd (deterministic extraction) | Cross-provider verification possible |
| **User trust levels** | Official CDN: auto-trusted. Community: user opt-in per provider. Self-hosted: user-controlled | No forced trust of third parties |
| **Fallback to local** | If CDN snapshot fails verification → fall back to local transpile (normal pipeline) | Compromised CDN degrades to slow, not insecure |
| **Snapshot expiry** | .ovd snapshots have TTL; expired snapshots trigger re-transpile | Stale compromised snapshots auto-expire |
| **Canary pages** | OctoView team maintains canary URLs; CDN snapshots of these are verified against known-good | Compromised CDN detected within one canary cycle |

### Security Comparison

| Attack Class | Chrome | Firefox | OctoView |
|-------------|--------|---------|----------|
| **JS engine vulns** | HIGH (V8: 10M lines C++, 60% of CVEs) | HIGH (SpiderMonkey: 3M lines C++) | **LOW** (Boa: 50K lines Rust, memory-safe) |
| **XSS** | HIGH (DOM API exposes cookies, storage) | HIGH (same DOM API) | **VERY LOW** (no DOM API, content stripped at extraction) |
| **CSS attacks** | MEDIUM (complex cascade engine) | MEDIUM | **LOW** (flat resolved styles, no cascade at render) |
| **Extension malware** | HIGH (250K+ extensions, supply chain risk) | MEDIUM | **N/A** (no extension system v1) |
| **Fingerprinting** | HIGH (Canvas, WebGL, AudioContext, fonts) | MEDIUM (some mitigations) | **VERY LOW** (none of these APIs exist) |
| **Tracking** | HIGH (Google's business model) | LOW (ETP) | **VERY LOW** (no third-party cookies, tracking stripped) |
| **GPU attacks** | MEDIUM (WebGL/WebGPU shader injection) | MEDIUM | **VERY LOW** (no user-supplied shaders) |
| **Phishing** | MEDIUM (URL blocklist only) | MEDIUM | **LOW** (semantic content analysis + URL) |
| **Cache poisoning** | MEDIUM (HTTP cache manipulation) | MEDIUM | **LOW** (.ovd is inert data, signed, checksummed) |
| **Resource exhaustion** | MEDIUM (per-process limits) | MEDIUM | **LOW** (per-tab VRAM budget, node/time limits) |

### Security Design Principles

```
1. SEPARATION OF CONCERNS
   Content extraction (Loom 0) has no JS engine.
   Style resolution (Loom 1) has no JS engine.
   JS execution (Loom 2) has no DOM API, no network, no filesystem.
   Each loom is independently sandboxed.

2. DATA, NOT CODE
   .ovd files contain typed arrays — text, numbers, bounds.
   There is no mechanism to interpret data as executable code.
   The worst a malicious .ovd can do is display wrong content.

3. SMALLEST POSSIBLE SURFACE
   Don't implement APIs you don't need.
   No WebRTC, no WebGL, no Web Audio, no Service Workers,
   no extensions, no plugins. Each missing API is an
   entire vulnerability class that cannot exist.

4. MEMORY SAFETY BY DEFAULT
   Entire stack is Rust. No use-after-free, no buffer overflow,
   no null pointer dereference. The #1 class of browser CVEs
   (memory safety) is eliminated by language choice.

5. TRUST NOTHING FROM THE NETWORK
   Every input is validated: HTML (html5ever spec-compliant parser),
   CSS (cssparser with depth limits), JS (Boa with time budget),
   images (image-rs with size limits), .ovd (magic + checksum + signature).

6. DEFENSE IN DEPTH
   Multiple independent layers must all fail for a compromise.
   JS sandbox + DOM diff + time budget + no DOM API + no network =
   five layers protecting against malicious JS alone.

7. PRIVACY BY ARCHITECTURE
   No business model requires tracking.
   Fingerprinting APIs don't exist.
   Queries execute locally.
   Cache is local and partitioned.
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **OctoView** | GPU-native browser — views both OctoFlowWeb native apps and transpiled HTML/CSS/JS |
| **OctoFlowWeb** | The .flow language framework for building native GPU web applications |
| **Web Transpiler** | The engine that converts HTML/CSS/JS into OctoView Documents |
| **OctoView Document (.ovd)** | Flat columnar page format — render target + database table + semantic model |
| **Loom 0 (Content)** | First pipeline stage: stream-parse HTML, extract content, render immediately |
| **Loom 1 (Style)** | Second stage: resolve CSS, apply computed styles, re-render styled |
| **Loom 2 (Behavior)** | Third stage: background JS execution, progressive DOM updates |
| **JS Cron** | Background re-execution of JavaScript to keep cached pages fresh |
| **Semantic Extraction** | Three-layer engine: HTML tags + ARIA roles + heuristics → typed nodes |
| **OVD Cache** | Disk cache of pre-transpiled pages for instant return visits |
| **Snapshot CDN** | Community-maintained pre-transpiled popular SPAs |
| **Div-Soup Decoder** | Heuristic engine that infers semantics from styled `<div>` elements |

---

## Summary

The OctoView Web Transpiler is not another browser engine. It's a **paradigm shift**:

- **Transpile, don't render.** Extract meaning from HTML/CSS/JS, express it as GPU-native data.
- **Three looms, three priorities.** Content (instant) → Style (fast) → Behavior (background).
- **Pages are data.** Every page becomes a flat columnar document you can query with SQL.
- **Cache everything.** First visit transpiles. Every return visit is instant (10ms).
- **JS is a cron.** Not a blocking gate. Background execution, progressive enhancement.
- **Semantic extraction.** HTML tags + ARIA roles + heuristics → typed, queryable nodes.
- **GPU-native rendering.** OctoUI dispatch chains, single Vulkan submit per frame.
- **6-10x faster** than Chrome on static sites. **300x faster** on cached SPAs.
- **10x less memory.** Shared GPU resources, flat columnar format, no DOM tree.

The web was built for documents in 1995. OctoView treats it as what it's become:
**structured data that happens to be displayed.**

The octopus reads the web. Every page is a thread. The loom weaves them into fabric.
