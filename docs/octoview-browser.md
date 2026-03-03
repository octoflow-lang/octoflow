# OctoView Browser

GPU-native dual-mode browser for the OctoFlow platform. Renders web content through the OVD (OctoView Document) pipeline: HTML is fetched, parsed, styled, scripted, extracted to semantic nodes, laid out, and rendered to a Vulkan GPU surface.

## Status

**24 source files, ~11,000 lines of Rust, 163 tests.**

| Feature | Status |
|---------|--------|
| Vulkan rendering pipeline | Done |
| HTML parsing (html5ever) | Done |
| OVD semantic extraction | Done |
| CSS cascade (stylesheets + selectors + specificity + inheritance) | Done |
| CSS box model (margin, padding, width, border, display) | Done |
| JavaScript engine (Boa ES2024) | Done |
| Async page loading (non-blocking UI) | Done |
| Tabs, find-in-page, bookmarks, history | Done |
| OVD Inspector (F12) | Done |
| Flow Mode (.flow file rendering) | Done |
| Image rendering (PNG/JPEG) | Done |
| Loom Engine scaffolding (CPU) | Done |
| Events + fetch() API | Done |
| Table horizontal layout | Done |
| Container backgrounds + borders | Done |
| Text decoration (underline) | Done |
| Depth-based indentation | Done |
| Flexbox / CSS Grid layout | Phase 5 (planned) |
| Loom GPU migration | Deferred |

## Architecture

```
URL
 → fetch HTML (reqwest, background thread)
 → parse DOM (html5ever)
 → extract <style> + fetch <link> CSS (3 max, 2s timeout)
 → parse CSS → Stylesheet (selector chains pre-parsed)
 → match selectors → cascade → compute styles (specificity + inheritance)
 → extract OVD semantic tree from styled DOM
 → extract <script> tags → execute via Boa JS (64KB cap, 10 max)
 → apply DOM mutations to OVD
 → layout (text wrapping, block flow, images)
 → render to CPU framebuffer
 → upload to Vulkan swapchain → GPU present
```

Page loading runs on a background thread. The UI stays responsive during fetch, parse, and JS execution — only the final layout step runs on the main thread.

### Dual Mode

- **Web Mode**: Standard HTML/CSS/JS rendering via the pipeline above
- **Flow Mode**: `.flow` source files rendered as syntax-highlighted views (`flow://` protocol)

The Loom Engine sits between OVD generation and pixel output. It observes CSS→pixel mappings to learn GPU-native rendering patterns, eventually replacing CPU rendering with compute shaders.

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `main.rs` | 1,070 | App struct, event loop, navigation, keyboard/mouse handling |
| `extract.rs` | 927 | DOM→OVD extraction, CSS cascade, style computation, script extraction |
| `selector_match.rs` | 759 | CSS selector parsing, matching, specificity calculation |
| `layout.rs` | 733 | Block layout engine, text wrapping, image sizing |
| `chrome.rs` | 632 | Browser chrome (toolbar, URL bar, tab bar, status bar, find bar) |
| `pipeline.rs` | 640 | Vulkan render pipeline (shaders, descriptor sets, command buffers) |
| `js_engine.rs` | 564 | Boa JS integration, DOM mutation capture, browser API stubs |
| `vulkan.rs` | 549 | Vulkan context (instance, device, swapchain, synchronization) |
| `inspector.rs` | 508 | OVD Inspector panel (F12), node tree, property display |
| `page.rs` | 406 | Page loading pipeline (sync + async), welcome page, flow page |
| `stylesheet.rs` | 367 | CSS stylesheet parser (rules, declarations, comment stripping) |
| `css.rs` | 275 | Inline style parser, color parsing (hex, rgb, named) |
| `text.rs` | 261 | Font rendering (fontdue + Inter TTF), text measurement, wrapping |
| `framebuffer.rs` | 245 | CPU framebuffer (pixel buffer, rect/image drawing, blending) |
| `loom.rs` | 247 | Loom Engine scaffolding (CPU backend, GPU backend placeholder) |
| `side_panel.rs` | 238 | Side panel UI (bookmarks/history list, hit testing) |
| `find.rs` | 212 | Find-in-page (live search, match navigation, highlight) |
| `bookmarks.rs` | 170 | Bookmark store + browsing history (JSON persistence) |
| `tabs.rs` | 144 | Tab management (add, close, switch, per-tab history/scroll) |
| `renderer.rs` | 134 | Content area renderer (layout boxes → framebuffer pixels) |
| `ovd.rs` | 121 | OVD document model (node types, styles, semantic tree) |
| `image_loader.rs` | 117 | Image cache (URL→decoded pixels, deduplication) |
| `flow_mode.rs` | 88 | Flow Mode detection and .flow→OVD conversion |
| `fetch.rs` | 73 | HTTP fetch (HTML + CSS), URL normalization, timeouts |

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `winit` | 0.30 | Window creation and event loop |
| `ash` | 0.38 | Vulkan bindings (thin, unsafe) |
| `html5ever` | 0.38 | HTML5 spec-compliant parser |
| `markup5ever_rcdom` | 0.38 | DOM tree (reference-counted) |
| `reqwest` | 0.12 | HTTP client (blocking + native-tls) |
| `cssparser` | 0.36 | CSS tokenizer (from Servo) |
| `boa_engine` | 0.20 | JavaScript engine (pure Rust, ES2024) |
| `fontdue` | 0.9 | Font rasterization |
| `image` | 0.25 | PNG/JPEG decoding |
| `url` | 2 | URL parsing and resolution |
| `serde` / `serde_json` | 1 | Bookmark/history JSON serialization |

## Build and Run

```bash
# Build (requires MSVC + Vulkan SDK on Windows)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" build -p octoview-browser

# Run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" run --bin octoview-browser

# Run with URL argument
powershell.exe ... run --bin octoview-browser -- "https://example.com"

# Test
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" test -p octoview-browser
```

Requires: Rust toolchain, Vulkan-capable GPU + driver, Vulkan SDK, MSVC (Windows).

The main thread spawns with 8MB stack (Boa JS engine requires more than the 1MB Windows default). Uses winit's `any_thread(true)` for compatibility.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Ctrl+L | Focus URL bar |
| Enter | Navigate to URL (when URL bar focused) |
| Escape | Unfocus URL bar / close find bar / close side panel |
| F5 | Reload page |
| F12 | Toggle OVD Inspector |
| Alt+Left | Back |
| Alt+Right | Forward |
| Ctrl+F | Toggle find-in-page |
| Ctrl+T | New tab |
| Ctrl+W | Close tab |
| Ctrl+Tab | Next tab |
| Ctrl+Shift+Tab | Previous tab |
| Ctrl+D | Bookmark current page |
| Ctrl+B | Toggle bookmarks panel |
| Ctrl+H | Toggle history panel |
| Space / PageDown | Scroll down |
| PageUp | Scroll up |
| Home / End | Top / bottom of page |

## CSS Engine

The CSS engine processes stylesheets through a three-stage pipeline:

1. **Parse**: CSS text → `Vec<CssRule>` with pre-parsed `SelectorChain` per rule
2. **Match**: For each DOM element, test all rules against element + ancestors
3. **Cascade**: Sort matches by specificity (tag < class < id < inline), merge declarations, apply inheritance

**Selector support:** tag, `.class`, `#id`, compound (`div.foo#bar`), descendant (` `), child (`>`), comma groups (`h1, h2, h3`).

**Inherited properties:** `color`, `font-size`, `font-weight`, `text-align`.
**Non-inherited:** `background-color`, `display`, `margin`, `padding`, `width`, `max-width`, `border`, `text-decoration`.

**Box model properties:**
- `margin` / `margin-top` / `margin-bottom` / `margin-left` / `margin-right` — shorthand + individual
- `padding` / `padding-top` / `padding-bottom` / `padding-left` / `padding-right` — shorthand + individual
- `width`, `max-width` — explicit width constraints
- `border` / `border-width` / `border-color` — rendered as rectangles
- `text-decoration: underline` — link underlines
- `display: block | inline | inline-block | none | table-row | table-cell`
- `line-height` — text line spacing
- Length units: `px`, `pt`, `em`, `rem`, bare numbers

**Performance guards:**
- Selectors pre-parsed at stylesheet load (not per-element)
- DOM depth capped at 64
- Ancestor chain capped at 32 for selector matching
- External CSS fetches capped at 3 with 2s timeout

## JavaScript Engine

Boa ES2024 engine with DOM mutation capture pattern:

1. JS calls `document.getElementById("foo").setInnerHTML("bar")`
2. Native function captures this as `DomMutation::SetInnerHtml { id, html }`
3. After all scripts execute, mutations are applied to the OVD document

**Registered globals:**
- `console.log()`, `console.warn()`
- `document.getElementById()`, `document.querySelector()`, `document.write()`
- `alert()`
- `window` (stub: `location`, `navigator`, `addEventListener`, `setTimeout`, `getComputedStyle`)

**Element proxy methods:** `setInnerHTML()`, `setTextContent()`, `addClass()`, `removeClass()`, `setStyle()`, `hide()`, `show()`

**Safety limits:**
- Scripts > 64KB skipped (YouTube bundles are 500KB+)
- Maximum 10 scripts per page
- Thread-local mutation storage (safe, single-threaded)

## OVD Document Model

The OVD (OctoView Document) is a flat semantic tree:

```rust
enum NodeType {
    Page, Heading, Paragraph, Link, ListItem,
    Image, CodeBlock, Separator, BlockQuote, Table
}

struct OvdNode {
    node_type: NodeType,
    text: String,
    href: String,       // for links
    level: u32,         // for headings (1-6)
    style: OvdStyle,    // computed CSS styles
}

struct OvdStyle {
    color: Option<(u8, u8, u8)>,
    background_color: Option<(u8, u8, u8)>,
    font_size: Option<f32>,
    bold: bool,
    text_align: Option<String>,
}
```

## Rendering Pipeline

```
OVD nodes → layout engine → positioned LayoutBoxes
  → CPU framebuffer (pixel buffer, u8 RGBA)
  → Vulkan staging buffer upload
  → full-screen quad render (vertex + fragment shader)
  → swapchain present
```

The Vulkan pipeline renders a single full-screen textured quad. All content compositing happens on the CPU framebuffer — the GPU just displays the final image. This will be replaced by Loom Engine compute shaders as the engine matures.

## Commit History

```
4f7b573 feat(browser): async page loading — UI stays responsive during fetch
3ec63d9 fix(browser): window stub + script size cap to prevent YouTube freeze
3904a3b fix(browser): 8MB stack for Boa JS engine via any_thread event loop
c084965 feat(browser): Boa JS engine integration with DOM mutation capture
0e9e914 fix(browser): prevent crash on large sites + speed up CSS cascade
aab9a4f feat(browser): CSS cascade - extract stylesheets, match selectors, apply styles
19e9be3 feat(browser): CSS stylesheet parser + selector matching engine
b780c33 OctoView CSS Engine implementation plan: 8 tasks
59615cb OctoView Browser Engine design: CSS + JS + DOM in 4 phases
51a53b8 fix(browser): URL bar select-all on focus
6ffc5ae fix(browser): correct Inter.ttf include_bytes path
72b3b32 feat(browser): bookmarks, side panel, and Loom Engine scaffolding
e11f84b feat(browser): OVD Inspector with node tree, properties, and selection
a04d4b4 feat(browser): Flow Mode detection and .flow file rendering
d787da0 feat(browser): find-in-page with live search and match navigation
```

## Known Limitations

- **No flexbox/grid layout** — box model works, but no flex/grid positioning (Phase 5)
- **No float/position** — no `float: left/right`, no `position: absolute/fixed`
- **No form submission** — input fields and forms are not interactive
- **No cookies** — session state not maintained across navigations
- **YouTube/Gmail/etc.** — JS-rendered SPAs show minimal content (massive bundles exceed 64KB cap)
- **No web fonts** — uses bundled Inter TTF for all text
- **Debug build only** — no release optimization yet

## Next Steps (Phase 5: Flexbox + Advanced Layout)

1. `display: flex` with `flex-direction`, `justify-content`, `align-items`
2. `float: left/right` with clearing
3. `position: absolute/fixed/relative` with `top/left/right/bottom`
4. `overflow: hidden/scroll`
5. Cookie jar (reqwest cookie store)
6. Form submission (GET/POST)
