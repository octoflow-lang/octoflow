# OctoView Browser — GPU-Native Dual-Mode Browser

**Date:** 2026-02-26
**Status:** Approved for implementation

## Vision

OctoView Browser is a GPU-native browser with two rendering modes:

1. **Web Mode** — Fetch HTML/CSS/JS pages, transpile to OVD, render on GPU
2. **Flow Mode** — Load `.flow` files, render natively via OctoUI on GPU

This makes OctoView the **developer's browser**: build web content in `.flow` instead of HTML/CSS/JS. The browser renders both the existing web AND native GPU applications through the same rendering pipeline.

Controlling the rendering pipeline gives interception advantages no external scraper can match. The OVD format is the universal IR — HTML pages produce OVD nodes, `.flow` files produce widget trees, both render through the same SPIR-V compute kernels.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 OctoView Browser                │
│                                                 │
│  ┌──────────────┐        ┌──────────────┐       │
│  │   Web Mode   │        │  Flow Mode   │       │
│  │ URL → fetch  │        │ .flow file   │       │
│  │ HTML → parse │        │ OctoUI eval  │       │
│  │ DOM → OVD    │        │ Widget tree  │       │
│  └──────┬───────┘        └──────┬───────┘       │
│         │                       │               │
│         ▼                       ▼               │
│  ┌──────────────────────────────────────┐       │
│  │         OVD Render Tree              │       │
│  │  (flat typed nodes → layout boxes)   │       │
│  └──────────────┬───────────────────────┘       │
│                 │                               │
│  ┌──────────────▼───────────────────────┐       │
│  │      GPU Layout Engine               │       │
│  │  Block flow, inline text, scroll     │       │
│  └──────────────┬───────────────────────┘       │
│                 │                               │
│  ┌──────────────▼───────────────────────┐       │
│  │   OctoUI Renderer (SPIR-V)           │       │
│  │  ui_clear.spv + ui_rect.spv +        │       │
│  │  ui_text.spv → Vulkan surface        │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  ┌──────────────────────────────────────┐       │
│  │   Browser Chrome (OctoUI widgets)    │       │
│  │  URL bar, tabs, back/fwd, status     │       │
│  └──────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

## Input Paths

### Web Mode (browse the existing web)

```
URL → reqwest/headless_chrome → raw HTML
  → html5ever parse → DOM tree
  → extract.rs three-layer transpiler → OVD nodes
  → layout engine → positioned boxes
  → OctoUI SPIR-V kernels → pixels on Vulkan surface
```

### Flow Mode (native .flow content)

```
.flow file → OctoUI widget tree (already GPU-native)
  → OctoUI layout → positioned boxes
  → OctoUI SPIR-V kernels → pixels on Vulkan surface
```

## Simplified Layout Engine (v1)

Map OVD node types to layout primitives instead of implementing full CSS:

| OVD NodeType | Layout Behavior |
|---|---|
| Page | Root container, vertical flow |
| Heading | Block, bold, scaled font (H1=32px, H2=24px, H3=20px, H4=18px, H5=16px, H6=14px) |
| Paragraph | Block, line-wrapped inline text |
| Link | Inline, colored (#0066CC), clickable hit region |
| Image | Block, aspect-ratio placeholder (v1: show alt text + dimensions) |
| List | Block, indented children |
| ListItem | Block with bullet/number prefix |
| Table | Grid layout (equal-width columns) |
| TableRow | Horizontal flow |
| TableCell | Fixed-width cell with padding |
| CodeBlock | Block, monospace font, background (#F5F5F5) |
| Blockquote | Block, left border, indented |
| Section/Card | Block, optional border |
| Separator | Horizontal line |
| Navigation | Block, horizontal link flow |
| Form/Input/Button | OctoUI native widgets (TextInput, Button) |

### Text Rendering

- fontdue for glyph rasterization (pure Rust, zero dependencies)
- GPU texture atlas: rasterize glyphs on CPU, upload to GPU texture
- ui_text.spv renders positioned quads sampling from the atlas
- Two fonts: proportional (system sans-serif) + monospace (for code)

### Line Wrapping

- CPU-side text measurement via fontdue glyph metrics
- Break text into lines at word boundaries
- Emit positioned quads for each glyph line
- GPU renders all quads in one dispatch

### Scrolling

- All content laid out in a vertical document
- Scroll = Y offset applied to all content positions
- GPU clips to viewport bounds
- Smooth scroll via frame interpolation

## Browser Chrome

Built entirely from OctoUI widgets:

```
┌─ toolbar ─────────────────────────────────────┐
│ [←] [→] [↻]  [________________url_bar______] │
├───────────────────────────────────────────────┤
│                                               │
│            content viewport                   │
│         (OVD → layout → GPU render)           │
│                                               │
│                                               │
├───────────────────────────────────────────────┤
│ status: 911 nodes │ Web Mode │ 234ms          │
└───────────────────────────────────────────────┘
```

- Toolbar: OctoUI Row + Button + TextInput
- Content viewport: Custom render area for layout engine output
- Status bar: OctoUI StatusBar (node count, mode, timing)
- Keyboard shortcuts: Ctrl+L (focus URL bar), F5 (reload), Alt+Left/Right (back/forward)

## Data Interception

Because we control the entire pipeline, we intercept at every stage:

| Stage | What We Capture | Advantage Over External Scrapers |
|---|---|---|
| Fetch | Raw HTML, headers, cookies | Same as Playwright |
| Parse | DOM tree before extraction | Not available externally |
| Extract | Typed OVD nodes (semantic) | Zero overhead — OVD IS the render input |
| Layout | Positioned boxes with coordinates | Layout-aware queries ("right 30% of page") |
| Render | GPU draw commands | Filter nodes before rendering (skip ads) |

## Developer Content Model

Developers write `.flow` instead of HTML/CSS/JS:

```flow
use ext.ui

fn main()
    let app = ui_app("My Dashboard")
    ui_heading(app, "Sales Report", 1)
    ui_text(app, "Q4 2026 results")

    let table = ui_table(app, ["Product", "Revenue", "Growth"])
    ui_table_row(table, ["OctoFlow", "$1.2M", "+45%"])
    ui_table_row(table, ["OctoView", "$800K", "+120%"])

    ui_button(app, "Export CSV", fn() export_data() end)
    ui_run(app)
end
```

No DOM, no CSS cascade, no JavaScript runtime. One file, GPU-native rendering.

## Project Structure

```
apps/octoview-browser/
├── Cargo.toml
├── src/
│   ├── main.rs         # Window creation (winit), Vulkan init (ash), event loop
│   ├── browser.rs      # Browser state: URL, history, navigation stack
│   ├── chrome.rs       # Browser chrome: toolbar, status bar (OctoUI widgets)
│   ├── layout.rs       # OVD nodes → positioned LayoutBox tree
│   ├── text.rs         # fontdue glyph atlas, text measurement, line wrapping
│   ├── render.rs       # Vulkan pipeline setup, draw command submission
│   ├── input.rs        # Mouse click → hit test → navigate, keyboard handling
│   └── viewport.rs     # Scroll state, content clipping, viewport management
├── shaders/
│   ├── rect.spv        # Rectangle/background fill (from OctoUI or new)
│   ├── text.spv        # Text quad rendering (from OctoUI or new)
│   └── clear.spv       # Screen clear (from OctoUI)
```

## Dependencies

```toml
winit = "0.30"              # Window + event loop (cross-platform)
ash = "0.38"                # Raw Vulkan bindings (already in flowgpu-vulkan)
fontdue = "0.9"             # Glyph rasterization (pure Rust, fast, zero deps)

# Internal crates (workspace members):
# flowgpu-spirv             # SPIR-V emitter (for custom shaders if needed)
# flowgpu-vulkan            # Vulkan device, memory, dispatch

# OctoView CLI (shared modules):
# html5ever, markup5ever    # HTML parsing
# reqwest                   # HTTP client
# headless_chrome           # JS rendering fallback
```

## v1 Scope

### In v1 (ship first)
- [x] Window with Vulkan rendering (winit + ash)
- [x] GPU text rendering (fontdue atlas → SPIR-V text shader)
- [x] Block layout engine (headings, paragraphs, links, lists, code, quotes)
- [x] Inline text flow (line wrapping, font sizes, bold/italic)
- [x] Link clicking (mouse hit test → navigate)
- [x] Scroll (mouse wheel + keyboard PageUp/PageDown/Home/End)
- [x] URL bar (type URL, press Enter to navigate)
- [x] Back/Forward navigation with history
- [x] HTML fetching → OVD → GPU render pipeline
- [x] Status bar (node count, mode, render time)
- [x] Visually distinct "OctoView look" — dark theme, clean typography

### v2 (later)
- CSS color/background support from OVD styles
- Image rendering (decode PNG/JPEG → GPU texture)
- Form inputs rendered as OctoUI widgets
- Tab bar (multiple pages)
- Bookmarks
- History panel
- OVD Inspector (developer tools)
- Flow Mode (.flow file rendering)
- Find in page (Ctrl+F)

## Success Criteria

v1 is done when:
1. Window opens with GPU-rendered content
2. Type URL → fetches HTML → transpiles to OVD → renders readable page
3. Click links → navigates to new page
4. Scroll works smoothly (mouse wheel + keyboard)
5. Back/Forward buttons work
6. Pages are readable: headings sized, links colored, code monospaced
7. Status bar shows node count + render time
8. Visually distinct from Chrome/Firefox — this is clearly a different browser

## Rendering Philosophy: Purity vs Convention

Two rendering tiers, unified by the Loom Engine:

### Flow Mode — Pure GPU (the reason OctoView exists)
`.flow` content runs directly in the Loom Engine. No parsing, no extraction, no
transpilation. The widget tree IS the render input. Near-zero CPU involvement.
This is the native path — maximum GPU utilization, minimum overhead.

### Web Mode — Convention + Loom Acceleration
HTML/CSS/JS follows the conventional browser pipeline (fetch → parse → layout →
render) but heavy stages are accelerated through Loom compute kernels. Text
shaping, layout box computation, glyph rasterization, rect fills — all dispatch
to SPIR-V where it makes sense. We respect the web's conventions but refuse its
CPU bottleneck.

### v1 → v2 Upgrade Path
v1 uses CPU framebuffer → GPU texture (scaffolding). v2 replaces CPU drawing
with Loom dispatches piece by piece: glyph atlas via compute first, then rect
fills, then layout computation itself. The API surface stays the same — only the
backend changes from CPU to Loom.

## Competitive Position

No existing browser or scraper offers:
- GPU-native rendering from semantic document model (not CSS)
- Zero-overhead data extraction (the render input IS the structured data)
- Layout-aware querying (spatial relationships, not just DOM)
- Native .flow content as alternative to HTML/CSS/JS
- Single SPIR-V pipeline for both web content and native apps
