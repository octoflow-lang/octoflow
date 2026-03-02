# OctoView Browser v2 — Design Document

**Date:** 2026-02-26
**Status:** Approved for implementation
**Prerequisite:** v1 complete (3,800 lines, 12 tests, merged to master)

## Goal

Upgrade OctoView Browser from basic HTML text rendering to a feature-complete browser with CSS colors, inline images, tabs, find-in-page, Flow Mode, developer tools, bookmarks/history, and GPU-native Loom Engine rendering.

## Approach: Incremental Enhancement

All features build on the existing CPU framebuffer architecture. Each feature is independently testable. Loom Engine migration is the final step — it replaces the rendering backend without changing the API surface.

```
v1 (done): HTML → OVD → CPU layout → CPU draw → GPU texture → present
v2 (this): Add features 1-7 on CPU scaffolding, then swap backend to Loom
```

---

## Feature 1: CSS Color/Background Extraction

**Problem:** extract.rs throws away all styling. Pages render in default colors only.

**Solution:** Parse `style=""` attributes on HTML elements. Extract key-value pairs for color, background, font properties.

### OvdNode Style Extension

```rust
pub struct OvdStyle {
    pub color: Option<[u8; 3]>,
    pub background: Option<[u8; 3]>,
    pub font_weight: Option<u16>,      // 400=normal, 700=bold
    pub font_size_px: Option<f32>,
    pub text_align: Option<TextAlign>, // Left/Center/Right
    pub border_color: Option<[u8; 3]>,
    pub display_none: bool,
}
```

### Extraction

Simple inline style parser: split `style` attribute by `;`, split each by `:`, match known properties. Also handle legacy HTML attributes (`bgcolor`, `color`).

Supported properties: `color`, `background-color`, `background`, `font-size`, `font-weight`, `text-align`, `display`, `border-color`.

Color parsing: `#RGB`, `#RRGGBB`, `rgb(r,g,b)`, named colors (16 HTML basic colors).

### Layout Integration

`layout.rs` checks `node.style` and overrides LayoutStyle defaults. CSS color → text_color, background-color → bg_color, font-weight:bold → FontStyle::Bold.

### Not in Scope

External stylesheets, `<style>` blocks, CSS selectors, cascade, specificity, computed styles. Those are v3 (full CSS parser).

---

## Feature 2: Image Rendering

**Problem:** v1 shows gray placeholder boxes with alt text.

**Solution:** Fetch, decode, resize, and blit images inline.

### Pipeline

```
<img src="url"> → resolve URL → fetch bytes → decode PNG/JPEG
    → resize to fit viewport width → RGBA pixels
    → blit onto CPU framebuffer at layout position
```

### Implementation

- **Decode:** `image` crate (already in workspace). Decode to RGBA8.
- **Fetch:** During page load, after HTML extraction, walk OVD nodes for Image types. Fetch each `src` URL. Cap: 20 images per page, 5MB per image.
- **Resize:** Scale down to fit `content_width`, preserve aspect ratio. No upscale.
- **Blit:** New `Framebuffer::draw_image(x, y, w, h, rgba_pixels)` — copies decoded pixels with bounds clipping.
- **Layout:** `LayoutContent::Image` holds decoded pixels + dimensions. Layout computes box height from actual image size instead of 120px placeholder.
- **Cache:** `HashMap<String, DecodedImage>` keyed by URL. Cleared on navigation.

### Not in Scope

SVG, animated GIF, lazy loading, srcset, responsive images.

---

## Feature 3: Tabs

**Problem:** Single-page browsing only.

### UI Layout

```
┌─ toolbar ──────────────────────────────────────┐
│ [←] [→] [↻]  [____________url_bar____________] │
├─ tab bar ──────────────────────────────────────┤
│ [Tab 1 ✕] [Tab 2 ✕] [+]                        │
├────────────────────────────────────────────────┤
│              content viewport                   │
├────────────────────────────────────────────────┤
│ status bar                                      │
└────────────────────────────────────────────────┘
```

### Data Model

```rust
struct Tab {
    url: String,
    title: String,
    page: Option<PageState>,
    scroll_y: f32,
    history: Vec<String>,
    history_idx: usize,
}

// App holds:
tabs: Vec<Tab>,
active_tab: usize,
```

Each tab owns its own page state, scroll position, and navigation history. URL bar, back/forward, and status bar reflect the active tab. Switching tabs swaps displayed state — no re-fetching.

### Chrome

- Tab bar: 24px height, between toolbar and content
- Tab button: truncated title (max 120px) + close button (✕)
- `[+]` button opens new tab with welcome page
- Active tab: primary color. Inactive: dimmed.

### Keyboard

- `Ctrl+T` new tab
- `Ctrl+W` close tab
- `Ctrl+Tab` / `Ctrl+Shift+Tab` switch tabs
- Middle-click link opens in new tab

### Not in Scope

Tab dragging/reordering, overflow scrolling, pinned tabs. Max ~10 tabs.

---

## Feature 4: Find in Page (Ctrl+F)

### UI

```
┌─ toolbar ──────────────────────────────────────┐
├─ find bar ─────────────────────────────────────┤
│ Find: [________query______] [↑] [↓] 3/17  [✕] │
├─ tab bar ──────────────────────────────────────┤
```

Find bar appears below toolbar when Ctrl+F pressed.

### Implementation

- **Search:** Walk layout boxes, case-insensitive substring match on text content. Collect matches as `(box_index, char_offset, length)`.
- **Highlighting:** During render, matched substrings get yellow background (`[255, 220, 50]`). Active match gets orange (`[255, 140, 0]`).
- **Navigation:** Enter/↓ = next match, Shift+Enter/↑ = previous. Auto-scroll to center active match.
- **Live search:** Matches update as you type.

### State

```rust
struct FindState {
    query: String,
    cursor: usize,
    matches: Vec<FindMatch>,
    active_match: usize,
    visible: bool,
}

struct FindMatch {
    box_idx: usize,
    char_start: usize,
    char_len: usize,
    y_position: f32,
}
```

### Keyboard

`Ctrl+F` opens/focuses, `Escape` closes.

---

## Feature 5: Flow Mode (.flow File Rendering)

**The core vision:** `.flow` files render natively through OctoUI instead of the HTML→OVD pipeline.

### Detection

```rust
fn is_flow_url(url: &str) -> bool {
    url.ends_with(".flow") || url.starts_with("flow://")
}
```

### Pipeline

```
.flow file → flowgpu-cli --render-to-buffer
    → OctoUI widget tree evaluation
    → OctoUI pipeline renders one frame
    → Pixel buffer to stdout/temp file
    → Browser reads pixels, blits to framebuffer
```

### Bridge

Shell out to `flowgpu-cli` with a `--render-to-buffer` flag. The subprocess evaluates the .flow file, runs one OctoUI frame, writes rendered pixels. Browser displays them.

This is the simplest bridge — no embedding the FlowGPU VM.

### Limitations (v2)

- Static rendering only (one frame snapshot)
- No live widget interactivity
- Re-renders on F5/reload
- Interactive Flow Mode (live widget events) is v3

### Status Bar

Shows "Flow Mode" instead of "Web Mode" when rendering .flow content.

---

## Feature 6: OVD Inspector (Developer Tools)

### UI

```
│    content viewport (70%)    │   OVD Inspector (30%)   │
│                              │  ┌─ Node Tree ────────┐ │
│                              │  │ ▸ Page              │ │
│                              │  │   ▸ Heading "…"     │ │
│                              │  │   ▸ Paragraph "…"   │ │
│                              │  │   ▸ Link "…" →href  │ │
│                              │  └────────────────────┘ │
│                              │  ┌─ Properties ───────┐ │
│                              │  │ type: Heading       │ │
│                              │  │ level: 1            │ │
│                              │  │ text: "OctoView…"   │ │
│                              │  │ layout: 16,24 1248… │ │
│                              │  └────────────────────┘ │
```

### Implementation

- **Toggle:** `F12` opens/closes. Content re-layouts at new width.
- **Node tree:** Flat OVD node list, indented by depth. Truncated text per line. Independent scroll.
- **Selection:** Click node → blue outline on its LayoutBox in viewport. Properties panel shows full details.
- **Properties:** All OvdNode fields + LayoutBox dimensions + computed style.

### State

```rust
struct InspectorState {
    visible: bool,
    scroll_y: f32,
    selected_node: Option<usize>,
    panel_width_pct: f32,  // 0.30
}
```

### Not in Scope

Editable properties, CSS panel, network tab, console.

---

## Feature 7: Bookmarks & History

### Storage

```
~/.octoview/
├── bookmarks.json
└── history.json
```

### Bookmarks

- `Ctrl+D` bookmarks current page
- `Ctrl+B` toggles bookmarks sidebar (left, 25%)
- Click to navigate, Delete to remove
- Persists across sessions

### History

- Every navigation appends entry (URL + title + timestamp)
- `Ctrl+H` toggles history sidebar (left, 25%)
- Reverse chronological, grouped by date
- Cap: 1000 entries, oldest trimmed

### Data

```rust
struct BookmarkEntry {
    url: String,
    title: String,
    added_at: u64,
}

struct HistoryEntry {
    url: String,
    title: String,
    visited_at: u64,
}
```

### Panel Rules

- Only one left panel at a time (bookmarks OR history)
- Inspector is right panel, independent
- Both can be open simultaneously (25% left + 30% right + 45% content)

---

## Feature 8: Loom Engine Migration

**Final step.** Replace CPU framebuffer drawing with Loom GPU compute dispatches.

### What Migrates

| CPU (current) | Loom (target) |
|---|---|
| `fb.clear(r,g,b)` | `vm_dispatch(ui_clear.spv)` |
| `fb.draw_rect(...)` | `vm_dispatch(ui_rect.spv)` |
| `text.draw_text(...)` | `gdi_text_add() + vm_dispatch(ui_text_blit.spv)` |
| `fb.draw_image(...)` | `vm_dispatch(ui_image_blit.spv)` — new kernel |
| Staging buffer upload | Eliminated — GPU buffer is the framebuffer |

### Architecture Shift

```
v1: layout → CPU draw → RGBA buffer → staging → GPU texture → present
v2: layout → Loom dispatch queue → GPU compute → GPU buffer → present
```

### Migration Order

1. Integrate `flowgpu-vulkan` VM as dependency
2. Create `LoomRenderer` wrapping `VulkanCompute`
3. Replace clear + rects with `ui_clear.spv` + `ui_rect.spv`
4. Replace text with GDI atlas + `ui_text_blit.spv`
5. Add `ui_image_blit.spv` for image blitting
6. Remove `Framebuffer` struct and staging buffer
7. Present GPU buffer directly to swapchain

### New Kernel

`ui_image_blit.spv` — push constants: `dest_x, dest_y, src_w, src_h, screen_w, total_pixels, heap_offset`. Copies RGBA from heap to planar RGB buffer.

### Dependencies

All other features must work first — this replaces the backend they all use.

### Not in Scope

Double-buffered async Loom dispatch (v3 performance optimization).

---

## Implementation Order

```
1. CSS colors/backgrounds     (extract.rs + layout.rs changes)
2. Image rendering            (image decode + framebuffer blit)
3. Tabs                       (data model + chrome + input)
4. Find in page               (search + highlight + UI bar)
5. Flow Mode                  (subprocess bridge + detection)
6. OVD Inspector              (right panel + node tree)
7. Bookmarks & History        (persistence + left panels)
8. Loom Engine migration      (GPU compute replaces CPU draw)
```

Each feature is independently testable. Features 1-2 improve page rendering. Features 3-4 improve usability. Features 5-7 add new capabilities. Feature 8 is the architectural upgrade.

## Success Criteria

v2 is done when:
1. Pages show inline colors and backgrounds from HTML style attributes
2. Images decode and render inline (PNG/JPEG)
3. Multiple tabs with independent navigation
4. Ctrl+F finds and highlights text with match navigation
5. `.flow` files render via OctoUI subprocess
6. F12 opens OVD inspector showing node tree + properties
7. Bookmarks and history persist across sessions
8. All rendering uses Loom GPU compute — no CPU framebuffer
