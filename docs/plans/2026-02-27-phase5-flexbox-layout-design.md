# Phase 5: Flexbox + Advanced Layout + Cookies + Forms

**Date:** 2026-02-27
**Status:** Approved
**Baseline:** 163 tests, ~11,000 lines, 24 files

## Scope

1. `display: flex` with flex-direction, justify-content, align-items, flex-wrap, gap
2. `float: left/right` with clear
3. `position: relative/absolute/fixed` with top/left/right/bottom
4. `overflow: hidden/scroll`
5. Cookie jar (session persistence across navigations)
6. Form submission (GET/POST with input serialization)

## Design

### 1. Flexbox

**OvdStyle additions:**
- `flex_direction: Option<FlexDirection>` — Row (default) | Column
- `justify_content: Option<JustifyContent>` — FlexStart | FlexEnd | Center | SpaceBetween | SpaceAround
- `align_items: Option<AlignItems>` — FlexStart | FlexEnd | Center | Stretch
- `flex_wrap: Option<FlexWrap>` — NoWrap | Wrap
- `flex_grow: Option<f32>`, `flex_shrink: Option<f32>`, `flex_basis: Option<f32>`
- `gap: Option<f32>`

**Layout algorithm** (`layout_flex_container`):
1. Collect child nodes (same parent_id, consecutive in node list)
2. Measure each child's intrinsic size (text wrap width, image size, or explicit width)
3. Distribute space along main axis per justify_content
4. Align items on cross axis per align_items
5. If flex_wrap == Wrap and items overflow, start new flex line

### 2. Float

**OvdStyle additions:**
- `float: Option<Float>` — Left | Right
- `clear: Option<Clear>` — Left | Right | Both

**Layout approach:**
- `FloatContext` tracks left/right float stacks with (x, y, width, bottom_y)
- Block content starts at x offset past active floats
- Text wraps around float boundaries
- `clear` advances cursor_y past specified floats

### 3. Position

**OvdStyle additions:**
- `position: Option<Position>` — Static | Relative | Absolute | Fixed
- `top: Option<f32>`, `left: Option<f32>`, `right: Option<f32>`, `bottom: Option<f32>`

**Layout approach:**
- **Relative:** Normal flow, then offset by top/left (visual only)
- **Absolute:** Remove from flow, position relative to nearest positioned ancestor (or viewport)
- **Fixed:** Position relative to viewport, not affected by scroll

### 4. Overflow

**OvdStyle addition:**
- `overflow: Option<Overflow>` — Visible | Hidden | Scroll | Auto

**Rendering approach:**
- **Hidden:** Clip children to parent bounds during render (skip pixels outside)
- **Scroll/Auto:** Same as hidden for now (nested scroll is complex)

### 5. Cookie Jar

**Approach:** Use reqwest's built-in `cookie::Jar` (Arc-shared).

**Changes:**
- `fetch.rs`: New `create_client_with_cookies(jar: Arc<Jar>) -> Client`
- `main.rs`: `App` holds `Arc<Jar>`, passes to all fetch calls
- Cookies persist within session, not across restarts

### 6. Form Submission

**OVD additions:**
```rust
pub struct FormInfo {
    pub action: String,    // URL to submit to
    pub method: String,    // "get" or "post"
}

pub struct InputInfo {
    pub name: String,      // input name attribute
    pub value: String,     // current value
    pub input_type: String, // text, password, hidden, submit, checkbox, radio
}
```

**Flow:**
1. `extract.rs`: Parse `<form>`, `<input>`, `<select>`, `<textarea>` attributes
2. `ovd.rs`: Store FormInfo on Form nodes, InputInfo on InputField nodes
3. `main.rs`: On Button click, find enclosing Form, gather InputField values, submit
4. `fetch.rs`: `submit_form(url, method, params, jar)` — GET query string or POST body

## File Changes

| File | New Enums/Structs | New Functions | Properties |
|------|-------------------|---------------|------------|
| `ovd.rs` | FlexDirection, JustifyContent, AlignItems, FlexWrap, Float, Clear, Position, Overflow, FormInfo, InputInfo | — | ~16 new fields |
| `css.rs` | — | parse_flex_direction, parse_justify_content, parse_align_items, parse_position, parse_overflow | ~16 new property matches |
| `extract.rs` | — | — | Cascade for all new properties; form/input attribute extraction |
| `layout.rs` | FloatContext | layout_flex_container, layout_with_floats | Flex + float + position + overflow integration |
| `renderer.rs` | — | — | Overflow clipping, fixed-position rendering |
| `fetch.rs` | — | create_client_with_cookies, submit_form | Cookie jar parameter |
| `main.rs` | — | handle_form_submit | Cookie jar lifecycle, form click detection |

## Test Plan

- Flex row: 3 children laid out horizontally with correct widths
- Flex column: 3 children laid out vertically
- Flex justify-content: space-between distributes space
- Flex align-items: center vertically centers children
- Flex wrap: items wrap to next line when overflow
- Float left: content flows around floated element
- Float clear: cursor advances past floats
- Position relative: visual offset from normal position
- Position absolute: removed from flow, positioned at coordinates
- Overflow hidden: children clipped to parent bounds
- Cookie jar: cookies from Set-Cookie persist across navigations
- Form GET: inputs serialized to query string
- Form POST: inputs serialized to request body
