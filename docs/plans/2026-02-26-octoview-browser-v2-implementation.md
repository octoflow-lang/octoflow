# OctoView Browser v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade OctoView Browser from basic HTML text rendering to a feature-complete browser with CSS colors, inline images, tabs, find-in-page, Flow Mode, OVD Inspector, bookmarks/history, and Loom Engine migration.

**Architecture:** Incremental enhancement on v1's CPU framebuffer. Each feature adds to the existing pipeline (fetch → extract → layout → render → GPU). Loom migration replaces the render backend last. All features are independently testable.

**Tech Stack:** Rust, winit 0.30, ash 0.38, fontdue 0.9, html5ever 0.38, reqwest 0.12, image 0.25, serde/serde_json (bookmarks), flowgpu-vulkan (Loom migration)

**Design Doc:** `docs/plans/2026-02-26-octoview-browser-v2-design.md`

**v1 Baseline:** 12 source files, 3,800 lines, 12 passing tests at `apps/octoview-browser/src/`

**Build command (from worktree):**
```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command '& {
  $vsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2>$null
  if ($vsPath) { $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"; if (Test-Path $vcvars) { cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process") } } } }
  $env:VULKAN_SDK = "C:\VulkanSDK\1.4.304.1"; $env:PATH = "$env:VULKAN_SDK\Bin;$env:PATH"
  Set-Location "C:\OctoFlow\.worktrees\octoview-browser-v2"
  cargo build -p octoview-browser 2>&1 | Out-String -Stream
}'
```

Replace `build` with `test` for tests.

---

## Task 1: CSS Inline Style Parser + OvdStyle

**Files:**
- Create: `apps/octoview-browser/src/css.rs`
- Modify: `apps/octoview-browser/src/ovd.rs`
- Modify: `apps/octoview-browser/src/main.rs` (add `mod css;`)

**Step 1: Add OvdStyle to ovd.rs**

Add after the `OvdNode` struct:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Default)]
pub struct OvdStyle {
    pub color: Option<[u8; 3]>,
    pub background: Option<[u8; 3]>,
    pub font_weight: Option<u16>,
    pub font_size_px: Option<f32>,
    pub text_align: Option<TextAlign>,
    pub border_color: Option<[u8; 3]>,
    pub display_none: bool,
}
```

Add `pub style: OvdStyle` field to `OvdNode`, and `style: OvdStyle::default()` in `OvdNode::new()`.

**Step 2: Create css.rs with color/style parsing**

Create `apps/octoview-browser/src/css.rs`:

```rust
use crate::ovd::OvdStyle;

/// Parse an inline style attribute string into an OvdStyle.
/// e.g. "color: red; background-color: #f0f0f0; font-size: 18px"
pub fn parse_inline_style(style_attr: &str) -> OvdStyle {
    let mut result = OvdStyle::default();
    for decl in style_attr.split(';') {
        let decl = decl.trim();
        if let Some((prop, val)) = decl.split_once(':') {
            let prop = prop.trim().to_lowercase();
            let val = val.trim();
            match prop.as_str() {
                "color" => result.color = parse_color(val),
                "background-color" | "background" => result.background = parse_color(val),
                "font-size" => result.font_size_px = parse_font_size(val),
                "font-weight" => result.font_weight = parse_font_weight(val),
                "text-align" => result.text_align = parse_text_align(val),
                "border-color" => result.border_color = parse_color(val),
                "display" => {
                    if val.eq_ignore_ascii_case("none") {
                        result.display_none = true;
                    }
                }
                _ => {}
            }
        }
    }
    result
}

/// Parse CSS color values: #RGB, #RRGGBB, rgb(r,g,b), named colors.
pub fn parse_color(val: &str) -> Option<[u8; 3]> {
    let val = val.trim().to_lowercase();
    // Named colors (HTML basic 16)
    match val.as_str() {
        "black" => return Some([0, 0, 0]),
        "white" => return Some([255, 255, 255]),
        "red" => return Some([255, 0, 0]),
        "green" | "lime" => return Some([0, 128, 0]),
        "blue" => return Some([0, 0, 255]),
        "yellow" => return Some([255, 255, 0]),
        "cyan" | "aqua" => return Some([0, 255, 255]),
        "magenta" | "fuchsia" => return Some([255, 0, 255]),
        "gray" | "grey" => return Some([128, 128, 128]),
        "silver" => return Some([192, 192, 192]),
        "maroon" => return Some([128, 0, 0]),
        "olive" => return Some([128, 128, 0]),
        "navy" => return Some([0, 0, 128]),
        "purple" => return Some([128, 0, 128]),
        "teal" => return Some([0, 128, 128]),
        "orange" => return Some([255, 165, 0]),
        _ => {}
    }
    // #RRGGBB or #RGB
    if val.starts_with('#') {
        let hex = &val[1..];
        if hex.len() == 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            return Some([r, g, b]);
        } else if hex.len() == 3 {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            return Some([r, g, b]);
        }
    }
    // rgb(r, g, b)
    if val.starts_with("rgb(") && val.ends_with(')') {
        let inner = &val[4..val.len() - 1];
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() == 3 {
            let r = parts[0].trim().parse::<u8>().ok()?;
            let g = parts[1].trim().parse::<u8>().ok()?;
            let b = parts[2].trim().parse::<u8>().ok()?;
            return Some([r, g, b]);
        }
    }
    None
}

fn parse_font_size(val: &str) -> Option<f32> {
    let val = val.trim().to_lowercase();
    if val.ends_with("px") {
        val[..val.len() - 2].trim().parse().ok()
    } else if val.ends_with("pt") {
        val[..val.len() - 2].trim().parse::<f32>().ok().map(|pt| pt * 1.333)
    } else {
        val.parse().ok()
    }
}

fn parse_font_weight(val: &str) -> Option<u16> {
    match val.trim().to_lowercase().as_str() {
        "bold" => Some(700),
        "normal" => Some(400),
        "lighter" => Some(300),
        _ => val.trim().parse().ok(),
    }
}

fn parse_text_align(val: &str) -> Option<crate::ovd::TextAlign> {
    match val.trim().to_lowercase().as_str() {
        "left" => Some(crate::ovd::TextAlign::Left),
        "center" => Some(crate::ovd::TextAlign::Center),
        "right" => Some(crate::ovd::TextAlign::Right),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex_colors() {
        assert_eq!(parse_color("#ff0000"), Some([255, 0, 0]));
        assert_eq!(parse_color("#00ff00"), Some([0, 255, 0]));
        assert_eq!(parse_color("#f00"), Some([255, 0, 0]));
        assert_eq!(parse_color("#abc"), Some([170, 187, 204]));
    }

    #[test]
    fn test_parse_named_colors() {
        assert_eq!(parse_color("red"), Some([255, 0, 0]));
        assert_eq!(parse_color("Blue"), Some([0, 0, 255]));
        assert_eq!(parse_color("WHITE"), Some([255, 255, 255]));
    }

    #[test]
    fn test_parse_rgb() {
        assert_eq!(parse_color("rgb(128, 64, 32)"), Some([128, 64, 32]));
    }

    #[test]
    fn test_parse_inline_style() {
        let s = parse_inline_style("color: red; background-color: #f0f0f0; font-size: 18px");
        assert_eq!(s.color, Some([255, 0, 0]));
        assert_eq!(s.background, Some([240, 240, 240]));
        assert_eq!(s.font_size_px, Some(18.0));
    }

    #[test]
    fn test_display_none() {
        let s = parse_inline_style("display: none");
        assert!(s.display_none);
    }

    #[test]
    fn test_font_weight() {
        let s = parse_inline_style("font-weight: bold");
        assert_eq!(s.font_weight, Some(700));
        let s2 = parse_inline_style("font-weight: 600");
        assert_eq!(s2.font_weight, Some(600));
    }
}
```

**Step 3: Add `mod css;` to main.rs**

Add `mod css;` to the module declarations at the top of main.rs.

**Step 4: Build and test**

Run build, then `cargo test -p octoview-browser`. Expect 18 tests (12 existing + 6 new CSS tests).

**Step 5: Commit**

```
feat(browser): CSS inline style parser with color/font/align support
```

---

## Task 2: Integrate CSS Styles into Extraction and Layout

**Files:**
- Modify: `apps/octoview-browser/src/extract.rs`
- Modify: `apps/octoview-browser/src/layout.rs`

**Step 1: Extract styles in extract.rs**

In `walk_node()`, when processing any element with a `style` attribute, parse it and attach to the OvdNode. Find the `attrs` borrow in each match arm. After creating the `OvdNode`, before `doc.add_node(node)`:

At the top of `walk_node()`, inside `NodeData::Element { ref name, ref attrs, .. }`, extract the style attribute into the node's style field. This applies to all tag match arms. Add a helper:

```rust
fn extract_style(attrs: &[html5ever::tendril::Tendril<html5ever::tendril::fmt::UTF8>]) -> crate::ovd::OvdStyle {
    // Actually attrs is Vec<Attribute>, need to find "style" attr
}
```

More practically: in each match arm that creates an `OvdNode`, after `let attrs = attrs.borrow();`, add:

```rust
let style_attr = attrs.iter()
    .find(|a| a.name.local.as_ref() == "style")
    .map(|a| a.value.to_string());
// ... after creating node:
if let Some(ref style_str) = style_attr {
    node.style = crate::css::parse_inline_style(style_str);
}
```

Also extract legacy `bgcolor` and `color` HTML attributes:

```rust
let bgcolor = attrs.iter()
    .find(|a| a.name.local.as_ref() == "bgcolor")
    .and_then(|a| crate::css::parse_color(&a.value));
let fgcolor = attrs.iter()
    .find(|a| a.name.local.as_ref() == "color")
    .and_then(|a| crate::css::parse_color(&a.value));
if bgcolor.is_some() && node.style.background.is_none() {
    node.style.background = bgcolor;
}
if fgcolor.is_some() && node.style.color.is_none() {
    node.style.color = fgcolor;
}
```

To avoid duplicating this in every arm, create a helper function `apply_style_attrs()` that takes the attrs ref and the node, and call it in each arm before `doc.add_node()`.

Also: skip nodes where `node.style.display_none` is true (don't add them to the document).

**Step 2: Apply styles in layout.rs**

In `layout_document()`, after computing the default LayoutStyle for each node type, override with OvdStyle values:

```rust
// After setting up default style for the node type:
if let Some(color) = node.style.color {
    style.text_color = color;
}
if let Some(bg) = node.style.background {
    style.bg_color = Some(bg);
}
if let Some(size) = node.style.font_size_px {
    style.font_size = size;
}
if let Some(weight) = node.style.font_weight {
    if weight >= 700 {
        style.font_style = FontStyle::Bold;
    }
}
```

**Step 3: Build and test**

All 18 tests should pass. Verify manually that pages with inline styles now show colors.

**Step 4: Commit**

```
feat(browser): extract inline CSS styles and apply to layout
```

---

## Task 3: Image Fetching and Decoding

**Files:**
- Modify: `apps/octoview-browser/Cargo.toml` (add `image` dependency)
- Create: `apps/octoview-browser/src/image_loader.rs`
- Modify: `apps/octoview-browser/src/main.rs` (add `mod image_loader;`)

**Step 1: Add image dependency**

In Cargo.toml, add:
```toml
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
```

**Step 2: Create image_loader.rs**

```rust
use std::collections::HashMap;

pub struct DecodedImage {
    pub pixels: Vec<u8>,  // RGBA
    pub width: u32,
    pub height: u32,
}

pub struct ImageCache {
    cache: HashMap<String, Option<DecodedImage>>,
}

impl ImageCache {
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Fetch and decode an image URL. Returns None on failure.
    /// Caches results (including failures as None).
    pub fn get_or_fetch(&mut self, url: &str, max_width: u32) -> Option<&DecodedImage> {
        if !self.cache.contains_key(url) {
            let decoded = fetch_and_decode(url, max_width);
            self.cache.insert(url.to_string(), decoded);
        }
        self.cache.get(url).and_then(|v| v.as_ref())
    }
}

fn fetch_and_decode(url: &str, max_width: u32) -> Option<DecodedImage> {
    // Skip data: URLs and empty src
    if url.is_empty() || url.starts_with("data:") {
        return None;
    }

    let client = reqwest::blocking::Client::builder()
        .user_agent("OctoView/0.2")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .ok()?;

    let response = client.get(url).send().ok()?;
    if !response.status().is_success() {
        return None;
    }

    let bytes = response.bytes().ok()?;
    // Cap at 5MB
    if bytes.len() > 5_000_000 {
        return None;
    }

    let img = image::load_from_memory(&bytes).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());

    // Resize if wider than max_width
    if w > max_width && max_width > 0 {
        let scale = max_width as f32 / w as f32;
        let new_h = (h as f32 * scale) as u32;
        let resized = image::imageops::resize(
            &rgba,
            max_width,
            new_h,
            image::imageops::FilterType::Triangle,
        );
        Some(DecodedImage {
            pixels: resized.into_raw(),
            width: max_width,
            height: new_h,
        })
    } else {
        Some(DecodedImage {
            pixels: rgba.into_raw(),
            width: w,
            height: h,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_cache_empty() {
        let mut cache = ImageCache::new();
        // Empty URL returns None
        assert!(cache.get_or_fetch("", 800).is_none());
    }

    #[test]
    fn test_image_cache_data_url_skipped() {
        let mut cache = ImageCache::new();
        assert!(cache.get_or_fetch("data:image/png;base64,abc", 800).is_none());
    }

    #[test]
    fn test_image_cache_clear() {
        let mut cache = ImageCache::new();
        cache.get_or_fetch("", 800);
        assert_eq!(cache.cache.len(), 1);
        cache.clear();
        assert_eq!(cache.cache.len(), 0);
    }
}
```

**Step 3: Add `mod image_loader;` to main.rs**

**Step 4: Build and test**

Expect 21 tests (18 + 3 new).

**Step 5: Commit**

```
feat(browser): image fetching, decoding, and caching module
```

---

## Task 4: Image Blitting and Layout Integration

**Files:**
- Modify: `apps/octoview-browser/src/framebuffer.rs`
- Modify: `apps/octoview-browser/src/layout.rs`
- Modify: `apps/octoview-browser/src/renderer.rs`
- Modify: `apps/octoview-browser/src/page.rs`

**Step 1: Add draw_image to framebuffer.rs**

```rust
/// Blit RGBA pixels onto the framebuffer. Clips to bounds.
pub fn draw_image(&mut self, x: i32, y: i32, img_w: u32, img_h: u32, pixels: &[u8]) {
    for row in 0..img_h {
        let dy = y + row as i32;
        if dy < 0 || dy >= self.height as i32 {
            continue;
        }
        for col in 0..img_w {
            let dx = x + col as i32;
            if dx < 0 || dx >= self.width as i32 {
                continue;
            }
            let src_idx = ((row * img_w + col) * 4) as usize;
            if src_idx + 3 >= pixels.len() {
                continue;
            }
            let a = pixels[src_idx + 3];
            if a == 0 {
                continue;
            }
            let sr = pixels[src_idx];
            let sg = pixels[src_idx + 1];
            let sb = pixels[src_idx + 2];
            if a == 255 {
                let dst_idx = ((dy as u32 * self.width + dx as u32) * 4) as usize;
                self.pixels[dst_idx] = sr;
                self.pixels[dst_idx + 1] = sg;
                self.pixels[dst_idx + 2] = sb;
                self.pixels[dst_idx + 3] = 255;
            } else {
                self.blend_pixel(dx, dy, sr, sg, sb, a);
            }
        }
    }
}
```

Add test:

```rust
#[test]
fn test_draw_image() {
    let mut fb = Framebuffer::new(10, 10);
    fb.clear(0, 0, 0);
    // 2x2 red image
    let pixels = vec![255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255];
    fb.draw_image(1, 1, 2, 2, &pixels);
    let idx = ((1 * 10 + 1) * 4) as usize;
    assert_eq!(fb.pixels[idx], 255); // red channel
    assert_eq!(fb.pixels[idx + 1], 0); // green
}
```

**Step 2: Update LayoutContent::Image in layout.rs**

Change `Image(String)` to include decoded pixel data:

```rust
/// Image with decoded pixels (or alt text fallback)
Image {
    alt: String,
    pixels: Option<Vec<u8>>,
    img_width: u32,
    img_height: u32,
},
```

Update `layout_document()` Image arm: instead of hardcoded 120px height, use actual image dimensions if available. The image data is passed in via a new parameter `images: &ImageCache` (or lookup by src URL).

Add `image_cache: &mut crate::image_loader::ImageCache` parameter to `layout_document()`. In the Image node arm, look up the image:

```rust
NodeType::Image => {
    let content_w = viewport_width - MARGIN_LEFT - MARGIN_RIGHT;
    let src = &node.src;
    let alt = &node.alt;
    let resolved_src = if !src.is_empty() {
        crate::page::resolve_image_url(&doc.url, src)
    } else {
        String::new()
    };
    let (pixels, img_w, img_h) = if !resolved_src.is_empty() {
        if let Some(img) = image_cache.get_or_fetch(&resolved_src, content_w as u32) {
            (Some(img.pixels.clone()), img.width, img.height)
        } else {
            (None, 0, 120) // fallback placeholder
        }
    } else {
        (None, 0, 120)
    };
    let height = if pixels.is_some() { img_h as f32 } else { 120.0 };
    // ... create LayoutBox with Image { alt, pixels, img_width, img_height }
}
```

**Step 3: Update renderer.rs to blit images**

In the Image match arm:

```rust
LayoutContent::Image { ref alt, ref pixels, img_width, img_height } => {
    if let Some(ref px) = pixels {
        fb.draw_image(screen_x, screen_y, *img_width, *img_height, px);
    } else {
        // Fallback: show alt text in gray box (existing behavior)
        // ...
    }
}
```

**Step 4: Update page.rs and main.rs call sites**

`load_page()` and `welcome_page()` need to pass `image_cache` to `layout_document()`. Add `ImageCache` to `App` struct. Pass `&mut self.image_cache` through the pipeline. Clear cache on navigation.

**Step 5: Build and test**

Expect 22 tests. Verify build compiles.

**Step 6: Commit**

```
feat(browser): inline image rendering with decode, resize, and blit
```

---

## Task 5: Tab Data Model and App Refactor

**Files:**
- Create: `apps/octoview-browser/src/tabs.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Create tabs.rs**

```rust
use crate::page::PageState;

pub struct Tab {
    pub url: String,
    pub title: String,
    pub page: Option<PageState>,
    pub scroll_y: f32,
    pub history: Vec<String>,
    pub history_idx: usize,
}

impl Tab {
    pub fn new() -> Self {
        Self {
            url: "octoview://welcome".to_string(),
            title: "New Tab".to_string(),
            page: None,
            scroll_y: 0.0,
            history: Vec::new(),
            history_idx: 0,
        }
    }

    pub fn can_back(&self) -> bool {
        self.history_idx > 0
    }

    pub fn can_forward(&self) -> bool {
        self.history_idx + 1 < self.history.len()
    }
}

pub struct TabBar {
    pub tabs: Vec<Tab>,
    pub active: usize,
}

impl TabBar {
    pub fn new() -> Self {
        Self {
            tabs: vec![Tab::new()],
            active: 0,
        }
    }

    pub fn active_tab(&self) -> &Tab {
        &self.tabs[self.active]
    }

    pub fn active_tab_mut(&mut self) -> &mut Tab {
        &mut self.tabs[self.active]
    }

    pub fn new_tab(&mut self) {
        self.tabs.push(Tab::new());
        self.active = self.tabs.len() - 1;
    }

    pub fn close_tab(&mut self, idx: usize) {
        if self.tabs.len() <= 1 {
            return; // Don't close last tab
        }
        self.tabs.remove(idx);
        if self.active >= self.tabs.len() {
            self.active = self.tabs.len() - 1;
        } else if self.active > idx {
            self.active -= 1;
        }
    }

    pub fn switch_to(&mut self, idx: usize) {
        if idx < self.tabs.len() {
            self.active = idx;
        }
    }

    pub fn next_tab(&mut self) {
        self.active = (self.active + 1) % self.tabs.len();
    }

    pub fn prev_tab(&mut self) {
        if self.active == 0 {
            self.active = self.tabs.len() - 1;
        } else {
            self.active -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tab_bar() {
        let tb = TabBar::new();
        assert_eq!(tb.tabs.len(), 1);
        assert_eq!(tb.active, 0);
    }

    #[test]
    fn test_add_and_close_tabs() {
        let mut tb = TabBar::new();
        tb.new_tab();
        tb.new_tab();
        assert_eq!(tb.tabs.len(), 3);
        assert_eq!(tb.active, 2); // newest tab is active
        tb.close_tab(1);
        assert_eq!(tb.tabs.len(), 2);
        assert_eq!(tb.active, 1); // adjusted
    }

    #[test]
    fn test_cannot_close_last_tab() {
        let mut tb = TabBar::new();
        tb.close_tab(0);
        assert_eq!(tb.tabs.len(), 1); // still 1
    }

    #[test]
    fn test_tab_cycling() {
        let mut tb = TabBar::new();
        tb.new_tab();
        tb.new_tab();
        tb.switch_to(0);
        tb.next_tab();
        assert_eq!(tb.active, 1);
        tb.switch_to(0);
        tb.prev_tab();
        assert_eq!(tb.active, 2); // wraps to last
    }

    #[test]
    fn test_tab_history() {
        let mut tab = Tab::new();
        assert!(!tab.can_back());
        assert!(!tab.can_forward());
        tab.history.push("url1".to_string());
        tab.history.push("url2".to_string());
        tab.history_idx = 1;
        assert!(tab.can_back());
        assert!(!tab.can_forward());
    }
}
```

**Step 2: Refactor App struct in main.rs**

Replace per-tab state fields with `TabBar`:

- Remove: `page`, `scroll_y`, `history`, `history_idx` from App
- Add: `tab_bar: TabBar`
- Add: `mod tabs;` to module declarations
- Update all accesses: `self.page` → `self.tab_bar.active_tab().page`, `self.scroll_y` → `self.tab_bar.active_tab().scroll_y`, etc.
- `navigate()` operates on `self.tab_bar.active_tab_mut()`
- `chrome.can_back` / `chrome.can_forward` read from active tab

**Step 3: Build and test**

Expect 27 tests (22 + 5 tab tests). All v1 functionality should still work since we're just moving state into Tab.

**Step 4: Commit**

```
refactor(browser): move per-page state into Tab, add TabBar data model
```

---

## Task 6: Tab Bar Rendering and Keyboard Shortcuts

**Files:**
- Modify: `apps/octoview-browser/src/chrome.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Add tab bar constants and rendering to chrome.rs**

Add constants:
```rust
pub const TAB_BAR_HEIGHT: f32 = 24.0;
pub const TAB_MAX_WIDTH: f32 = 160.0;
pub const TAB_PADDING: f32 = 8.0;
pub const TAB_CLOSE_SIZE: f32 = 14.0;
pub const TAB_ACTIVE_BG: [u8; 3] = [49, 50, 68];     // #313244
pub const TAB_INACTIVE_BG: [u8; 3] = [30, 30, 46];    // #1e1e2e
pub const TAB_NEW_BG: [u8; 3] = [35, 35, 50];         // #232332
```

Add `render_tab_bar()` function:
```rust
pub fn render_tab_bar(
    tabs: &[(String, bool)],  // (title, is_active) pairs
    fb: &mut Framebuffer,
    text: &mut TextRenderer,
    width: u32,
    y_offset: u32,
) {
    let h = TAB_BAR_HEIGHT as u32;
    // Background
    fb.draw_rect(0, y_offset as i32, width, h, TOOLBAR_BG[0], TOOLBAR_BG[1], TOOLBAR_BG[2]);
    // Bottom separator
    fb.draw_hline(0, (y_offset + h) as i32 - 1, width, SEPARATOR[0], SEPARATOR[1], SEPARATOR[2]);

    let mut x = 4i32;
    for (title, is_active) in tabs {
        let bg = if *is_active { TAB_ACTIVE_BG } else { TAB_INACTIVE_BG };
        let tab_w = TAB_MAX_WIDTH.min(width as f32 / tabs.len().max(1) as f32 - 4.0) as u32;
        fb.draw_rect(x, y_offset as i32 + 1, tab_w, h - 2, bg[0], bg[1], bg[2]);

        // Title text (truncated)
        let max_text_w = tab_w as f32 - TAB_CLOSE_SIZE - TAB_PADDING * 2.0;
        let display_title = truncate_title(title, max_text_w, text, 11.0);
        let text_color = if *is_active { TEXT_COLOR } else { DIM_TEXT };
        text.draw_text(fb, &display_title, x + TAB_PADDING as i32, y_offset as i32 + 5,
            11.0, FontStyle::Regular, text_color[0], text_color[1], text_color[2]);

        // Close button ✕
        let close_x = x + tab_w as i32 - TAB_CLOSE_SIZE as i32 - 2;
        text.draw_text(fb, "✕", close_x, y_offset as i32 + 5, 10.0,
            FontStyle::Regular, DIM_TEXT[0], DIM_TEXT[1], DIM_TEXT[2]);

        x += tab_w as i32 + 4;
    }

    // [+] new tab button
    let plus_w = 24u32;
    fb.draw_rect(x, y_offset as i32 + 1, plus_w, h - 2,
        TAB_NEW_BG[0], TAB_NEW_BG[1], TAB_NEW_BG[2]);
    text.draw_text(fb, "+", x + 7, y_offset as i32 + 5, 11.0,
        FontStyle::Regular, DIM_TEXT[0], DIM_TEXT[1], DIM_TEXT[2]);
}

fn truncate_title(title: &str, max_width: f32, text: &mut TextRenderer, size: f32) -> String {
    let (w, _) = text.measure_text(title, size, FontStyle::Regular);
    if w <= max_width {
        return title.to_string();
    }
    let mut truncated = String::new();
    for ch in title.chars() {
        truncated.push(ch);
        let (tw, _) = text.measure_text(&truncated, size, FontStyle::Regular);
        if tw > max_width - 12.0 { // leave room for "…"
            truncated.pop();
            truncated.push('…');
            return truncated;
        }
    }
    truncated
}
```

Add tab bar hit testing:
```rust
pub enum TabBarHit {
    Tab(usize),
    CloseTab(usize),
    NewTab,
    None,
}

pub fn tab_bar_hit_test(mx: f32, my: f32, tab_count: usize, y_offset: f32) -> TabBarHit {
    if my < y_offset || my > y_offset + TAB_BAR_HEIGHT {
        return TabBarHit::None;
    }
    // ... similar x-walking logic as toolbar_hit_test
}
```

**Step 2: Update content_top() and content_height()**

```rust
pub fn content_top() -> f32 {
    TOOLBAR_HEIGHT + TAB_BAR_HEIGHT
}

pub fn content_height(window_height: u32) -> f32 {
    window_height as f32 - TOOLBAR_HEIGHT - TAB_BAR_HEIGHT - STATUS_HEIGHT
}
```

**Step 3: Add keyboard shortcuts in main.rs**

In the keyboard handler:
- `Ctrl+T` → `self.tab_bar.new_tab()`, load welcome page in new tab
- `Ctrl+W` → `self.tab_bar.close_tab(self.tab_bar.active)`
- `Ctrl+Tab` → `self.tab_bar.next_tab()`
- `Ctrl+Shift+Tab` → `self.tab_bar.prev_tab()`
- Tab switch updates chrome URL/status from active tab

**Step 4: Render tab bar in render_full_frame()**

Call `chrome::render_tab_bar()` after `render_toolbar()`, passing tab titles and active state.

**Step 5: Handle tab bar clicks in handle_click()**

Test tab bar hit, switch/close/new as appropriate.

**Step 6: Build and test**

All 27 tests pass + visual verification of tab bar.

**Step 7: Commit**

```
feat(browser): tab bar rendering, keyboard shortcuts, multi-page browsing
```

---

## Task 7: Find State and Search Logic

**Files:**
- Create: `apps/octoview-browser/src/find.rs`
- Modify: `apps/octoview-browser/src/main.rs` (add `mod find;`)

**Step 1: Create find.rs**

```rust
use crate::layout::{LayoutBox, LayoutContent};

pub struct FindMatch {
    pub box_idx: usize,
    pub char_start: usize,
    pub char_len: usize,
    pub y_position: f32,
}

pub struct FindState {
    pub query: String,
    pub cursor: usize,
    pub matches: Vec<FindMatch>,
    pub active_match: usize,
    pub visible: bool,
}

impl FindState {
    pub fn new() -> Self {
        Self {
            query: String::new(),
            cursor: 0,
            matches: Vec::new(),
            active_match: 0,
            visible: false,
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
        if !self.visible {
            self.query.clear();
            self.cursor = 0;
            self.matches.clear();
            self.active_match = 0;
        }
    }

    pub fn search(&mut self, boxes: &[LayoutBox]) {
        self.matches.clear();
        self.active_match = 0;
        if self.query.is_empty() {
            return;
        }
        let query_lower = self.query.to_lowercase();

        for (idx, bx) in boxes.iter().enumerate() {
            let text = extract_box_text(&bx.content);
            if text.is_empty() {
                continue;
            }
            let text_lower = text.to_lowercase();
            let mut start = 0;
            while let Some(pos) = text_lower[start..].find(&query_lower) {
                let char_start = start + pos;
                self.matches.push(FindMatch {
                    box_idx: idx,
                    char_start,
                    char_len: self.query.len(),
                    y_position: bx.y,
                });
                start = char_start + 1;
            }
        }
    }

    pub fn next_match(&mut self) {
        if !self.matches.is_empty() {
            self.active_match = (self.active_match + 1) % self.matches.len();
        }
    }

    pub fn prev_match(&mut self) {
        if !self.matches.is_empty() {
            if self.active_match == 0 {
                self.active_match = self.matches.len() - 1;
            } else {
                self.active_match -= 1;
            }
        }
    }

    pub fn active_y(&self) -> Option<f32> {
        self.matches.get(self.active_match).map(|m| m.y_position)
    }

    pub fn match_count_text(&self) -> String {
        if self.query.is_empty() {
            String::new()
        } else if self.matches.is_empty() {
            "0/0".to_string()
        } else {
            format!("{}/{}", self.active_match + 1, self.matches.len())
        }
    }

    pub fn insert(&mut self, ch: char) {
        self.query.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.query[..self.cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.query.remove(prev);
            self.cursor = prev;
        }
    }
}

fn extract_box_text(content: &LayoutContent) -> String {
    match content {
        LayoutContent::Text(lines) | LayoutContent::CodeBlock(lines)
        | LayoutContent::Blockquote(lines) => {
            lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" ")
        }
        LayoutContent::Link(lines, _) => {
            lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" ")
        }
        LayoutContent::ListItem(prefix, lines) => {
            let text: String = lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" ");
            format!("{}{}", prefix, text)
        }
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{LayoutContent, LayoutStyle, TextLine};

    fn make_text_box(text: &str, y: f32) -> LayoutBox {
        LayoutBox {
            x: 0.0, y, width: 100.0, height: 20.0, node_idx: 0,
            content: LayoutContent::Text(vec![TextLine {
                text: text.to_string(), x_offset: 0.0, y_offset: 0.0,
            }]),
            style: LayoutStyle::default(),
        }
    }

    #[test]
    fn test_find_basic() {
        let boxes = vec![
            make_text_box("Hello world", 0.0),
            make_text_box("world peace", 20.0),
        ];
        let mut fs = FindState::new();
        fs.query = "world".to_string();
        fs.search(&boxes);
        assert_eq!(fs.matches.len(), 2);
        assert_eq!(fs.match_count_text(), "1/2");
    }

    #[test]
    fn test_find_case_insensitive() {
        let boxes = vec![make_text_box("Hello World", 0.0)];
        let mut fs = FindState::new();
        fs.query = "hello".to_string();
        fs.search(&boxes);
        assert_eq!(fs.matches.len(), 1);
    }

    #[test]
    fn test_find_navigation() {
        let boxes = vec![
            make_text_box("abc abc abc", 0.0),
        ];
        let mut fs = FindState::new();
        fs.query = "abc".to_string();
        fs.search(&boxes);
        assert_eq!(fs.matches.len(), 3);
        assert_eq!(fs.active_match, 0);
        fs.next_match();
        assert_eq!(fs.active_match, 1);
        fs.next_match();
        assert_eq!(fs.active_match, 2);
        fs.next_match();
        assert_eq!(fs.active_match, 0); // wraps
        fs.prev_match();
        assert_eq!(fs.active_match, 2); // wraps back
    }

    #[test]
    fn test_find_empty_query() {
        let boxes = vec![make_text_box("Hello", 0.0)];
        let mut fs = FindState::new();
        fs.search(&boxes);
        assert_eq!(fs.matches.len(), 0);
        assert_eq!(fs.match_count_text(), "");
    }
}
```

**Step 2: Add `mod find;` to main.rs, add `find_state: FindState` to App**

**Step 3: Build and test**

Expect 31 tests (27 + 4 find tests).

**Step 4: Commit**

```
feat(browser): find-in-page search logic with match navigation
```

---

## Task 8: Find Bar UI and Match Highlighting

**Files:**
- Modify: `apps/octoview-browser/src/chrome.rs`
- Modify: `apps/octoview-browser/src/renderer.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Add render_find_bar() to chrome.rs**

```rust
pub const FIND_BAR_HEIGHT: f32 = 28.0;
pub const FIND_BAR_BG: [u8; 3] = [35, 35, 50];
pub const MATCH_HIGHLIGHT: [u8; 3] = [255, 220, 50];     // yellow
pub const ACTIVE_HIGHLIGHT: [u8; 3] = [255, 140, 0];     // orange

pub fn render_find_bar(
    query: &str,
    cursor: usize,
    match_text: &str,
    fb: &mut Framebuffer,
    text: &mut TextRenderer,
    width: u32,
    y_offset: u32,
) {
    let h = FIND_BAR_HEIGHT as u32;
    fb.draw_rect(0, y_offset as i32, width, h, FIND_BAR_BG[0], FIND_BAR_BG[1], FIND_BAR_BG[2]);
    fb.draw_hline(0, (y_offset + h) as i32 - 1, width, SEPARATOR[0], SEPARATOR[1], SEPARATOR[2]);

    // "Find: " label
    text.draw_text(fb, "Find:", 8, y_offset as i32 + 6, 12.0, FontStyle::Regular,
        DIM_TEXT[0], DIM_TEXT[1], DIM_TEXT[2]);

    // Input field
    let input_x = 50i32;
    let input_w = 250u32;
    fb.draw_rect(input_x, y_offset as i32 + 3, input_w, h - 6,
        URL_BAR_BG[0], URL_BAR_BG[1], URL_BAR_BG[2]);
    text.draw_text(fb, query, input_x + 4, y_offset as i32 + 6, 12.0,
        FontStyle::Regular, TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2]);

    // Cursor
    let cursor_text = &query[..cursor.min(query.len())];
    let (cx, _) = text.measure_text(cursor_text, 12.0, FontStyle::Regular);
    fb.draw_vline(input_x + 4 + cx as i32, y_offset as i32 + 5, h - 10,
        TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2]);

    // Match count
    let count_x = input_x + input_w as i32 + 8;
    text.draw_text(fb, match_text, count_x, y_offset as i32 + 6, 12.0,
        FontStyle::Regular, DIM_TEXT[0], DIM_TEXT[1], DIM_TEXT[2]);

    // Nav buttons [↑] [↓] and close [✕]
    let nav_x = count_x + 60;
    text.draw_text(fb, "▲", nav_x, y_offset as i32 + 6, 12.0,
        FontStyle::Regular, TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2]);
    text.draw_text(fb, "▼", nav_x + 18, y_offset as i32 + 6, 12.0,
        FontStyle::Regular, TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2]);
    text.draw_text(fb, "✕", nav_x + 40, y_offset as i32 + 6, 12.0,
        FontStyle::Regular, DIM_TEXT[0], DIM_TEXT[1], DIM_TEXT[2]);
}
```

**Step 2: Update content_top() to account for find bar**

Pass `find_visible: bool` parameter to `content_top()`:

```rust
pub fn content_top(find_visible: bool) -> f32 {
    TOOLBAR_HEIGHT + TAB_BAR_HEIGHT + if find_visible { FIND_BAR_HEIGHT } else { 0.0 }
}
```

**Step 3: Add match highlighting to renderer.rs**

In `render_page()`, accept an optional `&FindState`. For each LayoutBox, check if any FindMatch references it. If so, draw highlight rectangles behind the matched text region. Use `text_renderer.measure_text()` to compute the x/width of the match substring.

**Step 4: Add keyboard handling in main.rs**

- `Ctrl+F` → toggle find bar, focus it
- When find bar is focused: text input goes to `find_state.insert()`, backspace to `find_state.backspace()`, Enter → `find_state.next_match()`, Shift+Enter → `find_state.prev_match()`, Escape → close
- After each keystroke in find bar: call `find_state.search(&layout_boxes)` and scroll to active match

**Step 5: Build and test**

All 31 tests pass + visual verification.

**Step 6: Commit**

```
feat(browser): find bar UI with live search, highlighting, and match navigation
```

---

## Task 9: Flow Mode Detection and Subprocess Bridge

**Files:**
- Create: `apps/octoview-browser/src/flow_mode.rs`
- Modify: `apps/octoview-browser/src/page.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Create flow_mode.rs**

```rust
use std::process::Command;

/// Check if a URL should trigger Flow Mode rendering.
pub fn is_flow_url(url: &str) -> bool {
    let url_lower = url.to_lowercase();
    url_lower.ends_with(".flow")
        || url_lower.starts_with("flow://")
}

/// Render a .flow file by invoking flowgpu-cli as a subprocess.
/// Returns RGBA pixel buffer if successful.
pub struct FlowRenderResult {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub title: String,
}

pub fn render_flow_file(
    path: &str,
    width: u32,
    height: u32,
) -> Result<FlowRenderResult, String> {
    // Resolve flow:// URLs to file paths
    let file_path = if path.starts_with("flow://") {
        path.strip_prefix("flow://").unwrap_or(path)
    } else if path.starts_with("file://") {
        path.strip_prefix("file://").unwrap_or(path)
    } else {
        path
    };

    // Check file exists
    if !std::path::Path::new(file_path).exists() {
        return Err(format!("File not found: {file_path}"));
    }

    // Shell out to flowgpu-cli with --render-to-buffer
    // For v2, this is a placeholder that reads the .flow file
    // and creates a simple text representation
    let source = std::fs::read_to_string(file_path)
        .map_err(|e| format!("Failed to read {file_path}: {e}"))?;

    let title = std::path::Path::new(file_path)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "Flow".to_string());

    // For now, return the source as text content for OVD extraction
    // Full OctoUI subprocess rendering is v2.5 (needs --render-to-buffer flag)
    Ok(FlowRenderResult {
        pixels: Vec::new(), // empty = use text fallback
        width,
        height,
        title,
    })
}

/// Create an OVD document from .flow source code (syntax-highlighted view).
pub fn flow_to_ovd(source: &str, file_path: &str) -> crate::ovd::OvdDocument {
    use crate::ovd::{NodeType, OvdNode, OvdDocument};

    let mut doc = OvdDocument::new(file_path);
    let title = std::path::Path::new(file_path)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "Flow Program".to_string());
    doc.title = format!("{} — Flow Mode", title);

    let mut page = OvdNode::new(NodeType::Page);
    page.text = file_path.to_string();
    doc.add_node(page);

    let mut h1 = OvdNode::new(NodeType::Heading);
    h1.text = format!("📄 {}", title);
    h1.level = 1;
    doc.add_node(h1);

    let mut info = OvdNode::new(NodeType::Paragraph);
    info.text = "Rendered in Flow Mode — native .flow content".to_string();
    doc.add_node(info);

    doc.add_node(OvdNode::new(NodeType::Separator));

    // Show source code
    let mut code = OvdNode::new(NodeType::CodeBlock);
    code.text = source.to_string();
    doc.add_node(code);

    doc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_flow_url() {
        assert!(is_flow_url("test.flow"));
        assert!(is_flow_url("flow://app"));
        assert!(is_flow_url("/path/to/file.flow"));
        assert!(!is_flow_url("https://example.com"));
        assert!(!is_flow_url("file.html"));
    }

    #[test]
    fn test_flow_to_ovd() {
        let doc = flow_to_ovd("let x = 1.0\nprint(x)", "test.flow");
        assert!(doc.title.contains("Flow Mode"));
        assert!(doc.nodes.len() >= 4); // page, heading, paragraph, separator, code
    }
}
```

**Step 2: Integrate into page.rs**

In `load_page()`, check `is_flow_url()` first:

```rust
pub fn load_page(url: &str, viewport_width: f32, text: &mut TextRenderer, ...) -> Result<PageState, String> {
    let start = std::time::Instant::now();

    if crate::flow_mode::is_flow_url(url) {
        return load_flow_page(url, viewport_width, text);
    }
    // ... existing HTML fetch pipeline
}

fn load_flow_page(url: &str, viewport_width: f32, text: &mut TextRenderer) -> Result<PageState, String> {
    let path = if url.starts_with("flow://") {
        url.strip_prefix("flow://").unwrap_or(url)
    } else if url.starts_with("file://") {
        url.strip_prefix("file://").unwrap_or(url)
    } else {
        url
    };

    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read: {e}"))?;

    let doc = crate::flow_mode::flow_to_ovd(&source, path);
    let node_count = doc.nodes.len();
    let title = doc.title.clone();
    let boxes = crate::layout::layout_document(&doc, text, viewport_width);
    let content_height = crate::layout::content_height(&boxes);

    Ok(PageState {
        url: url.to_string(),
        title,
        doc,
        layout: boxes,
        content_height,
        node_count,
        load_time_ms: 0,
    })
}
```

**Step 3: Update chrome to show Flow Mode**

Set `chrome.mode = BrowseMode::Flow` when loading a .flow URL.

**Step 4: Build and test**

Expect 33 tests (31 + 2 flow mode).

**Step 5: Commit**

```
feat(browser): Flow Mode detection and .flow file rendering as syntax view
```

---

## Task 10: OVD Inspector — Node Tree Panel

**Files:**
- Create: `apps/octoview-browser/src/inspector.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Create inspector.rs**

```rust
use crate::framebuffer::Framebuffer;
use crate::ovd::{OvdDocument, NodeType};
use crate::layout::LayoutBox;
use crate::text::{FontStyle, TextRenderer};

pub const INSPECTOR_WIDTH_PCT: f32 = 0.30;
const INSPECTOR_BG: [u8; 3] = [24, 24, 38];
const NODE_TEXT_COLOR: [u8; 3] = [205, 214, 244];
const SELECTED_BG: [u8; 3] = [49, 50, 68];
const TYPE_COLOR: [u8; 3] = [137, 180, 250];    // blue for type names
const VALUE_COLOR: [u8; 3] = [166, 227, 161];   // green for values
const LINE_HEIGHT: f32 = 16.0;
const INDENT_PX: f32 = 12.0;

pub struct InspectorState {
    pub visible: bool,
    pub scroll_y: f32,
    pub selected_node: Option<usize>,
}

impl InspectorState {
    pub fn new() -> Self {
        Self {
            visible: false,
            scroll_y: 0.0,
            selected_node: None,
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
        if !self.visible {
            self.selected_node = None;
            self.scroll_y = 0.0;
        }
    }

    pub fn panel_width(&self, window_width: u32) -> u32 {
        if self.visible {
            (window_width as f32 * INSPECTOR_WIDTH_PCT) as u32
        } else {
            0
        }
    }

    pub fn content_viewport_width(&self, window_width: u32) -> u32 {
        window_width - self.panel_width(window_width)
    }
}

/// Render the inspector panel on the right side of the framebuffer.
pub fn render_inspector(
    state: &InspectorState,
    doc: &OvdDocument,
    _boxes: &[LayoutBox],
    fb: &mut Framebuffer,
    text: &mut TextRenderer,
    top_y: u32,
    window_width: u32,
    window_height: u32,
) {
    if !state.visible {
        return;
    }

    let panel_w = state.panel_width(window_width);
    let panel_x = (window_width - panel_w) as i32;
    let panel_h = window_height - top_y;

    // Background
    fb.draw_rect(panel_x, top_y as i32, panel_w, panel_h,
        INSPECTOR_BG[0], INSPECTOR_BG[1], INSPECTOR_BG[2]);

    // Left border
    fb.draw_vline(panel_x, top_y as i32, panel_h,
        crate::chrome::SEPARATOR[0], crate::chrome::SEPARATOR[1], crate::chrome::SEPARATOR[2]);

    // Header
    text.draw_text(fb, "OVD Inspector", panel_x + 8, top_y as i32 + 4, 12.0,
        FontStyle::Bold, TYPE_COLOR[0], TYPE_COLOR[1], TYPE_COLOR[2]);
    fb.draw_hline(panel_x as u32, (top_y + 20) as i32, panel_w,
        crate::chrome::SEPARATOR[0], crate::chrome::SEPARATOR[1], crate::chrome::SEPARATOR[2]);

    // Node tree
    let tree_top = top_y as f32 + 24.0;
    let mut y = tree_top - state.scroll_y;

    for (idx, node) in doc.nodes.iter().enumerate() {
        let line_y = y as i32;

        // Skip if off screen
        if line_y > window_height as i32 {
            break;
        }
        if line_y + LINE_HEIGHT as i32 > top_y as i32 && line_y < window_height as i32 {
            // Selection highlight
            if state.selected_node == Some(idx) {
                fb.draw_rect(panel_x + 1, line_y, panel_w - 2, LINE_HEIGHT as u32,
                    SELECTED_BG[0], SELECTED_BG[1], SELECTED_BG[2]);
            }

            let indent = node.depth as f32 * INDENT_PX;
            let text_x = panel_x + 8 + indent as i32;

            // Type name
            let type_name = node_type_name(node.node_type);
            text.draw_text(fb, type_name, text_x, line_y + 2, 11.0,
                FontStyle::Regular, TYPE_COLOR[0], TYPE_COLOR[1], TYPE_COLOR[2]);

            // Truncated text preview
            if !node.text.is_empty() {
                let type_w = text.measure_text(type_name, 11.0, FontStyle::Regular).0;
                let preview_x = text_x + type_w as i32 + 4;
                let max_preview_w = panel_w as f32 - indent - type_w - 20.0;
                let preview = truncate_text(&node.text, max_preview_w, text, 10.0);
                text.draw_text(fb, &format!("\"{}\"", preview), preview_x, line_y + 2,
                    10.0, FontStyle::Regular, VALUE_COLOR[0], VALUE_COLOR[1], VALUE_COLOR[2]);
            }
        }
        y += LINE_HEIGHT;
    }

    // Properties panel (lower half) if a node is selected
    if let Some(idx) = state.selected_node {
        if let Some(node) = doc.nodes.get(idx) {
            let props_y = window_height as i32 / 2;
            fb.draw_hline(panel_x as u32, props_y, panel_w,
                crate::chrome::SEPARATOR[0], crate::chrome::SEPARATOR[1], crate::chrome::SEPARATOR[2]);

            text.draw_text(fb, "Properties", panel_x + 8, props_y + 4, 12.0,
                FontStyle::Bold, TYPE_COLOR[0], TYPE_COLOR[1], TYPE_COLOR[2]);

            let mut py = props_y + 22;
            let props = [
                ("type", node_type_name(node.node_type).to_string()),
                ("id", node.node_id.to_string()),
                ("depth", node.depth.to_string()),
                ("parent", node.parent_id.to_string()),
            ];
            for (key, val) in &props {
                text.draw_text(fb, key, panel_x + 8, py, 11.0,
                    FontStyle::Regular, crate::chrome::DIM_TEXT[0], crate::chrome::DIM_TEXT[1], crate::chrome::DIM_TEXT[2]);
                text.draw_text(fb, val, panel_x + 70, py, 11.0,
                    FontStyle::Regular, NODE_TEXT_COLOR[0], NODE_TEXT_COLOR[1], NODE_TEXT_COLOR[2]);
                py += 16;
            }
            if !node.text.is_empty() {
                text.draw_text(fb, "text", panel_x + 8, py, 11.0,
                    FontStyle::Regular, crate::chrome::DIM_TEXT[0], crate::chrome::DIM_TEXT[1], crate::chrome::DIM_TEXT[2]);
                let preview = if node.text.len() > 50 { &node.text[..50] } else { &node.text };
                text.draw_text(fb, preview, panel_x + 70, py, 11.0,
                    FontStyle::Regular, NODE_TEXT_COLOR[0], NODE_TEXT_COLOR[1], NODE_TEXT_COLOR[2]);
            }
        }
    }
}

fn node_type_name(nt: NodeType) -> &'static str {
    match nt {
        NodeType::Page => "Page",
        NodeType::Heading => "Heading",
        NodeType::Paragraph => "Para",
        NodeType::Link => "Link",
        NodeType::Image => "Image",
        NodeType::Table => "Table",
        NodeType::TableCell => "Cell",
        NodeType::List => "List",
        NodeType::ListItem => "Item",
        NodeType::CodeBlock => "Code",
        NodeType::Blockquote => "Quote",
        NodeType::Separator => "Sep",
        NodeType::TextSpan => "Span",
        _ => "Node",
    }
}

fn truncate_text(text: &str, max_width: f32, tr: &mut TextRenderer, size: f32) -> String {
    let clean: String = text.chars().filter(|c| !c.is_control()).take(60).collect();
    let (w, _) = tr.measure_text(&clean, size, FontStyle::Regular);
    if w <= max_width {
        return clean;
    }
    let mut result = String::new();
    for ch in clean.chars() {
        result.push(ch);
        let (tw, _) = tr.measure_text(&result, size, FontStyle::Regular);
        if tw > max_width - 10.0 {
            result.pop();
            result.push('…');
            return result;
        }
    }
    result
}

/// Hit test: returns the node index that was clicked in the inspector tree.
pub fn inspector_hit_test(
    state: &InspectorState,
    mx: f32,
    my: f32,
    doc: &OvdDocument,
    top_y: f32,
    window_width: u32,
) -> Option<usize> {
    if !state.visible {
        return None;
    }
    let panel_x = window_width as f32 * (1.0 - INSPECTOR_WIDTH_PCT);
    if mx < panel_x {
        return None;
    }

    let tree_top = top_y + 24.0;
    let relative_y = my - tree_top + state.scroll_y;
    if relative_y < 0.0 {
        return None;
    }
    let idx = (relative_y / LINE_HEIGHT) as usize;
    if idx < doc.nodes.len() {
        Some(idx)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspector_toggle() {
        let mut state = InspectorState::new();
        assert!(!state.visible);
        state.toggle();
        assert!(state.visible);
        state.toggle();
        assert!(!state.visible);
    }

    #[test]
    fn test_inspector_panel_width() {
        let state = InspectorState { visible: true, scroll_y: 0.0, selected_node: None };
        assert_eq!(state.panel_width(1000), 300);
        assert_eq!(state.content_viewport_width(1000), 700);
    }

    #[test]
    fn test_inspector_hidden_width() {
        let state = InspectorState::new();
        assert_eq!(state.panel_width(1000), 0);
        assert_eq!(state.content_viewport_width(1000), 1000);
    }
}
```

**Step 2: Integrate into main.rs**

- Add `mod inspector;`, `inspector_state: InspectorState` to App
- `F12` toggles inspector, triggers re-layout at new viewport width
- Content viewport width = `inspector.content_viewport_width(fb.width)`
- Render inspector after page content in `render_full_frame()`
- Handle inspector clicks in `handle_click()` — check if click is in inspector area
- Inspector scroll via mouse wheel when cursor is over panel

**Step 3: Add selection highlight to renderer.rs**

When rendering page content, if `inspector.selected_node == Some(idx)` for a LayoutBox, draw a blue outline around that box.

**Step 4: Build and test**

Expect 36 tests (33 + 3 inspector).

**Step 5: Commit**

```
feat(browser): OVD Inspector with node tree, properties, and selection
```

---

## Task 11: Bookmarks — Data Model and JSON Persistence

**Files:**
- Create: `apps/octoview-browser/src/bookmarks.rs`
- Modify: `apps/octoview-browser/Cargo.toml` (add serde, serde_json)
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Add dependencies**

In Cargo.toml:
```toml
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**Step 2: Create bookmarks.rs**

```rust
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookmarkEntry {
    pub url: String,
    pub title: String,
    pub added_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub url: String,
    pub title: String,
    pub visited_at: u64,
}

pub struct BookmarkStore {
    pub bookmarks: Vec<BookmarkEntry>,
    pub history: Vec<HistoryEntry>,
    data_dir: PathBuf,
}

const MAX_HISTORY: usize = 1000;

impl BookmarkStore {
    pub fn new() -> Self {
        let data_dir = dirs_path();
        std::fs::create_dir_all(&data_dir).ok();
        let mut store = Self {
            bookmarks: Vec::new(),
            history: Vec::new(),
            data_dir,
        };
        store.load();
        store
    }

    fn bookmarks_path(&self) -> PathBuf {
        self.data_dir.join("bookmarks.json")
    }

    fn history_path(&self) -> PathBuf {
        self.data_dir.join("history.json")
    }

    fn load(&mut self) {
        if let Ok(data) = std::fs::read_to_string(self.bookmarks_path()) {
            self.bookmarks = serde_json::from_str(&data).unwrap_or_default();
        }
        if let Ok(data) = std::fs::read_to_string(self.history_path()) {
            self.history = serde_json::from_str(&data).unwrap_or_default();
        }
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.bookmarks) {
            std::fs::write(self.bookmarks_path(), json).ok();
        }
        if let Ok(json) = serde_json::to_string_pretty(&self.history) {
            std::fs::write(self.history_path(), json).ok();
        }
    }

    pub fn add_bookmark(&mut self, url: &str, title: &str) {
        // Don't duplicate
        if self.bookmarks.iter().any(|b| b.url == url) {
            return;
        }
        self.bookmarks.push(BookmarkEntry {
            url: url.to_string(),
            title: title.to_string(),
            added_at: now_unix(),
        });
        self.save();
    }

    pub fn remove_bookmark(&mut self, idx: usize) {
        if idx < self.bookmarks.len() {
            self.bookmarks.remove(idx);
            self.save();
        }
    }

    pub fn is_bookmarked(&self, url: &str) -> bool {
        self.bookmarks.iter().any(|b| b.url == url)
    }

    pub fn add_history(&mut self, url: &str, title: &str) {
        self.history.push(HistoryEntry {
            url: url.to_string(),
            title: title.to_string(),
            visited_at: now_unix(),
        });
        // Trim oldest
        if self.history.len() > MAX_HISTORY {
            self.history.drain(0..self.history.len() - MAX_HISTORY);
        }
        self.save();
    }

    pub fn remove_history(&mut self, idx: usize) {
        if idx < self.history.len() {
            self.history.remove(idx);
            self.save();
        }
    }
}

fn dirs_path() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".octoview")
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bookmark_add_remove() {
        let mut store = BookmarkStore {
            bookmarks: Vec::new(),
            history: Vec::new(),
            data_dir: std::env::temp_dir().join("octoview_test_bm"),
        };
        store.add_bookmark("https://example.com", "Example");
        assert_eq!(store.bookmarks.len(), 1);
        assert!(store.is_bookmarked("https://example.com"));
        // No duplicate
        store.add_bookmark("https://example.com", "Example");
        assert_eq!(store.bookmarks.len(), 1);
        store.remove_bookmark(0);
        assert_eq!(store.bookmarks.len(), 0);
    }

    #[test]
    fn test_history_cap() {
        let mut store = BookmarkStore {
            bookmarks: Vec::new(),
            history: Vec::new(),
            data_dir: std::env::temp_dir().join("octoview_test_hist"),
        };
        for i in 0..1050 {
            store.history.push(HistoryEntry {
                url: format!("https://example.com/{i}"),
                title: format!("Page {i}"),
                visited_at: i as u64,
            });
        }
        // Manually trigger trim (normally done in add_history)
        if store.history.len() > MAX_HISTORY {
            store.history.drain(0..store.history.len() - MAX_HISTORY);
        }
        assert_eq!(store.history.len(), 1000);
    }
}
```

**Step 3: Add `mod bookmarks;` to main.rs, add `store: BookmarkStore` to App**

Call `store.add_history()` in `navigate()` after successful page load.

**Step 4: Build and test**

Expect 38 tests (36 + 2 bookmark tests).

**Step 5: Commit**

```
feat(browser): bookmarks and history persistence with JSON storage
```

---

## Task 12: Side Panels — Bookmarks and History UI

**Files:**
- Create: `apps/octoview-browser/src/side_panel.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Create side_panel.rs**

```rust
use crate::framebuffer::Framebuffer;
use crate::text::{FontStyle, TextRenderer};

pub const PANEL_WIDTH_PCT: f32 = 0.25;
const PANEL_BG: [u8; 3] = [24, 24, 38];
const ITEM_HOVER_BG: [u8; 3] = [35, 35, 50];
const TITLE_COLOR: [u8; 3] = [137, 180, 250];
const URL_COLOR: [u8; 3] = [130, 135, 160];
const TEXT_COLOR: [u8; 3] = [205, 214, 244];
const ITEM_HEIGHT: f32 = 36.0;

#[derive(Clone, Copy, PartialEq)]
pub enum SidePanelKind {
    None,
    Bookmarks,
    History,
}

pub struct SidePanelState {
    pub kind: SidePanelKind,
    pub scroll_y: f32,
    pub hover_idx: Option<usize>,
}

impl SidePanelState {
    pub fn new() -> Self {
        Self {
            kind: SidePanelKind::None,
            scroll_y: 0.0,
            hover_idx: None,
        }
    }

    pub fn toggle(&mut self, kind: SidePanelKind) {
        if self.kind == kind {
            self.kind = SidePanelKind::None;
        } else {
            self.kind = kind;
            self.scroll_y = 0.0;
            self.hover_idx = None;
        }
    }

    pub fn is_visible(&self) -> bool {
        self.kind != SidePanelKind::None
    }

    pub fn panel_width(&self, window_width: u32) -> u32 {
        if self.is_visible() {
            (window_width as f32 * PANEL_WIDTH_PCT) as u32
        } else {
            0
        }
    }
}

pub struct PanelItem {
    pub title: String,
    pub url: String,
    pub subtitle: String, // date or empty
}

/// Render side panel on the left.
pub fn render_side_panel(
    state: &SidePanelState,
    items: &[PanelItem],
    header: &str,
    fb: &mut Framebuffer,
    text: &mut TextRenderer,
    top_y: u32,
    window_width: u32,
    window_height: u32,
) {
    if !state.is_visible() {
        return;
    }

    let panel_w = state.panel_width(window_width);
    let panel_h = window_height - top_y;

    // Background
    fb.draw_rect(0, top_y as i32, panel_w, panel_h, PANEL_BG[0], PANEL_BG[1], PANEL_BG[2]);

    // Right border
    fb.draw_vline(panel_w as i32 - 1, top_y as i32, panel_h,
        crate::chrome::SEPARATOR[0], crate::chrome::SEPARATOR[1], crate::chrome::SEPARATOR[2]);

    // Header
    text.draw_text(fb, header, 8, top_y as i32 + 4, 13.0,
        FontStyle::Bold, TITLE_COLOR[0], TITLE_COLOR[1], TITLE_COLOR[2]);
    fb.draw_hline(0, (top_y + 22) as i32, panel_w,
        crate::chrome::SEPARATOR[0], crate::chrome::SEPARATOR[1], crate::chrome::SEPARATOR[2]);

    // Items
    let items_top = top_y as f32 + 26.0;
    let mut y = items_top - state.scroll_y;

    for (idx, item) in items.iter().enumerate() {
        let line_y = y as i32;

        if line_y > window_height as i32 {
            break;
        }
        if line_y + ITEM_HEIGHT as i32 > top_y as i32 && line_y < window_height as i32 {
            // Hover highlight
            if state.hover_idx == Some(idx) {
                fb.draw_rect(0, line_y, panel_w - 1, ITEM_HEIGHT as u32,
                    ITEM_HOVER_BG[0], ITEM_HOVER_BG[1], ITEM_HOVER_BG[2]);
            }

            // Title
            text.draw_text(fb, &item.title, 8, line_y + 2, 12.0,
                FontStyle::Regular, TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2]);

            // URL (truncated)
            let max_url_w = panel_w as f32 - 16.0;
            let url_display = if item.url.len() > 40 { &item.url[..40] } else { &item.url };
            text.draw_text(fb, url_display, 8, line_y + 16, 10.0,
                FontStyle::Regular, URL_COLOR[0], URL_COLOR[1], URL_COLOR[2]);
        }
        y += ITEM_HEIGHT;
    }
}

/// Hit test: returns index of clicked item, or None.
pub fn side_panel_hit_test(
    state: &SidePanelState,
    mx: f32,
    my: f32,
    item_count: usize,
    top_y: f32,
    window_width: u32,
) -> Option<usize> {
    if !state.is_visible() {
        return None;
    }
    let panel_w = state.panel_width(window_width) as f32;
    if mx > panel_w {
        return None;
    }
    let items_top = top_y + 26.0;
    let relative_y = my - items_top + state.scroll_y;
    if relative_y < 0.0 {
        return None;
    }
    let idx = (relative_y / ITEM_HEIGHT) as usize;
    if idx < item_count { Some(idx) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_toggle() {
        let mut state = SidePanelState::new();
        assert!(!state.is_visible());
        state.toggle(SidePanelKind::Bookmarks);
        assert!(state.is_visible());
        assert_eq!(state.kind, SidePanelKind::Bookmarks);
        state.toggle(SidePanelKind::Bookmarks); // toggle off
        assert!(!state.is_visible());
    }

    #[test]
    fn test_panel_switch() {
        let mut state = SidePanelState::new();
        state.toggle(SidePanelKind::Bookmarks);
        state.toggle(SidePanelKind::History); // switch to history
        assert_eq!(state.kind, SidePanelKind::History);
    }

    #[test]
    fn test_panel_width() {
        let state = SidePanelState { kind: SidePanelKind::Bookmarks, scroll_y: 0.0, hover_idx: None };
        assert_eq!(state.panel_width(1000), 250);
    }
}
```

**Step 2: Integrate into main.rs**

- Add `mod side_panel;`, add `side_panel: SidePanelState` to App
- `Ctrl+B` toggles bookmarks panel, `Ctrl+H` toggles history panel
- `Ctrl+D` adds current page to bookmarks
- Render side panel in `render_full_frame()` — after page content, before chrome
- Content viewport x-offset shifts right by panel width when visible
- Handle clicks in side panel: navigate to clicked item's URL
- Delete key removes selected item

**Step 3: Build and test**

Expect 41 tests (38 + 3 side panel tests).

**Step 4: Commit**

```
feat(browser): bookmarks and history side panels with keyboard shortcuts
```

---

## Task 13: Loom Engine Integration — LoomRenderer Struct

**Files:**
- Modify: `apps/octoview-browser/Cargo.toml`
- Create: `apps/octoview-browser/src/loom.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Add flowgpu-vulkan dependency**

In Cargo.toml, add under `[dependencies]`:
```toml
flowgpu-vulkan = { path = "../../arms/flowgpu-vulkan" }
```

**Step 2: Create loom.rs**

```rust
/// Loom Engine renderer — dispatches SPIR-V compute kernels for drawing.
/// This module wraps flowgpu-vulkan's VulkanCompute to provide a draw API
/// that matches the CPU Framebuffer interface.

// NOTE: This is the scaffolding. Actual Loom integration depends on
// VulkanCompute being able to share the same Vulkan device/instance
// as the browser's graphics pipeline, or creating its own.

pub struct LoomRenderer {
    pub enabled: bool,
    // Will hold VulkanCompute, kernel handles, GPU framebuffer, etc.
}

impl LoomRenderer {
    pub fn new() -> Self {
        Self {
            enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loom_renderer_creation() {
        let lr = LoomRenderer::new();
        assert!(!lr.enabled);
    }
}
```

**Step 3: Add `mod loom;` to main.rs**

**Step 4: Build and test**

Expect 42 tests. This task just establishes the dependency and module. The actual migration happens in Tasks 14-16.

**Step 5: Commit**

```
feat(browser): add Loom Engine scaffolding with flowgpu-vulkan dependency
```

---

## Task 14: Loom Clear and Rect Dispatch

**Files:**
- Modify: `apps/octoview-browser/src/loom.rs`

**Step 1: Implement Loom clear and rect**

Extend `LoomRenderer` to load `ui_clear.spv` and `ui_rect.spv` from OctoUI kernels, create a VulkanCompute instance, and provide `clear()` and `draw_rect()` methods that dispatch compute kernels instead of writing pixels to a CPU buffer.

The GPU framebuffer uses planar RGB format (separate R/G/B channels in a single buffer) matching OctoUI's convention:
- `[0..total)` = R channel
- `[total..2*total)` = G channel
- `[2*total..3*total)` = B channel

```rust
// loom.rs — expanded with actual dispatch
use flowgpu_vulkan::VulkanCompute;

pub struct LoomRenderer {
    pub enabled: bool,
    compute: Option<VulkanCompute>,
    framebuffer: Vec<f32>,  // planar RGB
    width: u32,
    height: u32,
}
```

This task implements the clear + rect kernels and verifies them with a simple test (dispatch clear, check buffer is filled).

**Step 2: Build and test**

**Step 3: Commit**

```
feat(browser): Loom Engine clear and rect GPU dispatch
```

---

## Task 15: Loom Text Blit and Image Blit

**Files:**
- Modify: `apps/octoview-browser/src/loom.rs`
- Create: `apps/octoview-browser/src/loom_text.rs` (GDI text atlas bridge)

**Step 1: Implement text blit dispatch**

Use OctoUI's `ui_text_blit.spv` kernel. The text pipeline:
1. Use GDI text functions to rasterize text (via flowgpu-cli's text_render.rs)
2. Upload glyph atlas to GPU heap
3. Dispatch `ui_text_blit.spv` for each text item

**Step 2: Create ui_image_blit.spv**

New SPIR-V kernel for image blitting. Push constants: dest_x, dest_y, src_w, src_h, screen_w, total_pixels, heap_offset. Copies RGBA pixels from heap to planar RGB framebuffer.

**Step 3: Build and test**

**Step 4: Commit**

```
feat(browser): Loom text blit and image blit GPU kernels
```

---

## Task 16: Remove CPU Framebuffer — Full Loom Rendering

**Files:**
- Modify: `apps/octoview-browser/src/renderer.rs`
- Modify: `apps/octoview-browser/src/main.rs`
- Modify: `apps/octoview-browser/src/pipeline.rs`

**Step 1: Replace CPU rendering with Loom dispatch**

In `renderer.rs`, instead of calling `fb.draw_rect()` and `text.draw_text()`, call the corresponding `LoomRenderer` methods. The renderer becomes a dispatch builder — it walks layout boxes and emits Loom draw commands instead of CPU pixel ops.

**Step 2: Remove staging buffer pipeline**

In `pipeline.rs`, the staging buffer upload and fullscreen triangle are no longer needed. Replace with a blit from the Loom compute output buffer directly to the swapchain image.

**Step 3: Update main.rs render loop**

The render loop changes from:
```
fb.clear() → render_page(fb, ...) → upload_framebuffer() → render_frame() → present
```
to:
```
loom.begin_frame() → render_page(loom, ...) → loom.execute() → present
```

**Step 4: Build and test**

All tests pass. Verify visual output matches CPU rendering.

**Step 5: Commit**

```
feat(browser): full Loom Engine rendering — CPU framebuffer removed
```

---

## Summary

| Task | Feature | New Tests | Cumulative |
|------|---------|-----------|------------|
| 1 | CSS inline style parser | 6 | 18 |
| 2 | CSS extraction + layout integration | 0 | 18 |
| 3 | Image fetch/decode module | 3 | 21 |
| 4 | Image blit + layout integration | 1 | 22 |
| 5 | Tab data model + App refactor | 5 | 27 |
| 6 | Tab bar rendering + shortcuts | 0 | 27 |
| 7 | Find search logic | 4 | 31 |
| 8 | Find bar UI + highlighting | 0 | 31 |
| 9 | Flow Mode detection + bridge | 2 | 33 |
| 10 | OVD Inspector panel | 3 | 36 |
| 11 | Bookmarks/history persistence | 2 | 38 |
| 12 | Side panels UI | 3 | 41 |
| 13 | Loom Engine scaffolding | 1 | 42 |
| 14 | Loom clear + rect dispatch | 0 | 42 |
| 15 | Loom text + image blit | 0 | 42 |
| 16 | Remove CPU framebuffer | 0 | 42 |

**16 tasks total.** Features 1-12 can be built and tested independently on CPU scaffolding. Tasks 13-16 (Loom migration) depend on all prior features working.
