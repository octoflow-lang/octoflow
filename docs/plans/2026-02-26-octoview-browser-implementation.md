# OctoView Browser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a GPU-native browser that renders HTML pages via the OVD pipeline onto a Vulkan surface, with URL navigation, scrolling, and link clicking.

**Architecture:** winit creates a window and Vulkan surface. ash manages the Vulkan instance, device, swapchain, and graphics pipeline. A CPU-side framebuffer (RGBA pixel buffer) is rendered by fontdue (text) and simple rect/line drawing, then uploaded to a GPU texture each frame. A fullscreen-quad graphics pipeline displays the texture on the swapchain. The layout engine maps OVD nodes to positioned boxes. HTML is fetched via reqwest, parsed by html5ever, extracted to OVD by the existing extract module, laid out, and rendered.

**Tech Stack:** Rust, winit 0.30, ash 0.38, fontdue 0.9, html5ever 0.38, reqwest 0.12

**Design Doc:** `docs/plans/2026-02-26-octoview-browser-design.md`

---

## Task 1: Create Crate and Window

**Files:**
- Create: `apps/octoview-browser/Cargo.toml`
- Create: `apps/octoview-browser/src/main.rs`
- Modify: `Cargo.toml` (workspace members)

**Step 1: Add to workspace**

In root `Cargo.toml`, add `"apps/octoview-browser"` to workspace members list.

**Step 2: Create Cargo.toml**

```toml
[package]
name = "octoview-browser"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "octoview-browser"
path = "src/main.rs"

[dependencies]
winit = "0.30"
ash = "0.38"
raw-window-handle = "0.6"
fontdue = "0.9"
html5ever = "0.38"
markup5ever = "0.38"
markup5ever_rcdom = "0.38"
reqwest = { version = "0.12", features = ["blocking", "native-tls"] }
url = "2"
```

**Step 3: Create main.rs with a winit window**

```rust
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("OctoView Browser")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 800));
            self.window = Some(event_loop.create_window(attrs).unwrap());
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                // Will render here later
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { window: None };
    event_loop.run_app(&mut app).unwrap();
}
```

**Step 4: Build and verify window opens**

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" run -p octoview-browser
```

Expected: A blank window titled "OctoView Browser" opens at 1280x800. Close it to exit.

**Step 5: Commit**

```bash
git add apps/octoview-browser/ Cargo.toml
git commit -m "octoview-browser: winit window scaffold"
```

---

## Task 2: Vulkan Instance and Surface

**Files:**
- Create: `apps/octoview-browser/src/vulkan.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Create vulkan.rs with instance + surface creation**

This module handles all Vulkan setup: instance, surface, physical device, logical device, swapchain, and presentation.

Key functions to implement:
- `VulkanContext::new(window: &Window)` — create instance with surface extensions, create surface from window handle
- Physical device selection (prefer discrete GPU)
- Queue family selection (need graphics queue that supports present)
- Logical device creation
- Swapchain creation (VK_FORMAT_B8G8R8A8_SRGB, VK_PRESENT_MODE_FIFO)

Use `ash::khr::surface` and `ash::khr::win32_surface` extensions.

The struct should hold:
```rust
pub struct VulkanContext {
    entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    queue_family_index: u32,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}
```

**Step 2: Integrate into main.rs**

After window creation in `resumed()`, initialize VulkanContext. On RedrawRequested, acquire swapchain image, submit empty command buffer, present.

**Step 3: Build and verify**

Window should open and show a black/undefined screen without crashing. Vulkan validation layers should report no errors.

**Step 4: Commit**

```bash
git add apps/octoview-browser/src/
git commit -m "octoview-browser: Vulkan instance, surface, swapchain"
```

---

## Task 3: Framebuffer Texture and Fullscreen Quad

**Files:**
- Create: `apps/octoview-browser/src/pipeline.rs`
- Create: `apps/octoview-browser/shaders/fullscreen.vert` (GLSL, compile to SPIR-V)
- Create: `apps/octoview-browser/shaders/fullscreen.frag` (GLSL, compile to SPIR-V)
- Modify: `apps/octoview-browser/src/vulkan.rs`

**Step 1: Write shaders**

Vertex shader (fullscreen triangle, no vertex buffer needed):
```glsl
#version 450
layout(location = 0) out vec2 fragTexCoord;
void main() {
    fragTexCoord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(fragTexCoord * 2.0 - 1.0, 0.0, 1.0);
}
```

Fragment shader (sample texture):
```glsl
#version 450
layout(location = 0) in vec2 fragTexCoord;
layout(binding = 0) uniform sampler2D texSampler;
layout(location = 0) out vec4 outColor;
void main() {
    outColor = texture(texSampler, fragTexCoord);
}
```

Compile with: `glslangValidator -V fullscreen.vert -o fullscreen.vert.spv`

**Step 2: Create pipeline.rs**

- Create render pass (one color attachment matching swapchain format)
- Create framebuffers for each swapchain image view
- Load compiled SPIR-V shaders
- Create graphics pipeline (no vertex input, triangle list, 3 vertices)
- Create descriptor set layout (one combined image sampler at binding 0)
- Create texture image (VK_FORMAT_R8G8B8A8_SRGB, staging buffer upload)
- Create image view + sampler for the texture
- Create descriptor pool + descriptor set

**Step 3: Implement frame rendering**

Add to VulkanContext:
```rust
pub fn upload_framebuffer(&mut self, pixels: &[u8], width: u32, height: u32)
pub fn render_frame(&mut self) -> Result<(), vk::Result>
```

`upload_framebuffer` copies CPU RGBA pixels to the staging buffer, then issues a buffer-to-image copy command.

`render_frame` acquires a swapchain image, begins render pass, binds pipeline, binds descriptor set, draws 3 vertices (fullscreen triangle), ends render pass, submits, presents.

**Step 4: Test with a solid color**

In main.rs, create a Vec<u8> filled with a solid color (e.g., dark blue #1a1a2e), upload, render. Window should show that solid color.

**Step 5: Commit**

```bash
git add apps/octoview-browser/
git commit -m "octoview-browser: fullscreen quad pipeline, framebuffer upload"
```

---

## Task 4: CPU Framebuffer Rendering

**Files:**
- Create: `apps/octoview-browser/src/framebuffer.rs`

**Step 1: Create Framebuffer struct**

```rust
pub struct Framebuffer {
    pub pixels: Vec<u8>,  // RGBA, row-major
    pub width: u32,
    pub height: u32,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self
    pub fn clear(&mut self, r: u8, g: u8, b: u8)
    pub fn draw_rect(&mut self, x: i32, y: i32, w: u32, h: u32, r: u8, g: u8, b: u8)
    pub fn draw_rect_outline(&mut self, x: i32, y: i32, w: u32, h: u32, r: u8, g: u8, b: u8)
    pub fn draw_hline(&mut self, x: i32, y: i32, w: u32, r: u8, g: u8, b: u8)
    pub fn set_pixel(&mut self, x: i32, y: i32, r: u8, g: u8, b: u8, a: u8)
    pub fn blend_pixel(&mut self, x: i32, y: i32, r: u8, g: u8, b: u8, coverage: u8)
}
```

All drawing functions clip to bounds (negative x/y, overflow).

**Step 2: Test rect drawing**

Draw a white rectangle on dark background. Upload to Vulkan texture, verify it appears on screen.

**Step 3: Commit**

```bash
git add apps/octoview-browser/src/framebuffer.rs
git commit -m "octoview-browser: CPU framebuffer with rect drawing"
```

---

## Task 5: Text Rendering with fontdue

**Files:**
- Create: `apps/octoview-browser/src/text.rs`
- Embed: A default font (include_bytes! a TTF file, or use a system font path)

**Step 1: Create text rendering module**

```rust
pub struct TextRenderer {
    font: fontdue::Font,
    font_bold: fontdue::Font,
    font_mono: fontdue::Font,
    glyph_cache: HashMap<(char, u32, FontStyle), GlyphEntry>,
}

struct GlyphEntry {
    metrics: fontdue::Metrics,
    bitmap: Vec<u8>,  // alpha coverage per pixel
}

pub enum FontStyle {
    Regular,
    Bold,
    Monospace,
}

impl TextRenderer {
    pub fn new() -> Self  // loads embedded/system fonts
    pub fn measure_text(&mut self, text: &str, size: f32, style: FontStyle) -> (f32, f32)
    pub fn draw_text(
        &mut self,
        fb: &mut Framebuffer,
        text: &str,
        x: i32, y: i32,
        size: f32,
        style: FontStyle,
        r: u8, g: u8, b: u8,
    )
    pub fn line_height(&self, size: f32) -> f32
    pub fn wrap_text(&mut self, text: &str, max_width: f32, size: f32, style: FontStyle) -> Vec<String>
}
```

Use fontdue's `Font::rasterize()` to get per-glyph bitmaps. Cache them in a HashMap keyed by (char, size_px, style). Draw by blending each glyph's coverage into the framebuffer.

For fonts, embed a free TTF (e.g., Inter for proportional, JetBrains Mono for monospace) via `include_bytes!`. Or use system font discovery as fallback.

**Step 2: Test text rendering**

Draw "OctoView Browser" in white on dark background at 24px. Verify it renders correctly in the window.

**Step 3: Commit**

```bash
git add apps/octoview-browser/src/text.rs
git commit -m "octoview-browser: fontdue text rendering with glyph cache"
```

---

## Task 6: OVD Layout Engine

**Files:**
- Create: `apps/octoview-browser/src/layout.rs`

**Step 1: Define layout types**

```rust
pub struct LayoutBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub node_idx: usize,          // index into OvdDocument.nodes
    pub content: LayoutContent,
    pub style: LayoutStyle,
}

pub enum LayoutContent {
    Block,                         // container
    Text(Vec<TextLine>),           // wrapped text lines
    Link(Vec<TextLine>, String),   // wrapped text + href
    Image(String),                 // alt text (placeholder)
    Separator,                     // horizontal rule
    ListBullet(String),            // bullet/number prefix
}

pub struct TextLine {
    pub text: String,
    pub x_offset: f32,
    pub y_offset: f32,
}

pub struct LayoutStyle {
    pub font_size: f32,
    pub bold: bool,
    pub italic: bool,
    pub monospace: bool,
    pub text_color: [u8; 3],
    pub bg_color: Option<[u8; 3]>,
    pub left_margin: f32,
    pub top_margin: f32,
    pub bottom_margin: f32,
    pub indent: f32,
}
```

**Step 2: Implement layout function**

```rust
pub fn layout_document(
    doc: &OvdDocument,
    text: &mut TextRenderer,
    viewport_width: f32,
) -> Vec<LayoutBox>
```

Walk OVD nodes, map each NodeType to layout behavior:
- Heading: block, font size by level (H1=32, H2=24, H3=20, H4=18, H5=16, H6=14), bold, margin
- Paragraph: block, wrap text to viewport width minus margins
- Link: inline within parent, blue color (#0066CC), underline not needed (color suffices)
- CodeBlock: monospace, background #2d2d2d, padding
- Blockquote: indented, left border (draw as colored rect)
- List/ListItem: indented, bullet prefix
- Table/Row/Cell: equal-width columns, simple grid
- Separator: thin horizontal line

Track cursor_y as running vertical position. Each block advances cursor_y by its height plus margins.

**Step 3: Write unit tests**

Test with a small OvdDocument (2-3 nodes). Verify layout boxes have expected positions and sizes.

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_heading_layout() {
        let mut doc = OvdDocument::new("test://");
        doc.add_node(OvdNode::new(0, NodeType::Page, -1, 0));
        let mut h = OvdNode::new(0, NodeType::Heading, 0, 1);
        h.text = "Hello".to_string();
        h.level = 1;
        doc.add_node(h);

        let mut tr = TextRenderer::new();
        let boxes = layout_document(&doc, &mut tr, 800.0);

        assert!(boxes.len() >= 1);
        assert!(boxes[0].style.font_size >= 28.0);
        assert!(boxes[0].style.bold);
    }
}
```

**Step 4: Commit**

```bash
git add apps/octoview-browser/src/layout.rs
git commit -m "octoview-browser: OVD layout engine (block flow, text wrap)"
```

---

## Task 7: Page Renderer (Layout → Framebuffer)

**Files:**
- Create: `apps/octoview-browser/src/renderer.rs`

**Step 1: Implement render function**

```rust
pub fn render_page(
    fb: &mut Framebuffer,
    boxes: &[LayoutBox],
    text_renderer: &mut TextRenderer,
    scroll_y: f32,
    viewport_y: f32,       // top of content area (below toolbar)
    viewport_height: f32,
)
```

For each LayoutBox, offset by -scroll_y, skip if outside viewport. Draw:
- Background rect if bg_color is set
- Text lines using TextRenderer
- Link text in blue
- Blockquote left border (4px colored rect)
- Separator as thin horizontal line
- Code blocks with dark background + monospace text
- List bullets

**Step 2: Test with hardcoded OVD**

Create a small OvdDocument with a heading + paragraph + link. Layout it, render to framebuffer, verify visually.

**Step 3: Commit**

```bash
git add apps/octoview-browser/src/renderer.rs
git commit -m "octoview-browser: page renderer (layout boxes to framebuffer)"
```

---

## Task 8: Browser Chrome (Toolbar + Status Bar)

**Files:**
- Create: `apps/octoview-browser/src/chrome.rs`

**Step 1: Define chrome layout**

```rust
pub struct BrowserChrome {
    pub url_text: String,
    pub url_focused: bool,
    pub url_cursor: usize,
    pub status_text: String,
    pub mode: BrowseMode,
    pub can_back: bool,
    pub can_forward: bool,
}

pub enum BrowseMode {
    Web,
    Flow,
}

// Constants
pub const TOOLBAR_HEIGHT: f32 = 40.0;
pub const STATUS_HEIGHT: f32 = 24.0;
pub const BUTTON_SIZE: f32 = 32.0;
```

**Step 2: Implement chrome rendering**

```rust
pub fn render_toolbar(chrome: &BrowserChrome, fb: &mut Framebuffer, text: &mut TextRenderer)
pub fn render_status_bar(chrome: &BrowserChrome, fb: &mut Framebuffer, text: &mut TextRenderer)
```

Toolbar: dark background (#2d2d2d), back/fwd buttons (◀ ▶ as text or simple triangles), reload button (↻), URL input field (white text on slightly lighter background, with cursor when focused).

Status bar: bottom of window, dark background, left-aligned text showing node count + mode + render time.

**Step 3: Test toolbar rendering**

Render toolbar with a URL string. Verify buttons and URL field are visible.

**Step 4: Commit**

```bash
git add apps/octoview-browser/src/chrome.rs
git commit -m "octoview-browser: browser chrome (toolbar, URL bar, status)"
```

---

## Task 9: HTML Fetch + OVD Integration

**Files:**
- Create: `apps/octoview-browser/src/page.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Implement page loading**

Either import OctoView CLI modules as a library dependency, or duplicate the minimal needed code (extract_html, fetch_url). Simplest: add `octoview` as a workspace dependency and make its key modules pub, OR copy extract.rs/fetch.rs/ovd.rs into the browser crate.

Recommended: Copy the 3 essential files (ovd.rs, extract.rs, fetch.rs) into the browser crate's src/ to avoid coupling. These are self-contained.

```rust
// page.rs
pub struct PageState {
    pub url: String,
    pub doc: OvdDocument,
    pub layout: Vec<LayoutBox>,
    pub content_height: f32,
    pub load_time_ms: u128,
}

pub fn load_page(url: &str, viewport_width: f32, text: &mut TextRenderer) -> Result<PageState, String> {
    let start = std::time::Instant::now();

    // Fetch HTML
    let result = fetch::fetch_url(url)?;

    // Extract OVD
    let doc = extract::extract_html(&result.html, &result.url);

    // Layout
    let boxes = layout::layout_document(&doc, text, viewport_width);
    let content_height = boxes.iter()
        .map(|b| b.y + b.height)
        .fold(0.0f32, f32::max);

    let load_time_ms = start.elapsed().as_millis();

    Ok(PageState {
        url: result.url,
        doc,
        layout: boxes,
        content_height,
        load_time_ms,
    })
}
```

**Step 2: Wire into main event loop**

On startup, load a default page (e.g., `https://example.com` or a built-in welcome page). Store PageState in App struct. On RedrawRequested:
1. Clear framebuffer (dark background)
2. Render toolbar
3. Render page content (layout boxes with scroll offset)
4. Render status bar
5. Upload framebuffer to GPU
6. Present

**Step 3: Test**

Launch browser. It should load example.com and display rendered text content.

**Step 4: Commit**

```bash
git add apps/octoview-browser/src/
git commit -m "octoview-browser: HTML fetch + OVD extraction + rendering pipeline"
```

---

## Task 10: Input Handling (Keyboard + Mouse + Scroll)

**Files:**
- Create: `apps/octoview-browser/src/input.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Handle window events**

In the ApplicationHandler, handle:
- `WindowEvent::KeyboardInput` — keyboard shortcuts
- `WindowEvent::MouseWheel` — scroll content
- `WindowEvent::MouseInput` — click (link navigation, button clicks)
- `WindowEvent::CursorMoved` — track mouse position for hit testing

**Step 2: Implement keyboard handling**

```rust
pub fn handle_key(app: &mut App, key: Key, modifiers: ModifiersState) {
    match key {
        // Ctrl+L: Focus URL bar
        Key::Character("l") if modifiers.control_key() => app.chrome.url_focused = true,
        // Escape: Unfocus URL bar
        Key::Named(NamedKey::Escape) => app.chrome.url_focused = false,
        // Enter: Navigate to URL (when URL bar focused)
        Key::Named(NamedKey::Enter) if app.chrome.url_focused => {
            let url = app.chrome.url_text.clone();
            app.navigate(&url);
            app.chrome.url_focused = false;
        }
        // F5: Reload
        Key::Named(NamedKey::F5) => app.reload(),
        // Alt+Left: Back
        Key::Named(NamedKey::ArrowLeft) if modifiers.alt_key() => app.go_back(),
        // Alt+Right: Forward
        Key::Named(NamedKey::ArrowRight) if modifiers.alt_key() => app.go_forward(),
        // PageDown/Space: Scroll down
        Key::Named(NamedKey::PageDown) | Key::Named(NamedKey::Space) => {
            app.scroll_y += app.viewport_height() * 0.8;
        }
        // PageUp: Scroll up
        Key::Named(NamedKey::PageUp) => {
            app.scroll_y -= app.viewport_height() * 0.8;
        }
        // Home: Top
        Key::Named(NamedKey::Home) => app.scroll_y = 0.0,
        // End: Bottom
        Key::Named(NamedKey::End) => app.scroll_y = app.max_scroll(),
        // URL bar text input
        _ if app.chrome.url_focused => handle_url_input(app, key),
        _ => {}
    }
    app.clamp_scroll();
}
```

**Step 3: Implement scroll**

Mouse wheel events adjust scroll_y. Clamp between 0 and (content_height - viewport_height).

**Step 4: Commit**

```bash
git add apps/octoview-browser/src/input.rs
git commit -m "octoview-browser: keyboard, mouse, scroll input handling"
```

---

## Task 11: Link Hit Testing + Navigation

**Files:**
- Modify: `apps/octoview-browser/src/input.rs`
- Modify: `apps/octoview-browser/src/main.rs`

**Step 1: Implement hit testing**

When mouse clicks in the content area:
```rust
pub fn hit_test(boxes: &[LayoutBox], x: f32, y: f32, scroll_y: f32) -> Option<&LayoutBox> {
    for b in boxes.iter().rev() {
        let by = b.y - scroll_y;
        if x >= b.x && x <= b.x + b.width && y >= by && y <= by + b.height {
            if matches!(b.content, LayoutContent::Link(_, _)) {
                return Some(b);
            }
        }
    }
    None
}
```

**Step 2: Implement navigation**

```rust
pub struct NavigationState {
    pub history: Vec<String>,   // URL history
    pub current: usize,         // index into history
}

impl NavigationState {
    pub fn navigate(&mut self, url: &str) { ... }
    pub fn back(&mut self) -> Option<&str> { ... }
    pub fn forward(&mut self) -> Option<&str> { ... }
    pub fn can_back(&self) -> bool { ... }
    pub fn can_forward(&self) -> bool { ... }
}
```

On link click: resolve relative URL against current page URL, navigate. On toolbar back/forward button click: navigate history.

**Step 3: Implement cursor change**

When mouse hovers over a link, change cursor to pointer (winit supports `CursorIcon::Pointer`).

**Step 4: Test**

Load HN. Click a link. Verify navigation works and new page renders. Click back button. Verify original page returns.

**Step 5: Commit**

```bash
git add apps/octoview-browser/src/
git commit -m "octoview-browser: link clicking, navigation, history"
```

---

## Task 12: Polish and Integration Test

**Files:**
- Modify: `apps/octoview-browser/src/main.rs`
- Modify: various

**Step 1: Dark theme colors**

Define the OctoView color palette:
```rust
pub const BG_COLOR: [u8; 3] = [18, 18, 30];        // #12121e (dark navy)
pub const TOOLBAR_BG: [u8; 3] = [30, 30, 46];       // #1e1e2e
pub const URL_BAR_BG: [u8; 3] = [45, 45, 65];       // #2d2d41
pub const STATUS_BG: [u8; 3] = [24, 24, 38];        // #181826
pub const TEXT_COLOR: [u8; 3] = [205, 214, 244];     // #cdd6f4 (soft white)
pub const LINK_COLOR: [u8; 3] = [137, 180, 250];     // #89b4fa (blue)
pub const HEADING_COLOR: [u8; 3] = [245, 224, 220];  // #f5e0dc (warm white)
pub const CODE_BG: [u8; 3] = [30, 30, 46];           // #1e1e2e
pub const CODE_COLOR: [u8; 3] = [166, 227, 161];     // #a6e3a1 (green)
pub const BUTTON_BG: [u8; 3] = [49, 50, 68];         // #313244
pub const SEPARATOR_COLOR: [u8; 3] = [69, 71, 90];   // #45475a
```

**Step 2: Welcome page**

When launched with no arguments, show a built-in welcome page:
```
OctoView Browser v0.1

The web, rewoven. GPU-native browser for developers.

Type a URL in the address bar (Ctrl+L) to get started.

Keyboard Shortcuts:
  Ctrl+L    Focus URL bar
  Enter     Navigate to URL
  F5        Reload page
  Alt+←     Back
  Alt+→     Forward
  Space     Scroll down
  Home/End  Top/Bottom
```

**Step 3: Command-line URL argument**

```rust
fn main() {
    let url = std::env::args().nth(1);
    // If URL provided, load it on startup
}
```

**Step 4: Integration test**

Test these scenarios manually:
1. Launch → welcome page renders ✓
2. Ctrl+L → type `news.ycombinator.com` → Enter → HN loads ✓
3. Scroll down (mouse wheel) → content scrolls ✓
4. Click a link → navigates to new page ✓
5. Alt+Left → goes back to HN ✓
6. Status bar shows node count + render time ✓

**Step 5: Final commit**

```bash
git add apps/octoview-browser/
git commit -m "octoview-browser v0.1: GPU-native browser with OVD pipeline"
```

---

## Summary

| Task | Component | Estimated Complexity |
|------|-----------|---------------------|
| 1 | Crate + winit window | Small |
| 2 | Vulkan instance + swapchain | Medium (boilerplate) |
| 3 | Fullscreen quad pipeline | Medium (GPU plumbing) |
| 4 | CPU framebuffer drawing | Small |
| 5 | fontdue text rendering | Medium |
| 6 | OVD layout engine | Medium-Large |
| 7 | Page renderer | Medium |
| 8 | Browser chrome (toolbar) | Medium |
| 9 | HTML fetch + integration | Medium |
| 10 | Input handling | Medium |
| 11 | Link navigation + history | Medium |
| 12 | Polish + test | Small |

**Critical path:** Tasks 1→2→3→4→5→6→7→9 (window to rendered page). Tasks 8, 10, 11 can partially overlap.

**GPU upgrade path (v1.1):** Replace CPU framebuffer drawing in Tasks 4-5 with SPIR-V compute shader dispatches (reusing OctoUI kernel concepts). The layout engine and browser chrome stay the same.
