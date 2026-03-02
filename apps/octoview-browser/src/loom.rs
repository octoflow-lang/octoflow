/// Loom Engine renderer -- abstraction layer over GPU compute dispatch.
///
/// v2: Establishes the API surface matching Framebuffer interface.
/// GPU dispatch will replace CPU fallback when Vulkan device sharing is implemented.
///
/// The migration path:
///   v2.0: LoomRenderer wraps Framebuffer (this)
///   v2.1: clear() and draw_rect() dispatch to ui_clear.spv / ui_rect.spv
///   v2.2: draw_text() dispatches to ui_text_blit.spv
///   v2.3: draw_image() dispatches to ui_image_blit.spv
///   v2.4: Remove Framebuffer entirely -- GPU buffer -> swapchain direct

use crate::framebuffer::Framebuffer;

/// Rendering backend selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderBackend {
    /// CPU framebuffer (current, working)
    Cpu,
    /// GPU compute via Loom Engine (future)
    #[allow(dead_code)]
    Loom,
}

/// Loom-aware renderer that can switch between CPU and GPU backends.
#[allow(dead_code)]
pub struct LoomRenderer {
    pub backend: RenderBackend,
    // Future: VulkanCompute handle, kernel cache, GPU framebuffer
}

#[allow(dead_code)]
impl LoomRenderer {
    pub fn new() -> Self {
        Self {
            backend: RenderBackend::Cpu,
        }
    }

    /// Clear the framebuffer. On GPU backend, dispatches ui_clear.spv.
    pub fn clear(&self, fb: &mut Framebuffer, r: u8, g: u8, b: u8) {
        match self.backend {
            RenderBackend::Cpu => fb.clear(r, g, b),
            RenderBackend::Loom => {
                // TODO: dispatch ui_clear.spv with push constants (r, g, b, width, height)
                fb.clear(r, g, b); // fallback
            }
        }
    }

    /// Draw a filled rectangle. On GPU backend, dispatches ui_rect.spv.
    pub fn draw_rect(
        &self,
        fb: &mut Framebuffer,
        x: i32,
        y: i32,
        w: u32,
        h: u32,
        r: u8,
        g: u8,
        b: u8,
    ) {
        match self.backend {
            RenderBackend::Cpu => fb.draw_rect(x, y, w, h, r, g, b),
            RenderBackend::Loom => {
                // TODO: dispatch ui_rect.spv
                fb.draw_rect(x, y, w, h, r, g, b); // fallback
            }
        }
    }

    /// Draw an image. On GPU backend, dispatches ui_image_blit.spv.
    pub fn draw_image(
        &self,
        fb: &mut Framebuffer,
        x: i32,
        y: i32,
        img_w: u32,
        img_h: u32,
        pixels: &[u8],
    ) {
        match self.backend {
            RenderBackend::Cpu => fb.draw_image(x, y, img_w, img_h, pixels),
            RenderBackend::Loom => {
                // TODO: upload pixels to GPU heap, dispatch ui_image_blit.spv
                fb.draw_image(x, y, img_w, img_h, pixels); // fallback
            }
        }
    }

    /// Blend a pixel with alpha coverage. On GPU backend, dispatches ui_blend.spv.
    pub fn blend_pixel(
        &self,
        fb: &mut Framebuffer,
        x: i32,
        y: i32,
        r: u8,
        g: u8,
        b: u8,
        coverage: u8,
    ) {
        match self.backend {
            RenderBackend::Cpu => fb.blend_pixel(x, y, r, g, b, coverage),
            RenderBackend::Loom => {
                // TODO: dispatch ui_blend.spv
                fb.blend_pixel(x, y, r, g, b, coverage); // fallback
            }
        }
    }

    /// Draw a horizontal line. On GPU backend, dispatches ui_hline.spv.
    pub fn draw_hline(
        &self,
        fb: &mut Framebuffer,
        x: i32,
        y: i32,
        w: u32,
        r: u8,
        g: u8,
        b: u8,
    ) {
        match self.backend {
            RenderBackend::Cpu => fb.draw_hline(x, y, w, r, g, b),
            RenderBackend::Loom => {
                // TODO: dispatch ui_hline.spv
                fb.draw_hline(x, y, w, r, g, b); // fallback
            }
        }
    }

    /// Draw a vertical line. On GPU backend, dispatches ui_vline.spv.
    pub fn draw_vline(
        &self,
        fb: &mut Framebuffer,
        x: i32,
        y: i32,
        h: u32,
        r: u8,
        g: u8,
        b: u8,
    ) {
        match self.backend {
            RenderBackend::Cpu => fb.draw_vline(x, y, h, r, g, b),
            RenderBackend::Loom => {
                // TODO: dispatch ui_vline.spv
                fb.draw_vline(x, y, h, r, g, b); // fallback
            }
        }
    }

    /// Check if Loom GPU rendering is available and enabled.
    pub fn is_gpu_active(&self) -> bool {
        self.backend == RenderBackend::Loom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loom_renderer_creation() {
        let lr = LoomRenderer::new();
        assert_eq!(lr.backend, RenderBackend::Cpu);
        assert!(!lr.is_gpu_active());
    }

    #[test]
    fn test_loom_cpu_clear() {
        let lr = LoomRenderer::new();
        let mut fb = Framebuffer::new(4, 4);
        lr.clear(&mut fb, 255, 0, 0);
        // Check pixel at (0,0)
        assert_eq!(fb.pixels[0], 255); // R
        assert_eq!(fb.pixels[1], 0); // G
        assert_eq!(fb.pixels[2], 0); // B
        assert_eq!(fb.pixels[3], 255); // A
    }

    #[test]
    fn test_loom_cpu_draw_rect() {
        let lr = LoomRenderer::new();
        let mut fb = Framebuffer::new(10, 10);
        fb.clear(0, 0, 0);
        lr.draw_rect(&mut fb, 1, 1, 2, 2, 0, 255, 0);
        let idx = ((1 * 10 + 1) * 4) as usize;
        assert_eq!(fb.pixels[idx], 0); // R
        assert_eq!(fb.pixels[idx + 1], 255); // G channel
        assert_eq!(fb.pixels[idx + 2], 0); // B
    }

    #[test]
    fn test_loom_cpu_draw_image() {
        let lr = LoomRenderer::new();
        let mut fb = Framebuffer::new(10, 10);
        fb.clear(0, 0, 0);
        // 2x2 red image (RGBA)
        let pixels = vec![255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255];
        lr.draw_image(&mut fb, 1, 1, 2, 2, &pixels);
        let idx = ((1 * 10 + 1) * 4) as usize;
        assert_eq!(fb.pixels[idx], 255); // R
        assert_eq!(fb.pixels[idx + 1], 0); // G
        assert_eq!(fb.pixels[idx + 2], 0); // B
    }

    #[test]
    fn test_loom_cpu_blend_pixel() {
        let lr = LoomRenderer::new();
        let mut fb = Framebuffer::new(4, 4);
        fb.clear(0, 0, 0);
        lr.blend_pixel(&mut fb, 1, 1, 255, 255, 255, 128);
        let idx = (1 * 4 + 1) * 4;
        // 50% white on black -> roughly 128
        assert!(fb.pixels[idx] > 120 && fb.pixels[idx] < 135);
    }

    #[test]
    fn test_loom_cpu_draw_hline() {
        let lr = LoomRenderer::new();
        let mut fb = Framebuffer::new(10, 10);
        fb.clear(0, 0, 0);
        lr.draw_hline(&mut fb, 0, 2, 5, 255, 128, 0);
        // Check pixel at (2, 2) -- should be orange
        let idx = ((2 * 10 + 2) * 4) as usize;
        assert_eq!(fb.pixels[idx], 255); // R
        assert_eq!(fb.pixels[idx + 1], 128); // G
    }

    #[test]
    fn test_loom_cpu_draw_vline() {
        let lr = LoomRenderer::new();
        let mut fb = Framebuffer::new(10, 10);
        fb.clear(0, 0, 0);
        lr.draw_vline(&mut fb, 3, 0, 5, 0, 0, 255);
        // Check pixel at (3, 2) -- should be blue
        let idx = ((2 * 10 + 3) * 4) as usize;
        assert_eq!(fb.pixels[idx], 0); // R
        assert_eq!(fb.pixels[idx + 1], 0); // G
        assert_eq!(fb.pixels[idx + 2], 255); // B
    }

    #[test]
    fn test_backend_enum() {
        assert_ne!(RenderBackend::Cpu, RenderBackend::Loom);
        // Debug trait
        let s = format!("{:?}", RenderBackend::Cpu);
        assert_eq!(s, "Cpu");
    }
}
