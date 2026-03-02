/// CPU-side RGBA pixel buffer with drawing primitives.
/// This is the v1 scaffolding — v2 replaces these with Loom compute dispatches.
pub struct Framebuffer {
    pub pixels: Vec<u8>, // RGBA, row-major
    pub width: u32,
    pub height: u32,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            pixels: vec![0u8; (width * height * 4) as usize],
            width,
            height,
        }
    }

    /// Fill entire buffer with a solid color.
    pub fn clear(&mut self, r: u8, g: u8, b: u8) {
        for chunk in self.pixels.chunks_exact_mut(4) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
            chunk[3] = 255;
        }
    }

    /// Draw a filled rectangle. Clips to bounds.
    pub fn draw_rect(&mut self, x: i32, y: i32, w: u32, h: u32, r: u8, g: u8, b: u8) {
        let x0 = x.max(0) as u32;
        let y0 = y.max(0) as u32;
        let x1 = ((x as i64 + w as i64) as u32).min(self.width);
        let y1 = ((y as i64 + h as i64) as u32).min(self.height);

        for py in y0..y1 {
            let row_start = (py * self.width * 4) as usize;
            for px in x0..x1 {
                let idx = row_start + (px * 4) as usize;
                self.pixels[idx] = r;
                self.pixels[idx + 1] = g;
                self.pixels[idx + 2] = b;
                self.pixels[idx + 3] = 255;
            }
        }
    }

    /// Draw a 1px rectangle outline. Clips to bounds.
    #[allow(dead_code)]
    pub fn draw_rect_outline(
        &mut self,
        x: i32,
        y: i32,
        w: u32,
        h: u32,
        r: u8,
        g: u8,
        b: u8,
    ) {
        // Top and bottom edges
        self.draw_hline(x, y, w, r, g, b);
        self.draw_hline(x, y + h as i32 - 1, w, r, g, b);
        // Left and right edges
        self.draw_vline(x, y, h, r, g, b);
        self.draw_vline(x + w as i32 - 1, y, h, r, g, b);
    }

    /// Draw a horizontal line (1px tall). Clips to bounds.
    pub fn draw_hline(&mut self, x: i32, y: i32, w: u32, r: u8, g: u8, b: u8) {
        if y < 0 || y >= self.height as i32 {
            return;
        }
        let x0 = x.max(0) as u32;
        let x1 = ((x as i64 + w as i64) as u32).min(self.width);
        let row_start = (y as u32 * self.width * 4) as usize;
        for px in x0..x1 {
            let idx = row_start + (px * 4) as usize;
            self.pixels[idx] = r;
            self.pixels[idx + 1] = g;
            self.pixels[idx + 2] = b;
            self.pixels[idx + 3] = 255;
        }
    }

    /// Draw a vertical line (1px wide). Clips to bounds.
    pub fn draw_vline(&mut self, x: i32, y: i32, h: u32, r: u8, g: u8, b: u8) {
        if x < 0 || x >= self.width as i32 {
            return;
        }
        let y0 = y.max(0) as u32;
        let y1 = ((y as i64 + h as i64) as u32).min(self.height);
        for py in y0..y1 {
            let idx = ((py * self.width + x as u32) * 4) as usize;
            self.pixels[idx] = r;
            self.pixels[idx + 1] = g;
            self.pixels[idx + 2] = b;
            self.pixels[idx + 3] = 255;
        }
    }

    /// Set a single pixel (no blending). Clips to bounds.
    #[allow(dead_code)]
    pub fn set_pixel(&mut self, x: i32, y: i32, r: u8, g: u8, b: u8, a: u8) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let idx = ((y as u32 * self.width + x as u32) * 4) as usize;
        self.pixels[idx] = r;
        self.pixels[idx + 1] = g;
        self.pixels[idx + 2] = b;
        self.pixels[idx + 3] = a;
    }

    /// Blit RGBA pixels onto the framebuffer. Clips to bounds.
    /// Alpha compositing: opaque pixels overwrite, transparent skip, semi-transparent blend.
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

    /// Blend a pixel using alpha coverage (for text rendering).
    /// `coverage` is 0-255 where 255 = fully opaque foreground.
    pub fn blend_pixel(&mut self, x: i32, y: i32, r: u8, g: u8, b: u8, coverage: u8) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        if coverage == 0 {
            return;
        }
        let idx = ((y as u32 * self.width + x as u32) * 4) as usize;
        if coverage == 255 {
            self.pixels[idx] = r;
            self.pixels[idx + 1] = g;
            self.pixels[idx + 2] = b;
            self.pixels[idx + 3] = 255;
            return;
        }
        // Linear blend: dst = src * alpha + dst * (1 - alpha)
        let a = coverage as u32;
        let inv_a = 255 - a;
        self.pixels[idx] = ((r as u32 * a + self.pixels[idx] as u32 * inv_a) / 255) as u8;
        self.pixels[idx + 1] =
            ((g as u32 * a + self.pixels[idx + 1] as u32 * inv_a) / 255) as u8;
        self.pixels[idx + 2] =
            ((b as u32 * a + self.pixels[idx + 2] as u32 * inv_a) / 255) as u8;
        self.pixels[idx + 3] = 255;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear() {
        let mut fb = Framebuffer::new(4, 4);
        fb.clear(18, 18, 30);
        // Check first pixel
        assert_eq!(fb.pixels[0], 18);
        assert_eq!(fb.pixels[1], 18);
        assert_eq!(fb.pixels[2], 30);
        assert_eq!(fb.pixels[3], 255);
        // Check last pixel
        let last = (4 * 4 - 1) * 4;
        assert_eq!(fb.pixels[last], 18);
    }

    #[test]
    fn test_draw_rect_clipped() {
        let mut fb = Framebuffer::new(10, 10);
        fb.clear(0, 0, 0);
        // Partially off-screen rect
        fb.draw_rect(-2, -2, 5, 5, 255, 0, 0);
        // Pixel (0,0) should be red
        assert_eq!(fb.pixels[0], 255);
        assert_eq!(fb.pixels[1], 0);
        // Pixel (3,0) should still be black (x=-2+5=3, so x=0..3 are red)
        let idx = 3 * 4;
        assert_eq!(fb.pixels[idx], 0);
    }

    #[test]
    fn test_blend_pixel() {
        let mut fb = Framebuffer::new(4, 4);
        fb.clear(0, 0, 0);
        // 50% white on black background
        fb.blend_pixel(1, 1, 255, 255, 255, 128);
        let idx = (1 * 4 + 1) * 4;
        // Should be roughly 128 (128/255 * 255 ≈ 128)
        assert!(fb.pixels[idx] > 120 && fb.pixels[idx] < 135);
    }

    #[test]
    fn test_draw_image() {
        let mut fb = Framebuffer::new(10, 10);
        fb.clear(0, 0, 0);
        // 2x2 red image (RGBA)
        let pixels = vec![255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255];
        fb.draw_image(1, 1, 2, 2, &pixels);
        let idx = ((1 * 10 + 1) * 4) as usize;
        assert_eq!(fb.pixels[idx], 255); // R
        assert_eq!(fb.pixels[idx + 1], 0); // G
        assert_eq!(fb.pixels[idx + 2], 0); // B
    }

    #[test]
    fn test_out_of_bounds_no_panic() {
        let mut fb = Framebuffer::new(4, 4);
        fb.set_pixel(-1, -1, 255, 0, 0, 255);
        fb.set_pixel(100, 100, 255, 0, 0, 255);
        fb.blend_pixel(-5, -5, 255, 255, 255, 128);
        fb.draw_rect(-10, -10, 5, 5, 255, 0, 0);
        fb.draw_hline(-10, 2, 100, 255, 0, 0);
        // No panic = pass
    }
}
