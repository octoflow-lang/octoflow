# render (L2)
gui/render — GPU-accelerated rendering (fast clear, rect fill, blit, alpha blend)

## Functions
render_clear(color: int) → float
  Clear framebuffer to solid color
render_fill_rect(x: int, y: int, w: int, h: int, color: int) → float
  GPU-accelerated rectangle fill
render_blit(src: array, x: int, y: int, w: int, h: int) → float
  Copy pixel array to framebuffer
render_blend(src: array, x: int, y: int, w: int, h: int, alpha: float) → float
  Alpha-blend pixel array onto framebuffer
