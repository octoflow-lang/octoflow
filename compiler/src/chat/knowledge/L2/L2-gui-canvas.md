# canvas (L2)
gui/canvas — Canvas drawing API (line, rect, fill, circle, pixel)

## Functions
gui_canvas(x, y, w, h: int) → int — create canvas widget
gui_canvas_clear(id, color: int) → float
gui_canvas_line(id, x1, y1, x2, y2, color: int) → float
gui_canvas_rect(id, x, y, w, h, color: int) → float — outline
gui_canvas_fill(id, x, y, w, h, color: int) → float — filled
gui_canvas_circle(id, cx, cy, r, color: int) → float
gui_canvas_pixel(id, x, y, color: int) → float
