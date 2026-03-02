# gui_core (L2)
gui/gui_core — Core primitives (rect, border, text, pixel, line, circle)

## Functions
_gui_fill_rect(x, y, w, h, color: int) → float
_gui_draw_border(x, y, w, h, color: int) → float
_gui_draw_char(x, y: int, ch: string, color: int) → float
_gui_draw_text(x, y: int, text: string, color: int) → float
_gui_set_pixel(x, y, color: int) → float
_gui_draw_line(x1, y1, x2, y2, color: int) → float
_gui_draw_circle(cx, cy, r, color: int) → float
_gui_add_widget(type: string, x, y, w, h: int) → int
