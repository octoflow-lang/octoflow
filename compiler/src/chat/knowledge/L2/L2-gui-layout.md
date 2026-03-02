# layout (L2)
gui/layout — Layout helpers (vstack, hstack, grid — no manual coordinates)

## Functions
vstack(x, y, spacing: int) → int — begin vertical stack
vstack_label(id: int, text: string) → int
vstack_button(id, w, h: int, text: string) → int
vstack_checkbox(id: int, text: string) → int
vstack_slider(id, w: int, min, max: float) → int
hstack(x, y, spacing: int) → int — begin horizontal stack
hstack_label(id: int, text: string) → int
hstack_button(id, w, h: int, text: string) → int
grid(x, y, cols, spacing: int) → int — begin grid
grid_button(id, w, h: int, text: string) → int
grid_label(id: int, text: string) → int
