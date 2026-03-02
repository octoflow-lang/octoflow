# window (L2)
gui/window — Window init, event loop, and software renderer

## Functions
gui_init(title: string, w: int, h: int) → float
  Create window with title and dimensions, start event loop
gui_update() → int
  Process events, redraw widgets, return 1 while open
