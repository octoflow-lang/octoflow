# render (L2)
terminal/render — Unified terminal image renderer (dispatches to halfblock/kitty/sixel/digits)

## Functions
term_render(r: array, g: array, b: array, w: int, h: int, mode: string) → string
  Render RGB image using selected mode ("halfblock", "kitty", "sixel", "digits")
term_up(n: int) → string
  Return ANSI escape to move cursor up n lines (for animation)
