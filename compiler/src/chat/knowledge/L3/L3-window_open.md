# window_open (L3)

## Working Example
```flow
let win = window_open(640, 480, "OctoFlow Canvas")

let mut frame = 0.0
let mut running = 1.0

for tick in range(0, 10000)
  if running == 0.0
    break
  end

  let alive = window_alive(win)
  if alive == 0.0
    running = 0.0
    break
  end

  let evt = window_poll(win)
  if evt == "escape"
    running = 0.0
    break
  end

  let r = frame * 2.5
  let g = 100.0
  let b = 255.0 - frame * 2.5
  window_draw(win, r, g, b)

  frame = frame + 1.0
end

window_close(win)
print("closed after {frame} frames")
```

## Expected Output
```
closed after 120 frames
```

*(Window opens, fills with shifting color, closes on Escape or window close.)*

## Common Mistakes
- DON'T: `window_open(640, 480)` → DO: `window_open(640, 480, "title")` (title required)
- DON'T: `win.poll()` → DO: `window_poll(win)` (functions, not methods)
- DON'T: `if evt == true` → DO: `if evt == "escape"` (events are strings)

## Edge Cases
- window_draw RGB values are clamped to 0.0-255.0
- window_alive returns 0.0 after user clicks the close button
- window_poll returns "" when no event is pending
