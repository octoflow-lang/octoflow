# OctoFlow GUI Guide

OctoFlow includes a native GUI toolkit — pure OctoFlow, zero external dependencies,
software-rendered with GDI text output. Windows only. 16 widget types, 3 layout systems,
5 chart types, canvas drawing, and buffer visualization.

## Getting Started

### Hello World

```flow
use "gui/gui"

gui_init(400, 300, "Hello OctoFlow")
let btn = gui_button(150, 130, 100, 30, "Click Me")

while gui_running() == 1.0
  gui_update()
  if gui_clicked(btn) == 1.0
    gui_set_text(btn, "Clicked!")
  end
end
```

`gui_init(w, h, title)` opens a window, allocates pixel buffers, and initializes the
bitmap font. `gui_update()` polls events, updates widget state, and renders dirty frames
at ~60 FPS (16 ms sleep). `gui_running()` returns `1.0` while the window is open.

### Imports

The facade module `gui/gui` re-exports everything:

```flow
use "gui/gui"       // all widgets, canvas, layout, themes
```

Or import individual modules:

```flow
use "gui/widgets"   // widget creation + queries
use "gui/layout"    // vstack, hstack, grid
use "gui/canvas"    // canvas drawing commands
use "gui/plot"      // charting library
use "gui/theme"     // color presets
use "gui/buffer_view"  // array visualization
```

---

## Widget Reference

All widgets are created with absolute (x, y) positioning or via layout helpers.
Each creation function returns a widget ID used for queries and mutations.

### Basic Widgets

| Widget | Creation | Parameters | Notes |
|--------|----------|------------|-------|
| **Panel** | `gui_panel(x, y, w, h)` | position, size | Container with border |
| **Label** | `gui_label(x, y, text)` | position, text | Auto-sized to text width |
| **Button** | `gui_button(x, y, w, h, text)` | position, size, label | Hover + pressed states |
| **Checkbox** | `gui_checkbox(x, y, text)` | position, label | Toggle on click |
| **Radio** | `gui_radio(x, y, text, group)` | position, label, group ID | One active per group |
| **Separator** | `gui_separator(x, y, w)` | position, width | Horizontal line |

### Input Widgets

| Widget | Creation | Parameters | Notes |
|--------|----------|------------|-------|
| **Text Input** | `gui_textinput(x, y, w)` | position, width | Click to focus, type to edit |
| **Slider** | `gui_slider(x, y, w, min, max, initial)` | position, width, range | Click or drag |
| **Spinbox** | `gui_spinbox(x, y, w, min, max, initial, step)` | position, width, range, step | -/+ buttons |
| **Dropdown** | `gui_dropdown(x, y, w)` | position, width | Add items, select one |
| **Progress** | `gui_progress(x, y, w)` | position, width | 0-100 value display |

### Container Widgets

| Widget | Creation | Parameters | Notes |
|--------|----------|------------|-------|
| **Listbox** | `gui_listbox(x, y, w, h)` | position, size | Scrollable item list |
| **Tabs** | `gui_tabs(x, y, w, h)` | position, size | Tabbed content area |
| **Treeview** | `gui_treeview(x, y, w, h)` | position, size | Hierarchical node tree |
| **Canvas** | `gui_canvas(x, y, w, h)` | position, size | Custom drawing surface |

### Overlay Widgets

| Widget | Creation | Parameters | Notes |
|--------|----------|------------|-------|
| **Tooltip** | `gui_tooltip(target_id, text)` | target widget, text | Shows on hover |

### Widget State Queries

```flow
gui_running()             // 1.0 if window open, 0.0 if closed
gui_clicked(id)           // 1.0 if clicked this frame
gui_checked(id)           // 1.0 if checkbox/radio is checked
gui_slider_value(id)      // current slider value
gui_get_text(id)          // text input content (string)
gui_radio_value(group)    // widget ID of selected radio in group
gui_listbox_selected(id)  // selected item index (-1 if none)
gui_listbox_text(id, idx) // text of item at index
gui_dropdown_selected(id) // selected dropdown index
gui_dropdown_text(id, idx)// text of dropdown item at index
gui_tab_active(id)        // active tab index
gui_tree_selected(id)     // selected treeview node index
gui_tree_text(id, idx)    // text of treeview node
gui_tree_is_expanded(id, idx) // 1.0 if node expanded
gui_spinbox_value(id)     // current spinbox value
```

### Widget State Mutations

```flow
gui_set_text(id, text)        // change label/button text
gui_set_checked(id, val)      // set checkbox/radio state (0.0/1.0)
gui_set_slider(id, val)       // set slider value
gui_set_progress(id, val)     // set progress (0-100)
gui_set_visible(id, val)      // show/hide widget (0.0/1.0)
gui_set_enabled(id, val)      // enable/disable widget (0.0/1.0)
gui_set_spinbox(id, val)      // set spinbox value (clamped to range)
gui_set_dropdown(id, idx)     // set selected dropdown item
gui_set_tab(id, idx)          // set active tab
gui_tree_toggle(id, idx)      // toggle treeview node expand/collapse
```

### List/Dropdown/Tab Item Management

```flow
gui_listbox_add(id, text)     // append item to listbox
gui_dropdown_add(id, text)    // append option to dropdown
gui_tab_add(id, label)        // append tab
gui_tree_add(id, label, parent_node) // add treeview node (-1 for root)
```

### Example: Dropdown Selection

```flow
use "gui/gui"
gui_init(300, 200, "Dropdown")
let dd = gui_dropdown(50, 50, 200)
let _a1 = gui_dropdown_add(dd, "Option A")
let _a2 = gui_dropdown_add(dd, "Option B")
let _a3 = gui_dropdown_add(dd, "Option C")
let lbl = gui_label(50, 120, "Selected: Option A")

while gui_running() == 1.0
  gui_update()
  if gui_clicked(dd) == 1.0
    let idx = gui_dropdown_selected(dd)
    let text = gui_dropdown_text(dd, idx)
    gui_set_text(lbl, "Selected: " + text)
  end
end
```

---

## Layout System

Manual x/y positioning works but gets tedious. The layout module provides three
systems that compute positions automatically.

### Vertical Stack (vstack)

Places widgets top-to-bottom with configurable spacing.

```flow
use "gui/gui"
use "gui/layout"

gui_init(400, 400, "VStack Demo")
let v = vstack(20, 20, 8)                          // x=20, y=20, spacing=8px

let title = vstack_label(v, "Settings")
let name  = vstack_textinput(v, 200)
let vol   = vstack_slider(v, 200, 0, 100, 50)
let dark  = vstack_checkbox(v, "Dark mode")
let save  = vstack_button(v, 120, 34, "Save")

while gui_running() == 1.0
  gui_update()
end
```

**vstack functions:** `vstack_label`, `vstack_button`, `vstack_checkbox`, `vstack_slider`,
`vstack_textinput`, `vstack_progress`, `vstack_separator`, `vstack_radio`, `vstack_listbox`,
`vstack_spinbox`, `vstack_dropdown`, `vstack_tabs`, `vstack_canvas`, `vstack_treeview`.

Utility: `vstack_space(id, amount)` adds blank space. `vstack_cursor_y(id)` returns
the current Y position.

### Horizontal Stack (hstack)

Places widgets left-to-right.

```flow
let h = hstack(20, 300, 10)                        // x=20, y=300, spacing=10px
let b1 = hstack_button(h, 80, 30, "OK")
let b2 = hstack_button(h, 80, 30, "Cancel")
```

**hstack functions:** `hstack_label`, `hstack_button`, `hstack_checkbox`, `hstack_slider`,
`hstack_radio`, `hstack_textinput`, `hstack_spinbox`, `hstack_dropdown`, `hstack_canvas`,
`hstack_treeview`.

Utility: `hstack_space(id, amount)` adds blank space. `hstack_cursor_x(id)` returns
the current X position.

### Grid Layout

Places widgets in rows and columns with uniform cell sizing.

```flow
let g = grid(20, 20, 4, 60, 36, 4)                 // x=20, y=20, 4 cols, 60x36 cells, 4px gap

let b7 = grid_button(g, 0, 0, "7")
let b8 = grid_button(g, 0, 1, "8")
let b9 = grid_button(g, 0, 2, "9")
let bd = grid_button(g, 0, 3, "/")
let b4 = grid_button(g, 1, 0, "4")
let b5 = grid_button(g, 1, 1, "5")
let b6 = grid_button(g, 1, 2, "6")
let bm = grid_button(g, 1, 3, "*")
```

**grid functions:** `grid_button`, `grid_label`, `grid_checkbox`, `grid_slider`,
`grid_canvas`, `grid_textinput`, `grid_dropdown`, `grid_progress`, `grid_spinbox`.

Utility: `grid_cell_x(id, row, col)` and `grid_cell_y(id, row, col)` return
pixel coordinates for manual placement within grid cells.

### Form Layout (label + input pairs)

Combine hstack rows inside a vstack for form-style layout:

```flow
let form = vstack(20, 20, 10)
let r1 = hstack(20, 20, 10)
let _l1 = hstack_label(r1, "Name:")
let name = hstack_textinput(r1, 200)
let _a1 = vstack_space(form, 40)

let r2 = hstack(20, 70, 10)
let _l2 = hstack_label(r2, "Email:")
let email = hstack_textinput(r2, 200)
```

---

## Event Handling

### The Event Loop

Every GUI app follows the same pattern:

```flow
gui_init(800, 600, "My App")
// ... create widgets ...
while gui_running() == 1.0
  gui_update()       // polls events, updates state, renders
  // ... handle events ...
end
```

`gui_update()` handles everything: window events (close, resize), mouse tracking
(position, click, drag), keyboard input, widget state updates, dirty-flag rendering,
and the 16 ms frame sleep.

### Mouse Events

- **Click:** `gui_clicked(id)` returns `1.0` on the frame the widget was clicked
- **Hover:** Widgets automatically change appearance on hover (buttons lighten)
- **Drag:** Sliders support click-and-drag — holding mouse down updates value continuously
- **Mouse state:** `_gs[3]` = mouse X, `_gs[4]` = mouse Y, `_gs[5]` = mouse down,
  `_gs[6]` = mouse clicked (this frame)

### Keyboard Events

- **Text input:** Click a `gui_textinput` to focus it, then type. Backspace deletes.
- **Key state:** `_gs[7]` = key pressed (1.0 this frame), `_gss[0]` = key name string
- **Key names:** `"backspace"`, `"space"`, single character strings (`"a"`, `"1"`, etc.)

```flow
// Custom keyboard handling
if _gs[7] == 1.0
  let key = _gss[0]
  if key == "q"
    // handle Q press
  end
end
```

### Focus

`_gs[8]` holds the focused widget ID (-1 if none). Clicking a text input focuses it
(highlighted border). Clicking elsewhere unfocuses.

---

## Theming

### Built-in Themes

Three themes are available:

```flow
use "gui/theme"
let _t = gui_theme_dark()    // dark background (default)
let _t = gui_theme_light()   // light background
let _t = gui_theme_ocean()   // deep blue tones
```

Call any theme function at startup or at runtime to switch themes instantly.

### Custom Themes

The theme is stored in the `_tc[]` array (30 float values, RGB triplets):

| Index | Color Role | Default (dark) |
|-------|-----------|----------------|
| 0-2 | Background | 30, 30, 35 |
| 3-5 | Panel | 45, 45, 50 |
| 6-8 | Widget | 60, 62, 68 |
| 9-11 | Hover | 75, 78, 85 |
| 12-14 | Active/accent | 90, 130, 240 |
| 15-17 | Text | 220, 222, 228 |
| 18-20 | Text dim | 140, 142, 148 |
| 21-23 | Border | 80, 82, 88 |
| 24-26 | Check mark | 100, 180, 100 |
| 27-29 | Slider fill | 90, 130, 240 |

```flow
// Custom green theme
_tc[12] = 50.0    // accent R
_tc[13] = 200.0   // accent G
_tc[14] = 80.0    // accent B
_gs[9] = 1.0      // mark dirty to re-render
```

---

## Canvas & Drawing

The canvas widget is a custom drawing surface. Drawing commands are buffered and
rendered during `gui_update()`.

### Canvas Functions

```flow
let cvs = gui_canvas(x, y, w, h)                         // create canvas
let _c = gui_canvas_clear(cvs)                            // clear all commands
let _l = gui_canvas_line(cvs, x1, y1, x2, y2, r, g, b)  // draw line
let _r = gui_canvas_rect(cvs, x, y, w, h, r, g, b)      // draw rectangle outline
let _f = gui_canvas_fill(cvs, x, y, w, h, r, g, b)      // filled rectangle
let _c = gui_canvas_circle(cvs, cx, cy, radius, r, g, b) // circle outline
let _f = gui_canvas_fill_circle(cvs, cx, cy, radius, r, g, b) // filled circle
let _p = gui_canvas_pixel(cvs, px, py, r, g, b)          // single pixel
```

All coordinates are **canvas-relative** (0,0 = top-left of the canvas widget).
Colors are RGB floats in 0-255 range.

### Example: Drawing Shapes

```flow
use "gui/gui"
use "gui/canvas"
gui_init(500, 400, "Canvas Demo")
let cvs = gui_canvas(10, 10, 480, 380)

// Red rectangle
let _r = gui_canvas_fill(cvs, 20, 20, 100, 60, 255, 0, 0)
// Blue circle
let _c = gui_canvas_fill_circle(cvs, 250, 100, 40, 0, 100, 255)
// Green diagonal line
let _l = gui_canvas_line(cvs, 0, 0, 480, 380, 0, 255, 0)

while gui_running() == 1.0
  gui_update()
end
```

---

## Charts & Visualization

The `plot` module renders charts on canvas widgets. Five chart types are supported.

### Creating a Chart

```flow
use "gui/gui"
use "gui/canvas"
use "gui/plot"

gui_init(800, 500, "Chart Demo")
let cvs = gui_canvas(10, 10, 780, 480)

// Create plot with data range: x=[0,10], y=[0,50]
let p = plot_create(cvs, 0, 10, 0, 50)
let _pad = plot_set_padding(p, 40, 30)       // left=40px, bottom=30px padding
```

### Chart Types

**Line chart:**
```flow
let x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
let y = [5, 12, 8, 22, 15, 35, 28, 40, 32, 45, 38]
let _s = plot_series_line(p, x, y, 0, 200, 255)       // blue line
```

**Scatter plot:**
```flow
let _s = plot_series_scatter(p, x, y, 255, 100, 0)    // orange dots (3x3)
```

**Bar chart:**
```flow
let _s = plot_series_bar(p, x, y, 100, 180, 100, 0.8) // green bars, width=0.8
```

**Candlestick chart (OHLC):**
```flow
let dates = [1, 2, 3, 4, 5]
let open  = [100, 105, 102, 110, 108]
let high  = [108, 112, 110, 115, 114]
let low   = [98,  101, 99,  107, 105]
let close = [105, 102, 110, 108, 112]
let _s = plot_series_candle(p, dates, open, high, low, close,
  0, 200, 0,     // up color (green)
  200, 0, 0)     // down color (red)
```

### Drawing and Utilities

```flow
let _g = plot_grid(p, 5, 4)       // draw grid: 5 vertical, 4 horizontal lines
let _a = plot_autoscale(p)        // auto-set range from data (5% padding)
let _r = plot_set_range(p, xmin, xmax, ymin, ymax)  // manual range
let _d = plot_draw(p)             // render all series (clears canvas first)
let _c = plot_clear(p)            // remove all series data
let _x = plot_crosshair(p, mx, my) // draw crosshair at position
```

### Live Chart Updates

```flow
// Redraw each frame with new data
while gui_running() == 1.0
  gui_update()
  let _c = plot_clear(p)
  let _s = plot_series_line(p, new_x, new_y, 0, 200, 255)
  let _a = plot_autoscale(p)
  let _g = plot_grid(p, 5, 4)
  let _d = plot_draw(p)
end
```

---

## Buffer Visualization

The `buffer_view` module displays arrays (CPU or GPU-downloaded) as visual
representations on canvas widgets. Useful for debugging GPU compute results.

### Four Visualization Modes

```flow
use "gui/buffer_view"

// Display RGB arrays as a scaled image
let _v = buffer_view_image(cvs, r_arr, g_arr, b_arr, img_w, img_h)

// Display array as blue-to-red heatmap
let _v = buffer_view_heatmap(cvs, arr, data_w, data_h, min_val, max_val)

// Display array as auto-scaled line waveform
let _v = buffer_view_waveform(cvs, arr, r, g, b)

// Display array as value-distribution histogram
let _v = buffer_view_histogram(cvs, arr, num_bins, r, g, b)
```

- **Image:** Renders separate R/G/B arrays as pixels, scaled to fit canvas
- **Heatmap:** Maps values to blue (low) → green (mid) → red (high) gradient
- **Waveform:** Auto-scaled line chart with 5% Y-axis padding
- **Histogram:** Bins values into equal-width bars

---

## API Quick Reference

### Window & Lifecycle

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `gui_init` | (w, h, title) | 1.0/0.0 | Open window, allocate buffers |
| `gui_update` | () | 0.0 | Poll events, update state, render |
| `gui_running` | () | 1.0/0.0 | Window still open? |

### Widget Creation

| Function | Parameters | Returns |
|----------|-----------|---------|
| `gui_panel` | (x, y, w, h) | widget ID |
| `gui_label` | (x, y, text) | widget ID |
| `gui_button` | (x, y, w, h, text) | widget ID |
| `gui_checkbox` | (x, y, text) | widget ID |
| `gui_slider` | (x, y, w, min, max, initial) | widget ID |
| `gui_textinput` | (x, y, w) | widget ID |
| `gui_progress` | (x, y, w) | widget ID |
| `gui_radio` | (x, y, text, group) | widget ID |
| `gui_separator` | (x, y, w) | widget ID |
| `gui_listbox` | (x, y, w, h) | widget ID |
| `gui_spinbox` | (x, y, w, min, max, initial, step) | widget ID |
| `gui_dropdown` | (x, y, w) | widget ID |
| `gui_tabs` | (x, y, w, h) | widget ID |
| `gui_canvas` | (x, y, w, h) | widget ID |
| `gui_treeview` | (x, y, w, h) | widget ID |
| `gui_tooltip` | (target_id, text) | widget ID |

### Widget Queries

| Function | Returns |
|----------|---------|
| `gui_clicked(id)` | 1.0 on click frame |
| `gui_checked(id)` | 1.0 if checked |
| `gui_slider_value(id)` | float value |
| `gui_get_text(id)` | input string |
| `gui_radio_value(group)` | selected widget ID |
| `gui_listbox_selected(id)` | selected index |
| `gui_listbox_text(id, idx)` | item string |
| `gui_dropdown_selected(id)` | selected index |
| `gui_dropdown_text(id, idx)` | item string |
| `gui_tab_active(id)` | active tab index |
| `gui_spinbox_value(id)` | float value |
| `gui_tree_selected(id)` | selected node index |
| `gui_tree_text(id, idx)` | node label |
| `gui_tree_is_expanded(id, idx)` | 1.0 if expanded |

### Widget Mutations

| Function | Parameters |
|----------|-----------|
| `gui_set_text(id, text)` | Change display text |
| `gui_set_checked(id, val)` | Set check state |
| `gui_set_slider(id, val)` | Set slider value |
| `gui_set_progress(id, val)` | Set progress (0-100) |
| `gui_set_visible(id, val)` | Show/hide |
| `gui_set_enabled(id, val)` | Enable/disable |
| `gui_set_spinbox(id, val)` | Set value (clamped) |
| `gui_set_dropdown(id, idx)` | Set selection |
| `gui_set_tab(id, idx)` | Set active tab |
| `gui_tree_toggle(id, idx)` | Toggle expand |
| `gui_listbox_add(id, text)` | Add item |
| `gui_dropdown_add(id, text)` | Add option |
| `gui_tab_add(id, label)` | Add tab |
| `gui_tree_add(id, label, parent)` | Add node |

### Canvas Drawing

| Function | Parameters |
|----------|-----------|
| `gui_canvas_clear(id)` | Clear all commands |
| `gui_canvas_line(id, x1, y1, x2, y2, r, g, b)` | Draw line |
| `gui_canvas_rect(id, x, y, w, h, r, g, b)` | Rectangle outline |
| `gui_canvas_fill(id, x, y, w, h, r, g, b)` | Filled rectangle |
| `gui_canvas_circle(id, cx, cy, radius, r, g, b)` | Circle outline |
| `gui_canvas_fill_circle(id, cx, cy, radius, r, g, b)` | Filled circle |
| `gui_canvas_pixel(id, px, py, r, g, b)` | Single pixel |

### Chart API

| Function | Parameters |
|----------|-----------|
| `plot_create(canvas_id, xmin, xmax, ymin, ymax)` | Create plot |
| `plot_set_padding(id, left, bottom)` | Set margins |
| `plot_set_padding_full(id, left, bottom, top, right)` | Full margins |
| `plot_set_range(id, xmin, xmax, ymin, ymax)` | Set data range |
| `plot_autoscale(id)` | Auto-range from data |
| `plot_series_line(id, x, y, r, g, b)` | Add line series |
| `plot_series_scatter(id, x, y, r, g, b)` | Add scatter series |
| `plot_series_bar(id, x, y, r, g, b, width)` | Add bar series |
| `plot_series_candle(id, x, o, h, l, c, ur, ug, ub, dr, dg, db)` | Add candlestick |
| `plot_grid(id, x_ticks, y_ticks)` | Draw grid lines |
| `plot_crosshair(id, mx, my)` | Draw crosshair |
| `plot_draw(id)` | Render chart |
| `plot_clear(id)` | Remove all series |

### Layout

| Function | Parameters |
|----------|-----------|
| `vstack(x, y, spacing)` | Create vertical stack |
| `hstack(x, y, spacing)` | Create horizontal stack |
| `grid(x, y, cols, cell_w, cell_h, gap)` | Create grid |
| `vstack_*(lay_id, ...)` | Add widget to vstack |
| `hstack_*(lay_id, ...)` | Add widget to hstack |
| `grid_*(grid_id, row, col, ...)` | Add widget to grid |
| `vstack_space(id, amount)` | Add vertical gap |
| `hstack_space(id, amount)` | Add horizontal gap |
| `vstack_cursor_y(id)` | Current Y position |
| `hstack_cursor_x(id)` | Current X position |
| `grid_cell_x(id, row, col)` | Cell X coordinate |
| `grid_cell_y(id, row, col)` | Cell Y coordinate |

### Buffer Visualization

| Function | Parameters |
|----------|-----------|
| `buffer_view_image(cvs, r, g, b, w, h)` | RGB arrays as image |
| `buffer_view_heatmap(cvs, arr, w, h, min, max)` | Array as heatmap |
| `buffer_view_waveform(cvs, arr, r, g, b)` | Array as line chart |
| `buffer_view_histogram(cvs, arr, bins, r, g, b)` | Array as histogram |

### Themes

| Function | Description |
|----------|-------------|
| `gui_theme_dark()` | Dark background (default) |
| `gui_theme_light()` | Light background |
| `gui_theme_ocean()` | Deep blue tones |
