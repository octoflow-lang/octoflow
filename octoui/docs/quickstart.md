# OctoUI Quickstart

Get a counter app running in 5 minutes.

## Prerequisites

- OctoFlow compiler (`octoflow.exe`) built and on PATH
- Vulkan-capable GPU with drivers installed
- Windows (Win32 presentation layer)

## Step 1: Build Kernels

OctoUI uses GPU compute kernels for rendering. Build them once:

```bash
cd C:\FlowGPU
octoflow run octoui/kernels/emit_ui_clear.flow --allow-read --allow-write
octoflow run octoui/kernels/emit_ui_rect.flow --allow-read --allow-write
octoflow run octoui/kernels/emit_ui_text.flow --allow-read --allow-write
```

This generates `.spv` SPIR-V kernels in `octoui/kernels/`.

## Step 2: Run the Counter

```bash
octoflow run octoui/examples/counter.flow --allow-read --allow-write --allow-ffi --max-iters 1000000
```

You should see a window with:
- "COUNTER" title text
- A number display (starts at 0)
- "+" and "-" buttons

Click the buttons to change the counter value.

## Step 3: Write Your Own

Create a new `.flow` file:

```
use "octoui/engine/tree"
use "octoui/engine/layout"
use "octoui/engine/pipeline"
use "octoui/engine/dirty"
use "octoui/themes/dark"
use "octoui/widgets/core/text"
use "octoui/widgets/layout/column"
use "octoui/platform/desktop/window_win32"
use "octoui/platform/desktop/input"

let _w = ui_window_open(400.0, 200.0, "My App")

let root = ui_column(-1.0, 10.0)
let _sp = ui_tree_set_padding(root, 20.0)
let _sr = ui_tree_set_pos(root, 0.0, 0.0)
let _ss = ui_tree_set_size(root, 400.0, 200.0)

let label = ui_text(root, "HELLO WORLD", UI_COLOR_TEXT)

let _l = ui_layout_update()

let mut running = 1.0
while running == 1.0
  let alive = ui_poll_events()
  if alive == 0.0
    running = 0.0
  end

  if ui_is_dirty() == 1.0
    let _r = ui_pipeline_render()
    let _p = ui_pipeline_present()
    let _cd = ui_clear_dirty()
  else
    let _sl = sleep(16.0)
  end
end

let _c = ui_window_close()
```

## More Examples

```bash
# Hello world (static text)
octoflow run octoui/examples/hello.flow --allow-read --allow-write --allow-ffi

# Dashboard (multi-panel layout with two counters)
octoflow run octoui/examples/dashboard.flow --allow-read --allow-write --allow-ffi --max-iters 1000000

# Timer (countdown with progress bar, keyboard shortcuts)
octoflow run octoui/examples/timer.flow --allow-read --allow-write --allow-ffi --max-iters 1000000

# Form (text input, checkbox, slider)
octoflow run octoui/examples/form.flow --allow-read --allow-write --allow-ffi --max-iters 1000000
```

## How It Works

OctoUI renders entirely on the GPU using compute shader dispatch chains:

1. **Clear** — GPU fills framebuffer with background color
2. **Rectangles** — GPU draws colored rects for each widget
3. **Text** — GPU renders bitmap font glyphs per character
4. **Present** — CPU reads back R/G/B arrays, calls `window_draw`

All GPU work is batched into a single `vkQueueSubmit` per frame.
