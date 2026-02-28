# OctoUI Widget Reference

## Widget Types

| Constant | Value | Description |
|----------|-------|-------------|
| UI_BOX | 1.0 | Colored rectangle |
| UI_TEXT | 2.0 | Text label |
| UI_BUTTON | 3.0 | Clickable button |
| UI_ROW | 4.0 | Horizontal container |
| UI_COLUMN | 5.0 | Vertical container |
| UI_CHECKBOX | 6.0 | Checkbox toggle |
| UI_TEXTINPUT | 7.0 | Single-line text input |
| UI_SLIDER | 8.0 | Draggable slider |
| UI_RADIO | 9.0 | Radio button |
| UI_TOGGLE | 10.0 | Toggle switch |
| UI_TAB | 11.0 | Tab button |
| UI_LISTBOX | 12.0 | Selectable list |
| UI_SPINBOX | 13.0 | Numeric up/down |
| UI_SCROLL | 14.0 | Scroll container |
| UI_TABLE | 15.0 | Data table |
| UI_TREEVIEW | 16.0 | Tree view |
| UI_TEXTAREA | 17.0 | Multi-line text area |

## Core Widgets

### Box — `widgets/core/box.flow`

A simple colored rectangle.

```
use "octoui/widgets/core/box"

let rect = ui_box(parent, 100.0, 50.0, UI_COLOR_PRIMARY)
```

**Parameters:**
- `parent` — Parent widget ID (-1 for root-level)
- `w` — Width in pixels
- `h` — Height in pixels
- `color_idx` — Theme color index (UI_COLOR_*)

### Text — `widgets/core/text.flow`

A text label rendered with the bitmap font.

```
use "octoui/widgets/core/text"

let label = ui_text(parent, "HELLO", UI_COLOR_TEXT)
```

**Parameters:**
- `parent` — Parent widget ID
- `content` — Text string (uppercase recommended, lowercase maps to uppercase)
- `color_idx` — Theme color index for text color

**Auto-sizing:** Width = `len(content) * 10` pixels (at scale 2). Height = 16 pixels.

### Button — `widgets/input/button.flow`

A clickable button with hover and press visual states.

```
use "octoui/widgets/input/button"

let btn = ui_button(parent, 120.0, 40.0, "CLICK ME")

// In event loop:
if ui_button_clicked(btn) == 1.0
  // Handle click
end
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Width in pixels
- `h` — Height in pixels
- `label` — Button text (centered)

**Functions:**
- `ui_button_clicked(id)` — Returns 1.0 if button was clicked this frame
- `ui_button_clear_clicks()` — Clears per-frame click state (called by input system)
- `ui_button_update_visual(id)` — Updates button color based on state

**Click behavior:** Press-to-click (fires on mouse-down). The visual press state and any state updates render in the same frame for instant responsiveness.

**Keyboard:** Space or Enter activates the focused button.

**States:**
- Normal (0) — Surface color
- Hovered (1) — Hover color
- Pressed (2) — Active color

### Label — `widgets/core/label.flow`

A convenience widget: dimmed label text on the left, colored value on the right.

```
use "octoui/widgets/core/label"

let pair = ui_label(parent, "STATUS", "ACTIVE", UI_COLOR_PRIMARY)
let _u = ui_label_set_value(pair, "INACTIVE")
```

**Parameters:**
- `parent` — Parent widget ID
- `label_text` — Dimmed text on the left
- `value_text` — Value text on the right
- `value_color` — Theme color for value text

**Functions:**
- `ui_label_set_value(row_id, new_text)` — Update the value text

### Separator — `widgets/core/separator.flow`

A thin horizontal line for visual grouping between widgets.

```
use "octoui/widgets/core/separator"

let sep = ui_separator(parent, 300.0)  // 300px wide, 2px tall
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Width in pixels

Rendered as a 2px tall rectangle in the border color.

### Progress Bar — `widgets/core/progress.flow`

A horizontal progress bar with track and fill. Bind to a reactive state for automatic updates.

```
use "octoui/widgets/core/progress"

let pbar = ui_progress(parent, 300.0, 12.0)    // 300px wide, 12px tall
let pct_state = ui_state(50.0)                  // 50%
let _bp = ui_bind_progress(pbar, pct_state)     // Bind

// In event loop (after state changes):
let _pu = ui_progress_update()                  // Sync fill width
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Total width in pixels
- `h` — Height in pixels (typically 8-16)

**Functions:**
- `ui_bind_progress(track_id, state_id)` — Bind fill to reactive state (0.0 - 100.0)
- `ui_progress_update()` — Sync all progress bar fills to their bound states

The track renders in border color, the fill in primary color.

## Input Widgets

### Checkbox — `widgets/input/checkbox.flow`

A clickable toggle with label. Creates a row container with a 16x16 box and text.

```
use "octoui/widgets/input/checkbox"

let chk = ui_checkbox(parent, "ENABLE SOUND")

// Check state:
if ui_checkbox_checked(chk) == 1.0
  // Checkbox is checked
end

// Set programmatically:
let _s = ui_checkbox_set(chk, 1.0)  // Check
let _s = ui_checkbox_set(chk, 0.0)  // Uncheck
```

**Parameters:**
- `parent` — Parent widget ID
- `label` — Text shown next to checkbox

**Visual:** Unchecked = border color box. Checked = primary color fill.
Click or Space key (when focused) toggles state automatically via the input system.

### Text Input — `widgets/input/textinput.flow`

A single-line text input with keyboard support. Click to focus, type to enter text.

```
use "octoui/widgets/input/textinput"

let input = ui_textinput(parent, 200.0, 28.0)

// In event loop:
let key = ui_key_down()
let _tk = ui_textinput_process_key(key)

// Get value:
let text = ui_textinput_value(input)
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Width in pixels
- `h` — Height in pixels

**Functions:**
- `ui_textinput_value(id)` — Get current text
- `ui_textinput_set(id, text)` — Set text programmatically
- `ui_textinput_process_key(key)` — Process keyboard input for focused widget
- `ui_textinput_submitted(id)` — Returns 1.0 if Enter was pressed this frame
- `ui_textinput_set_placeholder(id, text)` — Set placeholder text (shown dimmed when empty and unfocused)
- `ui_textinput_set_readonly(id, flag)` — Set readonly mode (1.0 = no editing, no cursor)
- `ui_textinput_set_password(id, flag)` — Set password mode (1.0 = render * instead of characters)

**Focus:** Click the text input to focus it. Only the focused input receives keyboard events. A cursor is displayed at the current position when focused.

**Cursor movement:** Left/Right arrow keys move the cursor within the text. Home jumps to the start, End jumps to the end. Characters are inserted at the cursor position. Backspace deletes the character before the cursor, Delete removes the character after the cursor.

**Enter key:** Press Enter to submit. Check with `ui_textinput_submitted(id)` each frame.

**Placeholder:** Set placeholder text with `ui_textinput_set_placeholder(id, text)`. The placeholder appears dimmed when the input is empty and unfocused. When the user focuses and types, the placeholder disappears.

**Password mode:** Set with `ui_textinput_set_password(id, 1.0)`. Characters render as `*` instead of their actual values. The underlying value is preserved — `ui_textinput_value(id)` returns the real text.

**Readonly mode:** Set with `ui_textinput_set_readonly(id, 1.0)`. The input displays its text but cannot be edited. No cursor is shown. Enter still fires `ui_textinput_submitted`. Useful for displaying pre-filled, non-editable values.

### Text Area — `widgets/input/textarea.flow`

A multi-line text editing area with cursor, scroll, and readonly mode. Essential for editors, log viewers, and code displays.

```
use "octoui/widgets/input/textarea"

let ta = ui_textarea(parent, 400.0, 200.0)

// Set content (lines separated by chr(10)):
let nl = chr(10.0)
let _st = ui_textarea_set_text(ta, "LINE ONE" + nl + "LINE TWO" + nl + "LINE THREE")

// Or append lines:
let _al = ui_textarea_append_line(ta, "NEW LINE")

// In event loop:
let key = ui_key_down()
let _tp = ui_textarea_process_key(key)
let _tc = ui_textarea_process_click(ta, mx, my, clicked)

// Get content:
let text = ui_textarea_get_text(ta)
let line = ui_textarea_get_line(ta, 0.0)
let count = ui_textarea_line_count(ta)
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Width in pixels
- `h` — Height in pixels (determines visible lines: h / 16)

**Functions:**
- `ui_textarea(parent, w, h)` — Create textarea, returns widget ID
- `ui_textarea_set_text(id, text)` — Set content (splits on newline)
- `ui_textarea_append_line(id, text)` — Append a line
- `ui_textarea_get_text(id)` — Get all content (joined with newline)
- `ui_textarea_get_line(id, line_idx)` — Get single line
- `ui_textarea_line_count(id)` — Get number of lines
- `ui_textarea_set_readonly(id, flag)` — 1.0 = readonly (navigation only)
- `ui_textarea_process_key(key)` — Process keyboard for focused textarea
- `ui_textarea_process_click(id, mx, my, clicked)` — Handle click-to-position cursor
- `ui_textarea_scroll_to(id, line)` — Scroll to specific line
- `ui_textarea_scroll_to_bottom(id)` — Auto-scroll to last line
- `ui_textarea_set_cursor(id, line, col)` — Set cursor position

**Keyboard:**

| Key | Action |
|-----|--------|
| Arrow keys | Move cursor (wraps at line boundaries) |
| Home/End | Start/end of line |
| Page Up/Down | Scroll by viewport height |
| Enter | Split line (insert new line) |
| Backspace | Delete before cursor (merges lines at col 0) |
| Delete | Delete at cursor (merges at end of line) |
| Tab | Insert two spaces |
| Characters | Insert at cursor position |

**Readonly mode:** Navigation works (arrows, Page Up/Down) but editing is disabled. Useful for log viewers and code displays.

**Rendering:** Only visible lines are rendered (viewport = floor(h / 16)). Line height is 16px. Cursor renders as a 2px vertical bar at the current position.

### Slider — `widgets/input/slider.flow`

A horizontal slider for numeric input. Click and drag to change value.

```
use "octoui/widgets/input/slider"

let slider = ui_slider(parent, 200.0, 16.0, 0.0, 100.0, 50.0)

// In event loop:
let _sp = ui_slider_process(ui_mouse_x(), ui_mouse_y(), ui_mouse_down())

// Get value:
let val = ui_slider_value(slider)
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Width in pixels
- `h` — Height in pixels
- `min_val` — Minimum value
- `max_val` — Maximum value
- `initial` — Initial value

**Functions:**
- `ui_slider_value(id)` — Get current value
- `ui_slider_set(id, val)` — Set value programmatically
- `ui_slider_process(mx, my, mdown)` — Process mouse drag (call each frame)
- `ui_slider_process_key(key)` — Process arrow key input for focused slider
- `ui_slider_update_all()` — Position all handles after layout

**Keyboard:** Left/Right arrow keys adjust value by 5% of range when slider is focused.

### Radio Button — `widgets/input/radio.flow`

Mutually exclusive selection within a group.

```
use "octoui/widgets/input/radio"

let group = ui_radio_group(parent)
let _r1 = ui_radio(group, "OPTION A")
let _r2 = ui_radio(group, "OPTION B")
let _r3 = ui_radio(group, "OPTION C")

// Get selected index (0-based):
let sel = ui_radio_selected(group)

// Set programmatically:
let _s = ui_radio_set(group, 1.0)  // Select "OPTION B"
```

**Functions:**
- `ui_radio_group(parent)` — Create a radio group (column container)
- `ui_radio(group, label)` — Add an option to the group
- `ui_radio_selected(group)` — Get selected index (0-based)
- `ui_radio_set(group, index)` — Set selection programmatically

**Visual:** Unselected = border color box. Selected = primary color fill. First option is selected by default.

**Keyboard:** Space key selects the focused radio button.

### Dropdown — `widgets/input/dropdown.flow`

A dropdown selector with a trigger button and a hidden option list.

```
use "octoui/widgets/input/dropdown"

let dd = ui_dropdown(parent, 160.0, 28.0)
let _o1 = ui_dropdown_option(dd, "DARK")
let _o2 = ui_dropdown_option(dd, "LIGHT")
let _o3 = ui_dropdown_option(dd, "AUTO")
let _fin = ui_dropdown_finalize(dd)

// In event loop (after ui_process_input):
let _dp = ui_dropdown_process(ui_mouse_x(), ui_mouse_y(), ui_mouse_clicked())

// Get selection:
let sel = ui_dropdown_selected(dd)  // 0-based index
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Width in pixels
- `h` — Height in pixels (trigger and option height)

**Functions:**
- `ui_dropdown(parent, w, h)` — Create dropdown, returns trigger widget ID
- `ui_dropdown_option(trigger, label)` — Add an option (call in sequence, no interleaving)
- `ui_dropdown_finalize(trigger)` — Set trigger text to first option
- `ui_dropdown_selected(trigger)` — Get selected index (0-based)
- `ui_dropdown_set(trigger, index)` — Set selection programmatically
- `ui_dropdown_process(mx, my, clicked)` — Process clicks (call each frame)

**Behavior:** Click trigger to open/close option list. Click option to select. Click outside or press Escape to close. Only one dropdown open at a time.

### Context Menu — `widgets/input/contextmenu.flow`

A right-click popup menu. Register context menus on widgets; right-click opens them at cursor position.

```
use "octoui/widgets/input/contextmenu"

let ctx = ui_contextmenu()
let ci_del = ui_contextmenu_item(ctx, "DELETE")
let ci_ren = ui_contextmenu_item(ctx, "RENAME")
let _reg = ui_contextmenu_register(my_widget, ctx)

// In event loop:
let rclicked = ui_right_clicked()
let lclicked = ui_mouse_clicked()
let _cp = ui_contextmenu_process(mx, my, rclicked, lclicked)

if ui_ctx_item_clicked(ci_del) == 1.0
  // Handle DELETE
end
```

**Functions:**
- `ui_contextmenu()` — Create context menu, returns menu index
- `ui_contextmenu_item(ctx_idx, label)` — Add item, returns item widget ID
- `ui_contextmenu_register(widget_id, ctx_idx)` — Attach menu to a widget
- `ui_contextmenu_process(mx, my, right_clicked, left_clicked)` — Process each frame
- `ui_ctx_item_clicked(item_id)` — Check if item was clicked this frame
- `ui_contextmenu_close_all()` — Close any open context menu

**Right-click detection:**
- `ui_right_down()` — Returns 1.0 if right mouse button is held
- `ui_right_clicked()` — Returns 1.0 if right mouse just clicked this frame

**Behavior:** Right-click a registered widget to open its context menu at cursor position. Left-click an item to select and close. Left-click outside or Escape to close. Mutual exclusion with menubar menus.

### Toggle Switch — `widgets/input/toggle.flow`

An on/off switch with a sliding knob. Visually distinct from checkboxes — track changes color and knob slides left/right.

```
use "octoui/widgets/input/toggle"

let tgl = ui_toggle(parent, "DARK MODE")

// Check state:
if ui_toggle_on(tgl) == 1.0
  // Toggle is ON
end

// Set programmatically:
let _s = ui_toggle_set(tgl, 1.0)  // ON
let _s = ui_toggle_set(tgl, 0.0)  // OFF
```

**Parameters:**
- `parent` — Parent widget ID
- `label` — Text shown next to the toggle switch

**Functions:**
- `ui_toggle(parent, label)` — Create toggle, returns row widget ID
- `ui_toggle_on(id)` — Returns 1.0 if toggle is ON
- `ui_toggle_set(id, val)` — Set state programmatically (0.0 or 1.0)
- `ui_toggle_toggle(track_id)` — Toggle state (called by input system)
- `ui_toggle_update_all()` — Position all knobs after layout (call once after `ui_layout_update`)

**Visual:** OFF = border-color track, knob on left. ON = primary-color track, knob on right. Track is 36x16 pixels, knob is 14x16 pixels.

**Keyboard:** Space key toggles the focused toggle switch.

### Tab Container — `widgets/input/tabs.flow`

A row of clickable tab buttons with switchable content panels. Only the active tab's panel is visible.

```
use "octoui/widgets/input/tabs"

let tabs = ui_tabs(parent, 400.0)
let _t1 = ui_tab(tabs, "GENERAL")
let _t2 = ui_tab(tabs, "ADVANCED")
let _t3 = ui_tab(tabs, "ABOUT")
let _fin = ui_tabs_finalize(tabs)

// Add widgets to each tab's content panel:
let p1 = ui_tab_panel(tabs, 0.0)
let _w1 = ui_text(p1, "GENERAL CONTENT", UI_COLOR_TEXT)

// After layout:
let _tu = ui_tabs_update_all()

// In event loop (after ui_process_input):
let _tp = ui_tabs_process()

// Get selected tab:
let sel = ui_tabs_selected(tabs)  // 0-based
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Total width of the tab container

**Functions:**
- `ui_tabs(parent, w)` — Create tab container, returns container widget ID
- `ui_tab(tabs_id, label)` — Add a tab (returns panel widget ID)
- `ui_tabs_finalize(tabs_id)` — Finalize after adding all tabs
- `ui_tab_panel(tabs_id, index)` — Get content panel ID by index
- `ui_tabs_selected(tabs_id)` — Get selected tab index (0-based)
- `ui_tabs_set(tabs_id, index)` — Set active tab programmatically
- `ui_tabs_process()` — Process tab clicks (call each frame)
- `ui_tabs_update_all()` — Position panels after layout

**Behavior:** Click a tab button to switch panels. Active tab button uses primary color, inactive tabs use surface color. Tab buttons are focusable and can be activated with Space/Enter.

### Listbox — `widgets/input/listbox.flow`

A selectable list of text items with a fixed viewport. Navigate with Up/Down arrow keys or click to select.

```
use "octoui/widgets/input/listbox"

let lb = ui_listbox(parent, 200.0, 5.0)  // 200px wide, 5 visible rows
let _o1 = ui_listbox_add(lb, "ITEM ONE")
let _o2 = ui_listbox_add(lb, "ITEM TWO")
let _o3 = ui_listbox_add(lb, "ITEM THREE")
let _fin = ui_listbox_finalize(lb)

// After layout:
let _lu = ui_listbox_update_all()

// In event loop:
let _lp = ui_listbox_process(key)

// Get selection:
let sel = ui_listbox_selected(lb)  // 0-based index
let text = ui_listbox_selected_text(lb)  // selected item text
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Width in pixels
- `visible_rows` — Number of rows visible at once (viewport height)

**Functions:**
- `ui_listbox(parent, w, visible_rows)` — Create listbox, returns container widget ID
- `ui_listbox_add(lb_id, text)` — Add an item
- `ui_listbox_finalize(lb_id)` — Finalize and display initial items
- `ui_listbox_selected(lb_id)` — Get selected index (0-based)
- `ui_listbox_selected_text(lb_id)` — Get selected item text
- `ui_listbox_set(lb_id, index)` — Set selection programmatically
- `ui_listbox_process(key)` — Process keyboard input for focused listbox
- `ui_listbox_update_all()` — Position display rows after layout

**Keyboard:** Up/Down arrow keys navigate items. Selection auto-scrolls to keep the selected item visible.

**Visual:** Selected item has primary color background. Items show in a fixed viewport; the list scrolls as you navigate past the visible area.

### Spinbox — `widgets/input/spinbox.flow`

A numeric input with [-] and [+] buttons. Configurable min, max, and step.

```
use "octoui/widgets/input/spinbox"

let spin = ui_spinbox(parent, 120.0, 26.0, 0.0, 100.0, 1.0, 50.0)
//                                         min  max   step initial

// In event loop:
let _sp = ui_spinbox_process()
let _sk = ui_spinbox_process_key(key)

// Get value:
let val = ui_spinbox_value(spin)
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Total width (includes +/- buttons)
- `h` — Height
- `min_val` — Minimum value
- `max_val` — Maximum value
- `step` — Increment per click/keypress
- `initial` — Starting value

**Functions:**
- `ui_spinbox(parent, w, h, min, max, step, initial)` — Create spinbox
- `ui_spinbox_value(id)` — Get current value
- `ui_spinbox_set(id, val)` — Set value programmatically
- `ui_spinbox_process()` — Process +/- button clicks (call each frame)
- `ui_spinbox_process_key(key)` — Process Up/Down arrow keys for focused spinbox

**Keyboard:** Up/Down arrow keys adjust value when the spinbox display is focused.

### Tooltip — `widgets/core/tooltip.flow`

Hover-triggered text popups. Register tooltip text for any widget.

```
use "octoui/widgets/core/tooltip"

let _tt = ui_tooltip_register(button_id, "SAVE CHANGES")

// In event loop:
let _tp = ui_tooltip_process(ui_mouse_x(), ui_mouse_y())
```

**Functions:**
- `ui_tooltip_register(widget_id, text)` — Attach tooltip to a widget
- `ui_tooltip_process(mx, my)` — Update tooltip state each frame

**Behavior:** When the mouse hovers over a registered widget for ~0.3 seconds, a floating text popup appears near the cursor. Moving the mouse away hides the tooltip. Tooltips render on top of all other widgets.

## Layout Widgets

### Column — `widgets/layout/column.flow`

Arranges children vertically.

```
use "octoui/widgets/layout/column"

let col = ui_column(parent, 10.0)  // 10px spacing between children
```

### Row — `widgets/layout/row.flow`

Arranges children horizontally.

```
use "octoui/widgets/layout/row"

let row = ui_row(parent, 20.0)  // 20px spacing between children
```

### Scroll — `widgets/layout/scroll.flow`

A vertically-scrolling container that clips children to a fixed viewport. Shows a scrollbar when content overflows.

```
use "octoui/widgets/layout/scroll"

let panel = ui_scroll(parent, 300.0, 200.0, 8.0)
//                            width  height spacing

// Add any widgets as children:
let _b1 = ui_box(panel, 280.0, 60.0, UI_COLOR_SURFACE)
let _b2 = ui_box(panel, 280.0, 60.0, UI_COLOR_PRIMARY)

// After layout:
let _su = ui_scroll_update_all()

// In event loop:
let _sk = ui_scroll_process(key)  // keyboard scrolling

// Programmatic scroll:
let _s = ui_scroll_by(panel, 20.0)   // scroll down 20px
let _s = ui_scroll_to(panel, 0.0)    // scroll to top
```

**Parameters:**
- `parent` — Parent widget ID
- `w` — Viewport width
- `h` — Viewport height
- `spacing` — Vertical spacing between children

**Functions:**
- `ui_scroll(parent, w, h, spacing)` — Create scroll container
- `ui_scroll_by(container, delta)` — Scroll by delta pixels (positive=down)
- `ui_scroll_to(container, offset)` — Scroll to absolute offset
- `ui_scroll_update_all()` — Position children and update scrollbar (call after layout)
- `ui_scroll_process(key)` — Process keyboard input for scroll navigation
- `ui_scroll_get_offset(container)` — Get current scroll offset
- `ui_scroll_content_height(container)` — Get total content height
- `ui_scroll_can_scroll(container)` — Check if content overflows viewport
- `ui_scroll_process_mouse(mx, my, mdown, clicked)` — Process scrollbar drag + track click

**Keyboard:** Up/Down arrows scroll by 20px. PageUp/PageDown scroll by viewport height. Home/End scroll to top/bottom.

**Mouse:** Drag the scrollbar thumb to scroll smoothly. Click on the scrollbar track to jump to that position.

**Scrollbar:** A track+thumb appears on the right edge when content overflows. Thumb size reflects the viewport/content ratio.

**Clipping:** Children partially visible at the viewport boundary are CPU-clipped — their rendered rect is trimmed so they don't overflow the scroll container.

### Section — `widgets/layout/section.flow`

A collapsible section (accordion) with a clickable header. Click the header button to expand/collapse the content body. The header shows "V TITLE" when expanded or "> TITLE" when collapsed.

```
use "octoui/widgets/layout/section"

let body = ui_section(parent, "SETTINGS")
// Add child widgets to the body:
let _chk = ui_checkbox(body, "OPTION A")
let _sld = ui_slider(body, 200.0, 16.0, 0.0, 100.0, 50.0)

// In event loop — processes header clicks:
let _sp = ui_section_process()
```

**Parameters:**
- `parent` — Parent widget ID
- `title` — Section header text

**Returns:** Body column ID — add child widgets to this container.

**API:**
- `ui_section(parent, title)` — Create section, returns body column ID
- `ui_section_process()` — Process header clicks (call each frame)
- `ui_section_is_expanded(body_id)` — Check if expanded
- `ui_section_expand(body_id)` — Programmatically expand
- `ui_section_collapse(body_id)` — Programmatically collapse

### Menubar — `widgets/layout/menubar.flow`

A horizontal menu bar with trigger buttons that open dropdown columns. Supports hover-to-switch: when one menu is open, hovering another trigger instantly opens it.

```
use "octoui/widgets/layout/menubar"

let bar = ui_menubar(root)
let file_m = ui_menu(bar, "FILE")
let mi_new = ui_menu_item(file_m, "NEW")
let mi_open = ui_menu_item(file_m, "OPEN")
let _sep = ui_menu_separator(file_m)
let mi_exit = ui_menu_item(file_m, "EXIT")

// In event loop (after ui_process_input):
let _mp = ui_menubar_process(mx, my, clicked)

// Check item clicks:
if ui_menu_item_clicked(mi_new) == 1.0
  // Handle NEW
end
```

**Functions:**
- `ui_menubar(parent)` — Create menubar row
- `ui_menu(bar_id, label)` — Add a menu trigger, returns menu index
- `ui_menu_item(menu_idx, label)` — Add an item, returns item widget ID
- `ui_menu_separator(menu_idx)` — Add a visual separator
- `ui_menubar_process(mx, my, clicked)` — Process interactions each frame
- `ui_menu_item_clicked(item_id)` — Check if item was clicked this frame
- `ui_menubar_close_all()` — Close any open menu

**Behavior:** Click trigger to open/close. Hover-to-switch when a menu is open. Click item to select and close. Click outside or Escape to close. Only one menu open at a time. Mutual exclusion with context menus.

### Status Bar — `widgets/layout/statusbar.flow`

A horizontal bar for displaying status text segments.

```
use "octoui/widgets/layout/statusbar"

let sbar = ui_statusbar(root, 480.0)
let st = ui_statusbar_text(sbar, "READY")

// Update text:
let _u = ui_statusbar_set_text(st, "FILE SAVED")
```

**Functions:**
- `ui_statusbar(parent, w)` — Create status bar row (22px tall, surface color)
- `ui_statusbar_text(bar_id, text)` — Add text segment, returns text widget ID
- `ui_statusbar_set_text(text_id, new_text)` — Update text content

## Overlay Widgets

### Modal — `widgets/core/modal.flow`

Overlay dialog with title bar and close button. Only one modal can be open at a time. Background widgets are dimmed and input-blocked.

```
use "octoui/widgets/core/modal"

// Create modal (initially hidden)
let dlg = ui_modal(300.0, 200.0, "CONFIRM")
let panel = ui_modal_panel(dlg)

// Add widgets to modal panel:
let msg = ui_text(panel, "ARE YOU SURE?", UI_COLOR_TEXT)
let ok_btn = ui_button(panel, 100.0, 28.0, "OK")

// Open the modal:
let _o = ui_modal_open(dlg)

// In event loop:
let _mp = ui_modal_process()  // handles close button

// Check if closed:
if ui_modal_is_open(dlg) == 0.0
  // Modal was dismissed
end
```

**Parameters:**
- `w` — Dialog width
- `h` — Dialog height
- `title` — Title bar text

**Functions:**
- `ui_modal(w, h, title)` — Create modal, returns modal ID
- `ui_modal_panel(modal_id)` — Get content panel for adding widgets
- `ui_modal_open(modal_id)` — Show the modal (centered on screen)
- `ui_modal_close(modal_id)` — Hide the modal
- `ui_modal_is_open(modal_id)` — Check if modal is open
- `ui_modal_active()` — Get active modal ID (-1 if none)
- `ui_modal_process()` — Process close button clicks (call each frame)
- `ui_modal_container()` — Get modal container widget ID (for pipeline)

**Behavior:** When a modal is open, ESC closes it. All input to background widgets is blocked. The background is dimmed to half brightness. The modal renders on top of everything via the overlay pipeline pass.

## Theme Colors

Defined in `themes/dark.flow` and `themes/light.flow`:

| Constant | Index | Dark RGB | Light RGB | Description |
|----------|-------|----------|-----------|-------------|
| UI_COLOR_BG | 0 | (30, 30, 35) | (240, 240, 245) | Window background |
| UI_COLOR_SURFACE | 1 | (45, 45, 50) | (220, 220, 228) | Widget surface |
| UI_COLOR_PRIMARY | 2 | (90, 130, 240) | (60, 100, 220) | Primary accent |
| UI_COLOR_TEXT | 3 | (220, 222, 228) | (30, 30, 35) | Main text |
| UI_COLOR_TEXT_DIM | 4 | (140, 142, 148) | (100, 102, 108) | Dimmed text |
| UI_COLOR_BORDER | 5 | (80, 82, 88) | (180, 182, 188) | Borders |
| UI_COLOR_HOVER | 6 | (75, 78, 85) | (200, 202, 210) | Hover state |
| UI_COLOR_ACTIVE | 7 | (90, 130, 240) | (60, 100, 220) | Active/pressed |

### Runtime Theme Switching

Switch themes at runtime using the light theme module:

```
use "octoui/themes/light"

// Switch to light theme
let _lt = ui_theme_load_light()
let _ta = ui_theme_apply_all()
let _d = ui_mark_dirty()

// Switch back to dark
let _dt = ui_theme_load_dark()
let _ta2 = ui_theme_apply_all()
let _d2 = ui_mark_dirty()
```

**How it works:** All widgets use `ui_tree_set_color_themed(id, color_idx)` which records the semantic color index. When the theme palette changes, `ui_theme_apply_all()` walks the entire widget tree and re-applies colors from the new palette.

### Notification — `widgets/core/notification.flow`

Pop-up toast notifications at the top-right of the screen.

```
use "octoui/widgets/core/notification"

// Queue a notification (shows for ~3 seconds)
let _n = ui_notify("SAVED", UI_COLOR_PRIMARY)

// In event loop:
let _np = ui_notify_process()
```

**Functions:**
- `ui_notify(text, color_idx)` — Queue a notification with background color
- `ui_notify_process()` — Update notification state each frame

**Behavior:** Notifications appear at the top-right corner, auto-dismiss after ~3 seconds (~180 frames). If multiple notifications are queued, they display one at a time in FIFO order. Notifications render as overlay on top of all other widgets.

## Data Widgets

### Table — `widgets/data/table.flow`

A data table with fixed header row and virtual-list scrollable body. Only visible rows have render widgets — data for all rows is stored in flat arrays.

```
use "octoui/widgets/data/table"

// Create table (body_h=192px viewport, 24px rows = 8 visible)
let tbl = ui_table(parent, 192.0, 24.0)

// Define columns
let _c1 = ui_table_column(tbl, "NAME", 140.0)
let _c2 = ui_table_column(tbl, "AGE", 60.0)
let _c3 = ui_table_column(tbl, "CITY", 100.0)
let _fin = ui_table_finalize(tbl)

// Add data cells (auto-wraps into rows at column count)
let _c = ui_table_cell(tbl, "ALICE")
let _c = ui_table_cell(tbl, "30")
let _c = ui_table_cell(tbl, "NYC")      // row 0 complete
let _c = ui_table_cell(tbl, "BOB")
let _c = ui_table_cell(tbl, "25")
let _c = ui_table_cell(tbl, "LA")       // row 1 complete

// After layout, refresh table positions:
let _l = ui_layout_update()
let _r = ui_table_refresh_all()

// In event loop:
let _tp = ui_table_process(mx, my, clicked)    // mouse clicks
let _tk = ui_table_process_key(key)            // keyboard nav

// Get selection:
let sel = ui_table_selected_row(tbl)   // -1 if none
```

**Functions:**

| Function | Returns | Description |
|----------|---------|-------------|
| `ui_table(parent, body_h, row_h)` | table ID | Create table container |
| `ui_table_column(table, label, width)` | 0.0 | Add a column |
| `ui_table_finalize(table)` | 0.0 | Create display row widgets |
| `ui_table_cell(table, text)` | 0.0 | Add a cell value |
| `ui_table_refresh(table)` | 0.0 | Update display positions |
| `ui_table_refresh_all()` | 0.0 | Refresh all tables |
| `ui_table_selected_row(table)` | row index | Get selected row (-1 = none) |
| `ui_table_process(mx, my, clicked)` | 0.0 | Process mouse clicks |
| `ui_table_process_key(key)` | 0.0 | Process keyboard navigation |

**Features:**
- Fixed header row with column labels
- Alternating row colors (BG/SURFACE striping)
- Row selection via click (click again to deselect)
- Keyboard navigation: Up/Down, Home/End, PageUp/PageDown
- Virtual list: only viewport rows have render widgets
- Auto-scrolling to keep selection visible

### Tree View — `widgets/data/treeview.flow`

A hierarchical tree view with expandable/collapsible nodes. Uses virtual-list pattern — only visible rows have render widgets. Nodes must be added in topological order (parent before children).

```
use "octoui/widgets/data/treeview"

// Create tree view (240px viewport, 20px rows)
let tree = ui_treeview(parent, 240.0, 20.0)

// Build hierarchy
let root = ui_treeview_node(tree, -1.0, "PROJECT")
let src  = ui_treeview_node(tree, root, "SRC")
let _f1  = ui_treeview_node(tree, src, "MAIN.FLOW")
let _f2  = ui_treeview_node(tree, src, "TEST.FLOW")
let docs = ui_treeview_node(tree, root, "DOCS")
let _d1  = ui_treeview_node(tree, docs, "README.MD")
let _fin = ui_treeview_finalize(tree)

// After layout, refresh:
let _l = ui_layout_update()
let _r = ui_treeview_refresh_all()

// In event loop:
let _tp = ui_treeview_process(mx, my, clicked)
let _tk = ui_treeview_process_key(key)

// Get selection:
let sel = ui_treeview_selected(tree)     // absolute node index (-1 if none)
let label = ui_treeview_label(sel)       // node label text
```

**Functions:**

| Function | Returns | Description |
|----------|---------|-------------|
| `ui_treeview(parent, body_h, row_h)` | tree ID | Create tree view container |
| `ui_treeview_node(tree, parent_node, label)` | node index | Add a node (-1.0 = root level) |
| `ui_treeview_finalize(tree)` | 0.0 | Compute depths, create display rows |
| `ui_treeview_refresh(tree)` | 0.0 | Update display positions |
| `ui_treeview_refresh_all()` | 0.0 | Refresh all tree views |
| `ui_treeview_selected(tree)` | node index | Get selected node (-1 = none) |
| `ui_treeview_label(node)` | string | Get node label text |
| `ui_treeview_toggle(tree, node)` | 0.0 | Toggle expand/collapse |
| `ui_treeview_process(mx, my, clicked)` | 0.0 | Process mouse clicks |
| `ui_treeview_process_key(key)` | 0.0 | Process keyboard navigation |

**Features:**
- Depth indentation (16px per level)
- Expand/collapse icons: `+` (collapsed), `-` (expanded), space (leaf)
- Click to select, Space to toggle expand/collapse
- Left arrow: collapse or select parent
- Right arrow: expand or select first child
- Virtual list: only viewport rows have render widgets
- Auto-scrolling to keep selection visible

---

## Reactive State

```
use "octoui/state/reactive"

let counter = ui_state(0.0)                // Create state
ui_bind_text(display_widget, counter)      // Bind to widget
ui_state_set(counter, ui_state_get(counter) + 1.0)  // Update
```

When `ui_state_set` changes a value, all bound widgets automatically update their text and the frame is marked dirty.

## Focus System

Click on any interactive widget to give it keyboard focus. Press **Tab** to cycle focus through all focusable widgets.

```
let focus = ui_get_focus()     // Get focused widget ID (-1 if none)
let _sf = ui_set_focus(id)     // Set focus programmatically
```

**Focusable types:** Button, Checkbox, Text Input, Slider, Radio, Toggle, Listbox, Spinbox, Scroll, Table, TreeView, TextArea.

**Visual feedback:** Focused widgets show a 2px primary-color border. Text inputs also show a blinking cursor when focused.

**Tab navigation:** The Tab key cycles focus forward through all visible focusable widgets, wrapping around at the end.

## Keyboard Input

```
use "octoui/platform/desktop/input"

// In event loop (after ui_poll_events):
if ui_key_pressed("escape") == 1.0
  // Escape key was pressed this frame
end

if ui_key_pressed("space") == 1.0
  // Space bar was pressed this frame
end

let key = ui_key_down()  // Get last key name (" " if none)
```

**Functions:**
- `ui_key_pressed(key)` — Returns 1.0 if key was pressed this frame
- `ui_key_down()` — Returns key name of last key-down event

**Key names:** `"escape"`, `"space"`, `"enter"`, `"left"`, `"right"`, `"tab"`, `"backspace"`, `"a"`-`"z"`, `"0"`-`"9"`, etc.

**Built-in keyboard actions** (handled automatically by `ui_process_input`):

| Key | Widget | Action |
|-----|--------|--------|
| Tab | Any | Cycle focus to next focusable widget |
| Space | Button | Click (fires ui_button_clicked) |
| Space | Checkbox | Toggle checked state |
| Space | Radio | Select this option |
| Enter | Button | Click (fires ui_button_clicked) |
| Enter | TextInput | Submit (fires ui_textinput_submitted) |
| Left/Right | Slider | Adjust value by 5% of range |
| Left/Right | TextInput | Move cursor within text |
| Home/End | TextInput | Jump to start/end of text |
| Delete | TextInput | Delete character after cursor |
| Space | Toggle | Toggle on/off state |
| Escape | Dropdown | Close open dropdown |
| Up/Down | Listbox | Navigate items (auto-scrolls viewport) |
| Up/Down | Spinbox | Increment/decrement value by step |
| Up/Down | Scroll | Scroll by 20px |
| PageUp/Down | Scroll | Scroll by viewport height |
| Home/End | Scroll | Scroll to top/bottom |
| Up/Down | Table | Move row selection |
| PageUp/Down | Table | Jump selection by viewport rows |
| Home/End | Table | Select first/last row |
| Up/Down | TreeView | Navigate visible nodes |
| Left | TreeView | Collapse node or select parent |
| Right | TreeView | Expand node or select first child |
| Space | TreeView | Toggle expand/collapse |
| Home/End | TreeView | Select first/last visible node |
| Escape | Modal | Close active modal dialog |

## Tree Operations

Low-level tree functions (from `engine/tree.flow`):

```
ui_tree_add(type, parent, w, h, text)  // Add widget → returns ID
ui_tree_set_color(id, r, g, b)         // Set RGB color (custom)
ui_tree_set_color_themed(id, color_idx)  // Set color by theme index (tracks for theme switch)
ui_tree_set_text(id, text)             // Update text content
ui_tree_set_pos(id, x, y)             // Set position
ui_tree_set_size(id, w, h)            // Set size
ui_tree_set_spacing(id, spacing)      // Set container spacing
ui_tree_set_padding(id, padding)      // Set container padding
ui_tree_count()                        // Total widget count
ui_tree_is_container(id)              // Check if ROW or COLUMN
ui_tree_set_visible(id, flag)         // Show (1.0) or hide (0.0) widget
ui_tree_set_enabled(id, flag)         // Enable (1.0) or disable (0.0) widget
ui_tree_is_descendant(widget, ancestor) // Check parent chain (up to 4 levels)
```

**Visibility:** Hidden widgets are skipped in rendering, layout sizing, layout positioning, and Tab navigation. Use `ui_tree_set_visible(id, 0.0)` to hide, `ui_tree_set_visible(id, 1.0)` to show.

**Disabled State:** Disabled widgets remain visible but render dimmed and do not accept input or focus. Rect colors blend 60% toward the background color. Text uses the TEXT_DIM color. Tab navigation skips disabled widgets. Focus is cleared if the currently focused widget is disabled. Use `ui_tree_set_enabled(id, 0.0)` to disable, `ui_tree_set_enabled(id, 1.0)` to re-enable.
