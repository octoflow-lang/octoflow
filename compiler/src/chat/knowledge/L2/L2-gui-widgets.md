# widgets (L2)
gui/widgets — 16 widget types, all return widget ID (int)

## Functions
gui_panel(x, y, w, h: int) → int — container
gui_label(x, y: int, text: string) → int
gui_button(x, y, w, h: int, text: string) → int
gui_checkbox(x, y: int, text: string) → int
gui_slider(x, y, w: int, min, max: float) → int
gui_textinput(x, y, w: int) → int
gui_progress(x, y, w: int, val: float) → int
gui_radio(x, y: int, text: string, group: int) → int
gui_separator(x, y, w: int) → int
gui_listbox(x, y, w, h: int, items: array) → int
gui_spinbox(x, y: int, min, max: float) → int
gui_dropdown(x, y, w: int, items: array) → int
gui_tabs(x, y, w: int, labels: array) → int
gui_treeview(x, y, w, h: int, data: array) → int
gui_tooltip(id: int, text: string) → float
gui_clicked(id: int) → int — 1 if clicked this frame
gui_checked(id: int) → int — checkbox/radio state
gui_slider_value(id: int) → float
gui_get_text(id: int) → string
