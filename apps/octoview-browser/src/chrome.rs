use crate::framebuffer::Framebuffer;
use crate::text::{FontStyle, TextRenderer};

// --- OctoView Dark Theme ---
pub const BG_COLOR: [u8; 3] = [18, 18, 30];          // #12121e page background
pub const TOOLBAR_BG: [u8; 3] = [30, 30, 46];         // #1e1e2e
pub const URL_BAR_BG: [u8; 3] = [45, 45, 65];         // #2d2d41
pub const URL_BAR_FOCUSED_BG: [u8; 3] = [55, 55, 80]; // #373750
pub const STATUS_BG: [u8; 3] = [24, 24, 38];          // #181826
pub const TEXT_COLOR: [u8; 3] = [205, 214, 244];       // #cdd6f4
pub const DIM_TEXT: [u8; 3] = [130, 135, 160];         // #8287a0
pub const BUTTON_BG: [u8; 3] = [49, 50, 68];          // #313244
pub const BUTTON_DISABLED: [u8; 3] = [35, 35, 50];    // #232332
pub const SEPARATOR: [u8; 3] = [69, 71, 90];          // #45475a

// Find bar constants
pub const FIND_BAR_HEIGHT: f32 = 28.0;
pub const FIND_BAR_BG: [u8; 3] = [35, 35, 50];

// Layout constants
pub const TOOLBAR_HEIGHT: f32 = 40.0;
pub const TAB_BAR_HEIGHT: f32 = 24.0;
pub const STATUS_HEIGHT: f32 = 24.0;
pub const BUTTON_SIZE: f32 = 28.0;
pub const BUTTON_MARGIN: f32 = 4.0;
pub const URL_BAR_MARGIN: f32 = 8.0;

// Tab bar constants
const TAB_WIDTH: f32 = 160.0;
const TAB_PADDING: f32 = 8.0;
const TAB_CLOSE_SIZE: f32 = 14.0;
const TAB_NEW_WIDTH: f32 = 28.0;
const TAB_ACTIVE_BG: [u8; 3] = [49, 50, 68];         // #313244 brighter
const TAB_INACTIVE_BG: [u8; 3] = [30, 30, 46];        // #1e1e2e same as toolbar
const TAB_CLOSE_HOVER: [u8; 3] = [69, 71, 90];        // #45475a

#[derive(Clone, Copy, PartialEq)]
pub enum BrowseMode {
    Web,
    #[allow(dead_code)]
    Flow,
}

pub struct BrowserChrome {
    pub url_text: String,
    pub url_focused: bool,
    pub url_cursor: usize,
    pub url_selected_all: bool,
    pub status_text: String,
    pub mode: BrowseMode,
    pub can_back: bool,
    pub can_forward: bool,
}

impl BrowserChrome {
    pub fn new() -> Self {
        Self {
            url_text: String::new(),
            url_focused: false,
            url_cursor: 0,
            url_selected_all: false,
            status_text: "Ready".to_string(),
            mode: BrowseMode::Web,
            can_back: false,
            can_forward: false,
        }
    }

    pub fn set_url(&mut self, url: &str) {
        self.url_text = url.to_string();
        self.url_cursor = self.url_text.len();
    }

    pub fn set_status(&mut self, node_count: usize, mode: BrowseMode, render_ms: u128) {
        let mode_str = match mode {
            BrowseMode::Web => "Web",
            BrowseMode::Flow => "Flow",
        };
        self.status_text = format!("{} nodes | {} Mode | {}ms", node_count, mode_str, render_ms);
        self.mode = mode;
    }

    /// Insert a character at cursor position.
    /// If all text is selected, clear it first (browser-style select-all replace).
    pub fn url_insert(&mut self, ch: char) {
        if self.url_selected_all {
            self.url_text.clear();
            self.url_cursor = 0;
            self.url_selected_all = false;
        }
        if self.url_cursor <= self.url_text.len() {
            self.url_text.insert(self.url_cursor, ch);
            self.url_cursor += ch.len_utf8();
        }
    }

    /// Delete character before cursor (backspace).
    pub fn url_backspace(&mut self) {
        if self.url_selected_all {
            self.url_text.clear();
            self.url_cursor = 0;
            self.url_selected_all = false;
            return;
        }
        if self.url_cursor > 0 {
            // Find the previous char boundary
            let prev = self.url_text[..self.url_cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.url_text.remove(prev);
            self.url_cursor = prev;
        }
    }

    /// Delete character at cursor (delete key).
    pub fn url_delete(&mut self) {
        if self.url_cursor < self.url_text.len() {
            self.url_text.remove(self.url_cursor);
        }
    }

    pub fn url_move_left(&mut self) {
        self.url_selected_all = false;
        if self.url_cursor > 0 {
            self.url_cursor = self.url_text[..self.url_cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    pub fn url_move_right(&mut self) {
        self.url_selected_all = false;
        if self.url_cursor < self.url_text.len() {
            self.url_cursor = self.url_text[self.url_cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.url_cursor + i)
                .unwrap_or(self.url_text.len());
        }
    }

    pub fn url_home(&mut self) {
        self.url_selected_all = false;
        self.url_cursor = 0;
    }

    pub fn url_end(&mut self) {
        self.url_selected_all = false;
        self.url_cursor = self.url_text.len();
    }

    pub fn focus_url(&mut self) {
        self.url_focused = true;
        self.url_cursor = self.url_text.len();
        self.url_selected_all = true;
    }

    /// Select all text in URL bar (Ctrl+A when focused).
    pub fn url_select_all(&mut self) {
        self.url_cursor = self.url_text.len();
    }
}

/// Render the toolbar at the top of the window.
pub fn render_toolbar(
    chrome: &BrowserChrome,
    fb: &mut Framebuffer,
    text: &mut TextRenderer,
    width: u32,
) {
    let h = TOOLBAR_HEIGHT as u32;

    // Toolbar background
    fb.draw_rect(0, 0, width, h, TOOLBAR_BG[0], TOOLBAR_BG[1], TOOLBAR_BG[2]);

    // Bottom separator
    fb.draw_hline(0, h as i32 - 1, width, SEPARATOR[0], SEPARATOR[1], SEPARATOR[2]);

    let mut x = BUTTON_MARGIN as i32 + 4;
    let btn_y = ((TOOLBAR_HEIGHT - BUTTON_SIZE) / 2.0) as i32;
    let btn_s = BUTTON_SIZE as u32;

    // Back button
    let back_color = if chrome.can_back { BUTTON_BG } else { BUTTON_DISABLED };
    fb.draw_rect(x, btn_y, btn_s, btn_s, back_color[0], back_color[1], back_color[2]);
    let arrow_color = if chrome.can_back { TEXT_COLOR } else { DIM_TEXT };
    text.draw_text(fb, "\u{25C0}", x + 7, btn_y + 4, 16.0, FontStyle::Regular,
        arrow_color[0], arrow_color[1], arrow_color[2]);
    x += btn_s as i32 + BUTTON_MARGIN as i32;

    // Forward button
    let fwd_color = if chrome.can_forward { BUTTON_BG } else { BUTTON_DISABLED };
    fb.draw_rect(x, btn_y, btn_s, btn_s, fwd_color[0], fwd_color[1], fwd_color[2]);
    let arrow_color = if chrome.can_forward { TEXT_COLOR } else { DIM_TEXT };
    text.draw_text(fb, "\u{25B6}", x + 7, btn_y + 4, 16.0, FontStyle::Regular,
        arrow_color[0], arrow_color[1], arrow_color[2]);
    x += btn_s as i32 + BUTTON_MARGIN as i32;

    // Reload button
    fb.draw_rect(x, btn_y, btn_s, btn_s, BUTTON_BG[0], BUTTON_BG[1], BUTTON_BG[2]);
    text.draw_text(fb, "\u{21BB}", x + 5, btn_y + 4, 16.0, FontStyle::Regular,
        TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2]);
    x += btn_s as i32 + URL_BAR_MARGIN as i32;

    // URL bar
    let url_x = x;
    let url_w = (width as i32 - url_x - URL_BAR_MARGIN as i32) as u32;
    let url_y = btn_y;
    let url_h = btn_s;
    let url_bg = if chrome.url_focused { URL_BAR_FOCUSED_BG } else { URL_BAR_BG };
    fb.draw_rect(url_x, url_y, url_w, url_h, url_bg[0], url_bg[1], url_bg[2]);

    // URL text
    let text_x = url_x + 8;
    let text_y = url_y + 5;
    let url_display = if chrome.url_text.is_empty() && !chrome.url_focused {
        "Type a URL or press Ctrl+L..."
    } else {
        &chrome.url_text
    };
    let url_color = if chrome.url_text.is_empty() && !chrome.url_focused {
        DIM_TEXT
    } else {
        TEXT_COLOR
    };

    // Selection highlight when all text is selected
    if chrome.url_focused && chrome.url_selected_all && !chrome.url_text.is_empty() {
        let (sel_w, _) = text.measure_text(&chrome.url_text, 14.0, FontStyle::Regular);
        fb.draw_rect(text_x, url_y + 3, sel_w as u32 + 2, url_h - 6, 60, 120, 200);
    }

    text.draw_text(fb, url_display, text_x, text_y, 14.0, FontStyle::Regular,
        url_color[0], url_color[1], url_color[2]);

    // Cursor when focused (not shown when all text is selected)
    if chrome.url_focused && !chrome.url_selected_all {
        let cursor_text = &chrome.url_text[..chrome.url_cursor];
        let (cursor_x_off, _) = text.measure_text(cursor_text, 14.0, FontStyle::Regular);
        let cx = text_x + cursor_x_off as i32;
        fb.draw_vline(cx, url_y + 4, url_h - 8, TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2]);
    }
}

/// Render the status bar at the bottom of the window.
pub fn render_status_bar(
    chrome: &BrowserChrome,
    fb: &mut Framebuffer,
    text: &mut TextRenderer,
    width: u32,
    height: u32,
) {
    let bar_y = height - STATUS_HEIGHT as u32;
    let bar_h = STATUS_HEIGHT as u32;

    // Background
    fb.draw_rect(0, bar_y as i32, width, bar_h, STATUS_BG[0], STATUS_BG[1], STATUS_BG[2]);

    // Top separator
    fb.draw_hline(0, bar_y as i32, width, SEPARATOR[0], SEPARATOR[1], SEPARATOR[2]);

    // Status text
    let text_y = bar_y as i32 + 4;
    text.draw_text(fb, &chrome.status_text, 8, text_y, 12.0, FontStyle::Regular,
        DIM_TEXT[0], DIM_TEXT[1], DIM_TEXT[2]);
}

/// Returns the Y coordinate where page content starts (below toolbar + tab bar + optional find bar).
pub fn content_top(find_visible: bool) -> f32 {
    TOOLBAR_HEIGHT + TAB_BAR_HEIGHT + if find_visible { FIND_BAR_HEIGHT } else { 0.0 }
}

/// Returns the available height for page content.
pub fn content_height(window_height: u32, find_visible: bool) -> f32 {
    window_height as f32 - TOOLBAR_HEIGHT - TAB_BAR_HEIGHT - STATUS_HEIGHT
        - if find_visible { FIND_BAR_HEIGHT } else { 0.0 }
}

/// Render the find bar below the tab bar.
pub fn render_find_bar(
    query: &str,
    cursor: usize,
    match_text: &str,
    fb: &mut Framebuffer,
    text: &mut TextRenderer,
    width: u32,
    y_offset: u32,
) {
    let h = FIND_BAR_HEIGHT as u32;
    let y = y_offset as i32;

    // Background
    fb.draw_rect(0, y, width, h, FIND_BAR_BG[0], FIND_BAR_BG[1], FIND_BAR_BG[2]);

    // Bottom separator
    fb.draw_hline(0, y + h as i32 - 1, width, SEPARATOR[0], SEPARATOR[1], SEPARATOR[2]);

    let text_y = y + 6;
    let mut x = 8i32;

    // "Find:" label
    text.draw_text(
        fb, "Find:", x, text_y, 12.0, FontStyle::Regular,
        DIM_TEXT[0], DIM_TEXT[1], DIM_TEXT[2],
    );
    let (label_w, _) = text.measure_text("Find:", 12.0, FontStyle::Regular);
    x += label_w as i32 + 6;

    // Input field background
    let input_w = 220u32;
    let input_h = 20u32;
    let input_y = y + (h as i32 - input_h as i32) / 2;
    fb.draw_rect(x, input_y, input_w, input_h, URL_BAR_BG[0], URL_BAR_BG[1], URL_BAR_BG[2]);

    // Query text
    let query_x = x + 4;
    let query_y = input_y + 3;
    if !query.is_empty() {
        text.draw_text(
            fb, query, query_x, query_y, 12.0, FontStyle::Regular,
            TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2],
        );
    }

    // Cursor
    let cursor_text = &query[..cursor.min(query.len())];
    let (cursor_x_off, _) = text.measure_text(cursor_text, 12.0, FontStyle::Regular);
    let cx = query_x + cursor_x_off as i32;
    fb.draw_vline(cx, input_y + 3, input_h - 6, TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2]);

    x += input_w as i32 + 8;

    // Match count text
    if !match_text.is_empty() {
        text.draw_text(
            fb, match_text, x, text_y, 12.0, FontStyle::Regular,
            DIM_TEXT[0], DIM_TEXT[1], DIM_TEXT[2],
        );
        let (mw, _) = text.measure_text(match_text, 12.0, FontStyle::Regular);
        x += mw as i32 + 8;
    }

    // Navigation buttons: up arrow, down arrow
    let btn_size = 18u32;
    let btn_y = y + (h as i32 - btn_size as i32) / 2;

    // Up (previous match)
    fb.draw_rect(x, btn_y, btn_size, btn_size, BUTTON_BG[0], BUTTON_BG[1], BUTTON_BG[2]);
    text.draw_text(
        fb, "\u{25B2}", x + 3, btn_y + 1, 11.0, FontStyle::Regular,
        TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2],
    );
    x += btn_size as i32 + 4;

    // Down (next match)
    fb.draw_rect(x, btn_y, btn_size, btn_size, BUTTON_BG[0], BUTTON_BG[1], BUTTON_BG[2]);
    text.draw_text(
        fb, "\u{25BC}", x + 3, btn_y + 1, 11.0, FontStyle::Regular,
        TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2],
    );
    x += btn_size as i32 + 8;

    // Close button
    fb.draw_rect(x, btn_y, btn_size, btn_size, BUTTON_BG[0], BUTTON_BG[1], BUTTON_BG[2]);
    text.draw_text(
        fb, "x", x + 4, btn_y + 1, 11.0, FontStyle::Regular,
        TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2],
    );
}

// --- Hit test regions for toolbar buttons ---

#[allow(dead_code)]
pub struct ButtonRegion {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

pub enum ToolbarHit {
    Back,
    Forward,
    Reload,
    UrlBar,
    None,
}

/// Hit test the toolbar. Returns which element was clicked.
pub fn toolbar_hit_test(mx: f32, my: f32) -> ToolbarHit {
    if my > TOOLBAR_HEIGHT {
        return ToolbarHit::None;
    }

    let btn_y = (TOOLBAR_HEIGHT - BUTTON_SIZE) / 2.0;
    let mut x = BUTTON_MARGIN + 4.0;

    // Back button
    if mx >= x && mx <= x + BUTTON_SIZE && my >= btn_y && my <= btn_y + BUTTON_SIZE {
        return ToolbarHit::Back;
    }
    x += BUTTON_SIZE + BUTTON_MARGIN;

    // Forward button
    if mx >= x && mx <= x + BUTTON_SIZE && my >= btn_y && my <= btn_y + BUTTON_SIZE {
        return ToolbarHit::Forward;
    }
    x += BUTTON_SIZE + BUTTON_MARGIN;

    // Reload button
    if mx >= x && mx <= x + BUTTON_SIZE && my >= btn_y && my <= btn_y + BUTTON_SIZE {
        return ToolbarHit::Reload;
    }
    x += BUTTON_SIZE + URL_BAR_MARGIN;

    // URL bar (everything to the right)
    if mx >= x && my >= btn_y && my <= btn_y + BUTTON_SIZE {
        return ToolbarHit::UrlBar;
    }

    ToolbarHit::None
}

// --- Tab bar ---

pub enum TabBarHit {
    Tab(usize),
    CloseTab(usize),
    NewTab,
    None,
}

/// Render the tab bar below the toolbar.
/// `tabs` is a slice of (title, is_active) pairs.
/// `y_offset` is the Y coordinate where the tab bar starts (typically TOOLBAR_HEIGHT).
pub fn render_tab_bar(
    tabs: &[(String, bool)],
    fb: &mut Framebuffer,
    text: &mut TextRenderer,
    width: u32,
    y_offset: u32,
) {
    let h = TAB_BAR_HEIGHT as u32;
    let y = y_offset as i32;

    // Tab bar background (same as toolbar for visual continuity)
    fb.draw_rect(0, y, width, h, TAB_INACTIVE_BG[0], TAB_INACTIVE_BG[1], TAB_INACTIVE_BG[2]);

    // Bottom separator
    fb.draw_hline(0, y + h as i32 - 1, width, SEPARATOR[0], SEPARATOR[1], SEPARATOR[2]);

    let mut tx = 0i32;
    let tab_h = h;
    let text_y = y + 4;

    for (i, (title, is_active)) in tabs.iter().enumerate() {
        let tab_w = TAB_WIDTH as u32;

        // Tab background
        let bg = if *is_active { TAB_ACTIVE_BG } else { TAB_INACTIVE_BG };
        fb.draw_rect(tx, y, tab_w, tab_h, bg[0], bg[1], bg[2]);

        // Right separator between tabs
        fb.draw_vline(tx + tab_w as i32 - 1, y, tab_h, SEPARATOR[0], SEPARATOR[1], SEPARATOR[2]);

        // Tab title (truncated to fit)
        let max_text_w = TAB_WIDTH - TAB_PADDING * 2.0 - TAB_CLOSE_SIZE - 4.0;
        let display_title = truncate_title(title, max_text_w, text);
        let color = if *is_active { TEXT_COLOR } else { DIM_TEXT };
        text.draw_text(
            fb,
            &display_title,
            tx + TAB_PADDING as i32,
            text_y,
            12.0,
            FontStyle::Regular,
            color[0],
            color[1],
            color[2],
        );

        // Close button "x" on the right side of the tab
        if tabs.len() > 1 {
            let close_x = tx + tab_w as i32 - TAB_CLOSE_SIZE as i32 - 4;
            let close_y = y + (tab_h as i32 - TAB_CLOSE_SIZE as i32) / 2;
            // Draw close button background
            fb.draw_rect(
                close_x,
                close_y,
                TAB_CLOSE_SIZE as u32,
                TAB_CLOSE_SIZE as u32,
                TAB_CLOSE_HOVER[0],
                TAB_CLOSE_HOVER[1],
                TAB_CLOSE_HOVER[2],
            );
            text.draw_text(
                fb,
                "x",
                close_x + 3,
                close_y + 0,
                11.0,
                FontStyle::Regular,
                TEXT_COLOR[0],
                TEXT_COLOR[1],
                TEXT_COLOR[2],
            );
        }

        // Active tab indicator (bright bottom line)
        if *is_active {
            fb.draw_hline(
                tx,
                y + h as i32 - 2,
                tab_w,
                137,
                180,
                250, // #89b4fa blue accent
            );
        }

        tx += tab_w as i32;

        // Don't overflow the window
        if tx as u32 + TAB_NEW_WIDTH as u32 > width {
            break;
        }

        let _ = i;
    }

    // [+] New tab button
    let new_btn_w = TAB_NEW_WIDTH as u32;
    fb.draw_rect(
        tx,
        y,
        new_btn_w,
        tab_h,
        TAB_INACTIVE_BG[0],
        TAB_INACTIVE_BG[1],
        TAB_INACTIVE_BG[2],
    );
    text.draw_text(
        fb,
        "+",
        tx + 9,
        text_y,
        14.0,
        FontStyle::Regular,
        DIM_TEXT[0],
        DIM_TEXT[1],
        DIM_TEXT[2],
    );
}

/// Hit test the tab bar area.
pub fn tab_bar_hit_test(
    mx: f32,
    my: f32,
    tab_count: usize,
    y_offset: f32,
    _viewport_width: f32,
) -> TabBarHit {
    // Check Y bounds
    if my < y_offset || my > y_offset + TAB_BAR_HEIGHT {
        return TabBarHit::None;
    }

    let mut tx = 0.0f32;

    for i in 0..tab_count {
        let tab_right = tx + TAB_WIDTH;

        if mx >= tx && mx < tab_right {
            // Check if it's the close button area
            if tab_count > 1 {
                let close_x = tab_right - TAB_CLOSE_SIZE - 4.0;
                let close_y = y_offset + (TAB_BAR_HEIGHT - TAB_CLOSE_SIZE) / 2.0;
                if mx >= close_x
                    && mx <= close_x + TAB_CLOSE_SIZE
                    && my >= close_y
                    && my <= close_y + TAB_CLOSE_SIZE
                {
                    return TabBarHit::CloseTab(i);
                }
            }
            return TabBarHit::Tab(i);
        }

        tx += TAB_WIDTH;
    }

    // Check [+] new tab button
    if mx >= tx && mx < tx + TAB_NEW_WIDTH {
        return TabBarHit::NewTab;
    }

    TabBarHit::None
}

/// Truncate a title to fit within max_width pixels, adding "..." if needed.
fn truncate_title(title: &str, max_width: f32, text: &mut TextRenderer) -> String {
    let (w, _) = text.measure_text(title, 12.0, FontStyle::Regular);
    if w <= max_width {
        return title.to_string();
    }

    // Binary search for the longest prefix that fits
    let chars: Vec<char> = title.chars().collect();
    let (ellipsis_w, _) = text.measure_text("...", 12.0, FontStyle::Regular);
    let target = max_width - ellipsis_w;

    let mut best = 0;
    for i in 1..=chars.len() {
        let prefix: String = chars[..i].iter().collect();
        let (pw, _) = text.measure_text(&prefix, 12.0, FontStyle::Regular);
        if pw > target {
            break;
        }
        best = i;
    }

    if best == 0 {
        "...".to_string()
    } else {
        let prefix: String = chars[..best].iter().collect();
        format!("{prefix}...")
    }
}
