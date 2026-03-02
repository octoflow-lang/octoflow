use crate::framebuffer::Framebuffer;
use crate::text::{FontStyle, TextRenderer};

pub const PANEL_WIDTH_PCT: f32 = 0.25;
const PANEL_BG: [u8; 3] = [24, 24, 38];
const ITEM_HOVER_BG: [u8; 3] = [35, 35, 50];
const TITLE_COLOR: [u8; 3] = [137, 180, 250];
const URL_COLOR: [u8; 3] = [130, 135, 160];
const TEXT_COLOR: [u8; 3] = [205, 214, 244];
const ITEM_HEIGHT: f32 = 36.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SidePanelKind {
    None,
    Bookmarks,
    History,
}

pub struct SidePanelState {
    pub kind: SidePanelKind,
    pub scroll_y: f32,
    pub hover_idx: Option<usize>,
}

impl SidePanelState {
    pub fn new() -> Self {
        Self {
            kind: SidePanelKind::None,
            scroll_y: 0.0,
            hover_idx: None,
        }
    }

    pub fn toggle(&mut self, kind: SidePanelKind) {
        if self.kind == kind {
            self.kind = SidePanelKind::None;
        } else {
            self.kind = kind;
            self.scroll_y = 0.0;
            self.hover_idx = None;
        }
    }

    pub fn is_visible(&self) -> bool {
        self.kind != SidePanelKind::None
    }

    pub fn panel_width(&self, window_width: u32) -> u32 {
        if self.is_visible() {
            (window_width as f32 * PANEL_WIDTH_PCT) as u32
        } else {
            0
        }
    }
}

pub struct PanelItem {
    pub title: String,
    pub url: String,
    #[allow(dead_code)]
    pub subtitle: String,
}

/// Render side panel on the left.
pub fn render_side_panel(
    state: &SidePanelState,
    items: &[PanelItem],
    header: &str,
    fb: &mut Framebuffer,
    text: &mut TextRenderer,
    top_y: u32,
    window_width: u32,
    window_height: u32,
) {
    if !state.is_visible() {
        return;
    }

    let panel_w = state.panel_width(window_width);
    let panel_h = window_height.saturating_sub(top_y);

    // Background
    fb.draw_rect(
        0, top_y as i32, panel_w, panel_h,
        PANEL_BG[0], PANEL_BG[1], PANEL_BG[2],
    );

    // Right border
    fb.draw_vline(
        panel_w as i32 - 1, top_y as i32, panel_h,
        crate::chrome::SEPARATOR[0], crate::chrome::SEPARATOR[1], crate::chrome::SEPARATOR[2],
    );

    // Header
    text.draw_text(
        fb, header, 8, top_y as i32 + 4, 13.0,
        FontStyle::Bold,
        TITLE_COLOR[0], TITLE_COLOR[1], TITLE_COLOR[2],
    );
    fb.draw_hline(
        0, (top_y + 22) as i32, panel_w,
        crate::chrome::SEPARATOR[0], crate::chrome::SEPARATOR[1], crate::chrome::SEPARATOR[2],
    );

    // Items
    let items_top = top_y as f32 + 26.0;
    let mut y = items_top - state.scroll_y;

    for (idx, item) in items.iter().enumerate() {
        let line_y = y as i32;

        if line_y > window_height as i32 {
            break;
        }
        if line_y + ITEM_HEIGHT as i32 > top_y as i32 && line_y < window_height as i32 {
            if state.hover_idx == Some(idx) {
                fb.draw_rect(
                    0, line_y, panel_w.saturating_sub(1), ITEM_HEIGHT as u32,
                    ITEM_HOVER_BG[0], ITEM_HOVER_BG[1], ITEM_HOVER_BG[2],
                );
            }

            text.draw_text(
                fb, &item.title, 8, line_y + 2, 12.0,
                FontStyle::Regular,
                TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2],
            );

            let url_display = if item.url.len() > 40 {
                &item.url[..40]
            } else {
                &item.url
            };
            text.draw_text(
                fb, url_display, 8, line_y + 16, 10.0,
                FontStyle::Regular,
                URL_COLOR[0], URL_COLOR[1], URL_COLOR[2],
            );
        }
        y += ITEM_HEIGHT;
    }
}

/// Hit test: returns index of clicked item, or None.
pub fn side_panel_hit_test(
    state: &SidePanelState,
    mx: f32,
    my: f32,
    item_count: usize,
    top_y: f32,
    window_width: u32,
) -> Option<usize> {
    if !state.is_visible() {
        return None;
    }
    let panel_w = state.panel_width(window_width) as f32;
    if mx > panel_w {
        return None;
    }
    let items_top = top_y + 26.0;
    let relative_y = my - items_top + state.scroll_y;
    if relative_y < 0.0 {
        return None;
    }
    let idx = (relative_y / ITEM_HEIGHT) as usize;
    if idx < item_count {
        Some(idx)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_toggle() {
        let mut state = SidePanelState::new();
        assert!(!state.is_visible());
        state.toggle(SidePanelKind::Bookmarks);
        assert!(state.is_visible());
        assert_eq!(state.kind, SidePanelKind::Bookmarks);
        state.toggle(SidePanelKind::Bookmarks);
        assert!(!state.is_visible());
    }

    #[test]
    fn test_panel_switch() {
        let mut state = SidePanelState::new();
        state.toggle(SidePanelKind::Bookmarks);
        state.toggle(SidePanelKind::History);
        assert_eq!(state.kind, SidePanelKind::History);
    }

    #[test]
    fn test_panel_width() {
        let state = SidePanelState {
            kind: SidePanelKind::Bookmarks,
            scroll_y: 0.0,
            hover_idx: None,
        };
        assert_eq!(state.panel_width(1000), 250);
        let hidden = SidePanelState::new();
        assert_eq!(hidden.panel_width(1000), 0);
    }

    #[test]
    fn test_hit_test_hidden() {
        let state = SidePanelState::new();
        assert_eq!(side_panel_hit_test(&state, 50.0, 100.0, 5, 64.0, 1000), None);
    }

    #[test]
    fn test_hit_test_outside_panel() {
        let state = SidePanelState {
            kind: SidePanelKind::Bookmarks,
            scroll_y: 0.0,
            hover_idx: None,
        };
        // Panel width = 250, click at x=300 should miss
        assert_eq!(side_panel_hit_test(&state, 300.0, 100.0, 5, 64.0, 1000), None);
    }

    #[test]
    fn test_hit_test_item() {
        let state = SidePanelState {
            kind: SidePanelKind::Bookmarks,
            scroll_y: 0.0,
            hover_idx: None,
        };
        // top_y=64, items_top=90 (64+26), first item at y=90..126 (ITEM_HEIGHT=36)
        // Click at y=100 should hit item 0
        assert_eq!(side_panel_hit_test(&state, 50.0, 100.0, 5, 64.0, 1000), Some(0));
        // Click at y=130 should hit item 1 ((130-90)/36 = 1.11 -> 1)
        assert_eq!(side_panel_hit_test(&state, 50.0, 130.0, 5, 64.0, 1000), Some(1));
    }
}
