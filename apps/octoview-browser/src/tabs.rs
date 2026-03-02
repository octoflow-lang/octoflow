use crate::page::PageState;

pub struct Tab {
    pub url: String,
    pub title: String,
    pub page: Option<PageState>,
    pub scroll_y: f32,
    pub history: Vec<String>,
    pub history_idx: usize,
}

impl Tab {
    pub fn new() -> Self {
        Self {
            url: "octoview://welcome".to_string(),
            title: "New Tab".to_string(),
            page: None,
            scroll_y: 0.0,
            history: Vec::new(),
            history_idx: 0,
        }
    }

    pub fn can_back(&self) -> bool {
        self.history_idx > 0
    }

    pub fn can_forward(&self) -> bool {
        self.history_idx + 1 < self.history.len()
    }
}

pub struct TabBar {
    pub tabs: Vec<Tab>,
    pub active: usize,
}

impl TabBar {
    pub fn new() -> Self {
        Self {
            tabs: vec![Tab::new()],
            active: 0,
        }
    }

    pub fn active_tab(&self) -> &Tab {
        &self.tabs[self.active]
    }

    pub fn active_tab_mut(&mut self) -> &mut Tab {
        &mut self.tabs[self.active]
    }

    pub fn new_tab(&mut self) {
        self.tabs.push(Tab::new());
        self.active = self.tabs.len() - 1;
    }

    pub fn close_tab(&mut self, idx: usize) {
        if self.tabs.len() <= 1 {
            return;
        }
        self.tabs.remove(idx);
        if self.active >= self.tabs.len() {
            self.active = self.tabs.len() - 1;
        } else if self.active > idx {
            self.active -= 1;
        }
    }

    pub fn switch_to(&mut self, idx: usize) {
        if idx < self.tabs.len() {
            self.active = idx;
        }
    }

    pub fn next_tab(&mut self) {
        self.active = (self.active + 1) % self.tabs.len();
    }

    pub fn prev_tab(&mut self) {
        if self.active == 0 {
            self.active = self.tabs.len() - 1;
        } else {
            self.active -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tab_bar() {
        let tb = TabBar::new();
        assert_eq!(tb.tabs.len(), 1);
        assert_eq!(tb.active, 0);
    }

    #[test]
    fn test_add_and_close_tabs() {
        let mut tb = TabBar::new();
        tb.new_tab();
        tb.new_tab();
        assert_eq!(tb.tabs.len(), 3);
        assert_eq!(tb.active, 2);
        tb.close_tab(1);
        assert_eq!(tb.tabs.len(), 2);
        assert_eq!(tb.active, 1);
    }

    #[test]
    fn test_cannot_close_last_tab() {
        let mut tb = TabBar::new();
        tb.close_tab(0);
        assert_eq!(tb.tabs.len(), 1);
    }

    #[test]
    fn test_tab_cycling() {
        let mut tb = TabBar::new();
        tb.new_tab();
        tb.new_tab();
        tb.switch_to(0);
        tb.next_tab();
        assert_eq!(tb.active, 1);
        tb.switch_to(0);
        tb.prev_tab();
        assert_eq!(tb.active, 2);
    }

    #[test]
    fn test_tab_history() {
        let mut tab = Tab::new();
        assert!(!tab.can_back());
        assert!(!tab.can_forward());
        tab.history.push("url1".to_string());
        tab.history.push("url2".to_string());
        tab.history_idx = 1;
        assert!(tab.can_back());
        assert!(!tab.can_forward());
    }
}
