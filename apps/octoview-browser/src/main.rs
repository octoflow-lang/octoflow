mod bookmarks;
mod chrome;
mod css;
mod extract;
mod fetch;
mod find;
mod flow_mode;
mod framebuffer;
mod image_loader;
mod inspector;
mod js_engine;
mod layout;
mod loom;
mod ovd;
mod page;
mod pipeline;
mod renderer;
mod selector_match;
mod side_panel;
mod stylesheet;
mod tabs;
mod text;
mod vulkan;

use bookmarks::BookmarkStore;
use chrome::{BrowseMode, BrowserChrome, TabBarHit, ToolbarHit};
use find::FindState;
use framebuffer::Framebuffer;
use inspector::InspectorState;
use image_loader::ImageCache;
use pipeline::RenderPipeline;
use side_panel::{SidePanelKind, SidePanelState};
use tabs::TabBar;
use text::TextRenderer;
use vulkan::VulkanContext;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

struct App {
    window: Option<Window>,
    vk_ctx: Option<VulkanContext>,
    render_pipeline: Option<RenderPipeline>,
    fb: Option<Framebuffer>,
    text_renderer: Option<TextRenderer>,
    image_cache: ImageCache,
    chrome: BrowserChrome,
    tab_bar: TabBar,
    find_state: FindState,
    inspector_state: InspectorState,
    bookmark_store: BookmarkStore,
    side_panel: SidePanelState,
    #[allow(dead_code)] // scaffolding — will replace Framebuffer calls in v2.1+
    loom_renderer: loom::LoomRenderer,
    modifiers: ModifiersState,
    mouse_x: f32,
    mouse_y: f32,
    needs_redraw: bool,
    initial_url: Option<String>,
    /// Receiver for background page loads (non-blocking UI).
    pending_load: Option<std::sync::mpsc::Receiver<Result<page::PageData, String>>>,
    /// Whether we pushed history for the pending load (vs navigate_no_history).
    pending_push_history: bool,
}

impl App {
    fn new(initial_url: Option<String>) -> Self {
        Self {
            window: None,
            vk_ctx: None,
            render_pipeline: None,
            fb: None,
            text_renderer: None,
            image_cache: ImageCache::new(),
            chrome: BrowserChrome::new(),
            tab_bar: TabBar::new(),
            find_state: FindState::new(),
            inspector_state: InspectorState::new(),
            bookmark_store: BookmarkStore::new(),
            side_panel: SidePanelState::new(),
            loom_renderer: loom::LoomRenderer::new(),
            modifiers: ModifiersState::default(),
            mouse_x: 0.0,
            mouse_y: 0.0,
            needs_redraw: true,
            initial_url,
            pending_load: None,
            pending_push_history: true,
        }
    }

    /// Mark the frame as dirty and schedule a redraw.
    fn mark_dirty(&mut self) {
        self.needs_redraw = true;
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn viewport_width(&self) -> f32 {
        self.fb
            .as_ref()
            .map(|fb| self.inspector_state.content_viewport_width(fb.width))
            .unwrap_or(1280.0)
    }

    fn viewport_height(&self) -> f32 {
        self.fb
            .as_ref()
            .map(|fb| chrome::content_height(fb.height, self.find_state.visible))
            .unwrap_or(700.0)
    }

    fn max_scroll(&self) -> f32 {
        let content_h = self
            .tab_bar
            .active_tab()
            .page
            .as_ref()
            .map(|p| p.content_height)
            .unwrap_or(0.0);
        (content_h - self.viewport_height()).max(0.0)
    }

    fn clamp_scroll(&mut self) {
        let max = self.max_scroll();
        let tab = self.tab_bar.active_tab_mut();
        tab.scroll_y = tab.scroll_y.clamp(0.0, max);
    }

    fn navigate(&mut self, url: &str) {
        self.chrome.set_url(url);
        self.chrome.status_text = format!("Loading {}...", url);
        self.mark_dirty();
        self.image_cache.clear();
        self.pending_push_history = true;

        // Spawn background thread for fetch + parse + JS
        let url_owned = url.to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let is_flow = flow_mode::is_flow_url(&url_owned);
            let result = if is_flow {
                page::load_flow_data(&url_owned)
            } else {
                page::load_page_data(&url_owned)
            };
            let _ = tx.send(result);
        });
        self.pending_load = Some(rx);
    }

    /// Check if a background page load has completed; if so, layout and apply.
    fn poll_pending_load(&mut self) {
        let rx = match &self.pending_load {
            Some(rx) => rx,
            None => return,
        };

        let result = match rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return, // still loading
            Err(_) => {
                self.pending_load = None;
                return;
            }
        };
        self.pending_load = None;

        let vw = self.viewport_width();
        let text = match self.text_renderer.as_mut() {
            Some(t) => t,
            None => return,
        };

        match result {
            Ok(data) => {
                let is_flow = data.is_flow;
                let ps = page::layout_page_data(data, vw, text, &mut self.image_cache);

                let title = if ps.title.is_empty() {
                    ps.url.clone()
                } else {
                    ps.title.clone()
                };
                let mode = if is_flow { BrowseMode::Flow } else { BrowseMode::Web };
                self.chrome.set_url(&ps.url);
                self.chrome
                    .set_status(ps.node_count, mode, ps.load_time_ms);

                if let Some(w) = &self.window {
                    w.set_title(&format!("{} - OctoView", title));
                }

                if self.pending_push_history {
                    let tab = self.tab_bar.active_tab_mut();
                    if tab.history.is_empty() || tab.history[tab.history_idx] != ps.url {
                        tab.history.truncate(tab.history_idx + 1);
                        tab.history.push(ps.url.clone());
                        tab.history_idx = tab.history.len() - 1;
                    }
                }

                let tab = self.tab_bar.active_tab_mut();
                tab.url = ps.url.clone();
                tab.title = title.clone();
                tab.page = Some(ps);
                tab.scroll_y = 0.0;

                if self.pending_push_history {
                    self.bookmark_store.add_history(&tab.url.clone(), &title);
                }
            }
            Err(e) => {
                self.chrome.status_text = format!("Error: {e}");
                eprintln!("Navigation failed: {e}");
            }
        }

        let tab = self.tab_bar.active_tab();
        self.chrome.can_back = tab.can_back();
        self.chrome.can_forward = tab.can_forward();
        self.mark_dirty();
    }

    fn go_back(&mut self) {
        let tab = self.tab_bar.active_tab_mut();
        if tab.history_idx > 0 {
            tab.history_idx -= 1;
            let url = tab.history[tab.history_idx].clone();
            // Navigate without pushing to history
            self.navigate_no_history(&url);
        }
    }

    fn go_forward(&mut self) {
        let tab = self.tab_bar.active_tab_mut();
        if tab.history_idx + 1 < tab.history.len() {
            tab.history_idx += 1;
            let url = tab.history[tab.history_idx].clone();
            self.navigate_no_history(&url);
        }
    }

    fn navigate_no_history(&mut self, url: &str) {
        self.chrome.set_url(url);
        self.chrome.status_text = format!("Loading {}...", url);
        self.mark_dirty();
        self.image_cache.clear();
        self.pending_push_history = false;

        let url_owned = url.to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let is_flow = flow_mode::is_flow_url(&url_owned);
            let result = if is_flow {
                page::load_flow_data(&url_owned)
            } else {
                page::load_page_data(&url_owned)
            };
            let _ = tx.send(result);
        });
        self.pending_load = Some(rx);
    }

    fn reload(&mut self) {
        let url = self.tab_bar.active_tab().url.clone();
        if self.tab_bar.active_tab().page.is_some() {
            self.navigate_no_history(&url);
        }
    }

    fn render_full_frame(&mut self) {
        let fb = match self.fb.as_mut() {
            Some(fb) => fb,
            None => return,
        };
        let text = match self.text_renderer.as_mut() {
            Some(t) => t,
            None => return,
        };

        let w = fb.width;
        let h = fb.height;

        // Clear
        fb.clear(
            chrome::BG_COLOR[0],
            chrome::BG_COLOR[1],
            chrome::BG_COLOR[2],
        );

        // Render page content from active tab
        let find_visible = self.find_state.visible;
        let active_tab = &self.tab_bar.tabs[self.tab_bar.active];
        if let Some(page) = &active_tab.page {
            renderer::render_page(
                fb,
                &page.layout,
                text,
                active_tab.scroll_y,
                chrome::content_top(find_visible),
                chrome::content_height(h, find_visible),
            );
        }

        // Render chrome on top
        chrome::render_toolbar(&self.chrome, fb, text, w);

        // Render tab bar
        let tab_info: Vec<(String, bool)> = self
            .tab_bar
            .tabs
            .iter()
            .enumerate()
            .map(|(i, t)| (t.title.clone(), i == self.tab_bar.active))
            .collect();
        chrome::render_tab_bar(&tab_info, fb, text, w, chrome::TOOLBAR_HEIGHT as u32);

        // Render find bar if visible
        if find_visible {
            let match_text = self.find_state.match_count_text();
            chrome::render_find_bar(
                &self.find_state.query,
                self.find_state.cursor,
                &match_text,
                fb,
                text,
                w,
                (chrome::TOOLBAR_HEIGHT + chrome::TAB_BAR_HEIGHT) as u32,
            );
        }

        chrome::render_status_bar(&self.chrome, fb, text, w, h);

        // Render inspector panel (on top of everything except status bar)
        {
            let active_tab = &self.tab_bar.tabs[self.tab_bar.active];
            if let Some(page) = &active_tab.page {
                inspector::render_inspector(
                    &self.inspector_state,
                    &page.doc,
                    &page.layout,
                    fb,
                    text,
                    chrome::content_top(find_visible),
                    w,
                    h,
                );
            }
        }

        // Render side panel (bookmarks or history)
        if self.side_panel.is_visible() {
            let panel_header = match self.side_panel.kind {
                SidePanelKind::Bookmarks => "Bookmarks",
                SidePanelKind::History => "History",
                SidePanelKind::None => "",
            };
            let items: Vec<side_panel::PanelItem> = match self.side_panel.kind {
                SidePanelKind::Bookmarks => {
                    self.bookmark_store.bookmarks.iter().map(|b| {
                        side_panel::PanelItem {
                            title: b.title.clone(),
                            url: b.url.clone(),
                            subtitle: String::new(),
                        }
                    }).collect()
                }
                SidePanelKind::History => {
                    self.bookmark_store.history.iter().rev().map(|h| {
                        side_panel::PanelItem {
                            title: h.title.clone(),
                            url: h.url.clone(),
                            subtitle: String::new(),
                        }
                    }).collect()
                }
                SidePanelKind::None => Vec::new(),
            };
            side_panel::render_side_panel(
                &self.side_panel,
                &items,
                panel_header,
                fb,
                text,
                chrome::content_top(find_visible) as u32,
                w,
                h,
            );
        }

        // Upload to GPU
        if let (Some(ctx), Some(rp)) = (&self.vk_ctx, &self.render_pipeline) {
            if let Err(e) = rp.upload_framebuffer(ctx, &fb.pixels) {
                eprintln!("Upload failed: {e}");
            }
        }

        self.needs_redraw = false;
    }

    /// Handle link hit test on mouse click.
    fn handle_click(&mut self, x: f32, y: f32) {
        // Check toolbar first
        match chrome::toolbar_hit_test(x, y) {
            ToolbarHit::Back => {
                self.go_back();
                return;
            }
            ToolbarHit::Forward => {
                self.go_forward();
                return;
            }
            ToolbarHit::Reload => {
                self.reload();
                return;
            }
            ToolbarHit::UrlBar => {
                self.chrome.focus_url();
                self.mark_dirty();
                return;
            }
            ToolbarHit::None => {}
        }

        // Check tab bar clicks
        let vw = self.viewport_width();
        match chrome::tab_bar_hit_test(
            x,
            y,
            self.tab_bar.tabs.len(),
            chrome::TOOLBAR_HEIGHT,
            vw,
        ) {
            TabBarHit::Tab(idx) => {
                self.tab_bar.switch_to(idx);
                self.sync_chrome_to_active_tab();
                self.mark_dirty();
                return;
            }
            TabBarHit::CloseTab(idx) => {
                self.tab_bar.close_tab(idx);
                self.sync_chrome_to_active_tab();
                self.mark_dirty();
                return;
            }
            TabBarHit::NewTab => {
                self.open_new_tab();
                return;
            }
            TabBarHit::None => {
                // Unfocus URL bar if clicking outside toolbar/tab bar
                if self.chrome.url_focused {
                    self.chrome.url_focused = false;
                    self.mark_dirty();
                }
            }
        }

        // Check inspector hit test (before content area clicks)
        if self.inspector_state.visible {
            let fb_w = self.fb.as_ref().map(|fb| fb.width).unwrap_or(1280);
            let fb_h = self.fb.as_ref().map(|fb| fb.height).unwrap_or(800);
            let top_y = chrome::content_top(self.find_state.visible);
            let tab = &self.tab_bar.tabs[self.tab_bar.active];
            if let Some(page) = &tab.page {
                if let Some(node_idx) = inspector::inspector_hit_test(
                    &self.inspector_state,
                    &page.doc,
                    x,
                    y,
                    top_y,
                    fb_w,
                    fb_h,
                ) {
                    self.inspector_state.selected_node = Some(node_idx);
                    self.mark_dirty();
                    return;
                }
                // If click is in the inspector panel but not on a node, consume it
                let panel_x = fb_w as f32 - self.inspector_state.panel_width(fb_w);
                if x >= panel_x && y >= top_y {
                    return;
                }
            }
        }

        // Check side panel hit test
        if self.side_panel.is_visible() {
            let fb_w = self.fb.as_ref().map(|fb| fb.width).unwrap_or(1280);
            let top_y = chrome::content_top(self.find_state.visible);
            let item_count = match self.side_panel.kind {
                SidePanelKind::Bookmarks => self.bookmark_store.bookmarks.len(),
                SidePanelKind::History => self.bookmark_store.history.len(),
                SidePanelKind::None => 0,
            };
            if let Some(idx) = side_panel::side_panel_hit_test(
                &self.side_panel, x, y, item_count, top_y, fb_w,
            ) {
                let nav = match self.side_panel.kind {
                    SidePanelKind::Bookmarks => {
                        self.bookmark_store.bookmarks.get(idx).map(|b| b.url.clone())
                    }
                    SidePanelKind::History => {
                        // History is displayed in reverse order
                        let rev_idx = self.bookmark_store.history.len().saturating_sub(1 + idx);
                        self.bookmark_store.history.get(rev_idx).map(|h| h.url.clone())
                    }
                    SidePanelKind::None => None,
                };
                if let Some(url) = nav {
                    self.navigate(&url);
                    return;
                }
            }
            // If click is inside the panel area, consume it
            let panel_w = self.side_panel.panel_width(fb_w) as f32;
            if x < panel_w && y >= top_y {
                return;
            }
        }

        // Check for link clicks in content area
        let nav_url = {
            let tab = &self.tab_bar.tabs[self.tab_bar.active];
            if let Some(page) = &tab.page {
                let content_y = y - chrome::content_top(self.find_state.visible) + tab.scroll_y;
                let mut found = None;
                for b in page.layout.iter().rev() {
                    if let layout::LayoutContent::Link(_, ref href) = b.content {
                        let bx = b.x;
                        let by = b.y;
                        if x >= bx
                            && x <= bx + b.width
                            && content_y >= by
                            && content_y <= by + b.height
                        {
                            found = Some(resolve_url(&page.url, href));
                            break;
                        }
                    }
                }
                found
            } else {
                None
            }
        };
        if let Some(url) = nav_url {
            self.navigate(&url);
        }
    }

    /// Sync chrome UI state (URL bar, back/forward buttons) to the active tab.
    fn sync_chrome_to_active_tab(&mut self) {
        let tab = self.tab_bar.active_tab();
        self.chrome.set_url(&tab.url);
        self.chrome.can_back = tab.can_back();
        self.chrome.can_forward = tab.can_forward();
        if let Some(w) = &self.window {
            w.set_title(&format!("{} - OctoView", tab.title));
        }
    }

    /// Open a new tab with the welcome page.
    fn open_new_tab(&mut self) {
        self.tab_bar.new_tab();
        // Load welcome page into the new tab
        let vw = self.viewport_width();
        if let Some(text) = self.text_renderer.as_mut() {
            let ps = page::welcome_page(vw, text, &mut self.image_cache);
            let tab = self.tab_bar.active_tab_mut();
            tab.url = ps.url.clone();
            tab.title = ps.title.clone();
            tab.history.push(ps.url.clone());
            tab.page = Some(ps);
        }
        self.sync_chrome_to_active_tab();
        self.mark_dirty();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("OctoView Browser")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 800));

        let w = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("Failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        let ctx = match VulkanContext::new(&w) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Vulkan init failed: {e}");
                event_loop.exit();
                return;
            }
        };

        let extent = ctx.swapchain_extent;
        eprintln!("Vulkan initialized: {}x{}", extent.width, extent.height);

        let rp = match RenderPipeline::new(&ctx, extent.width, extent.height) {
            Ok(rp) => rp,
            Err(e) => {
                eprintln!("Pipeline init failed: {e}");
                event_loop.exit();
                return;
            }
        };

        let fb = Framebuffer::new(extent.width, extent.height);
        let mut text_renderer = TextRenderer::new();

        // Load initial page into the first tab
        let initial_url = self.initial_url.take();
        if let Some(url) = initial_url {
            let vw = extent.width as f32;
            self.chrome.set_url(&url);
            let is_flow = flow_mode::is_flow_url(&url);
            let result = if is_flow {
                page::load_flow_page(&url, vw, &mut text_renderer, &mut self.image_cache)
            } else {
                page::load_page(&url, vw, &mut text_renderer, &mut self.image_cache)
            };
            match result {
                Ok(ps) => {
                    let mode = if is_flow { BrowseMode::Flow } else { BrowseMode::Web };
                    self.chrome.set_url(&ps.url);
                    self.chrome
                        .set_status(ps.node_count, mode, ps.load_time_ms);
                    let tab = self.tab_bar.active_tab_mut();
                    tab.url = ps.url.clone();
                    tab.title = if ps.title.is_empty() {
                        ps.url.clone()
                    } else {
                        ps.title.clone()
                    };
                    tab.history.push(ps.url.clone());
                    tab.page = Some(ps);
                }
                Err(e) => {
                    self.chrome.status_text = format!("Error: {e}");
                }
            }
        } else {
            // Welcome page
            let ps = page::welcome_page(extent.width as f32, &mut text_renderer, &mut self.image_cache);
            self.chrome.set_url("octoview://welcome");
            self.chrome.status_text = format!("{} nodes | Welcome", ps.node_count);
            let tab = self.tab_bar.active_tab_mut();
            tab.url = ps.url.clone();
            tab.title = ps.title.clone();
            tab.history.push(ps.url.clone());
            tab.page = Some(ps);
        }

        self.vk_ctx = Some(ctx);
        self.render_pipeline = Some(rp);
        self.fb = Some(fb);
        self.text_renderer = Some(text_renderer);
        self.window = Some(w);
        self.mark_dirty();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if let (Some(rp), Some(ctx)) =
                    (self.render_pipeline.as_mut(), self.vk_ctx.as_ref())
                {
                    ctx.wait_idle();
                    rp.destroy(ctx.device());
                }
                self.render_pipeline = None;
                event_loop.exit();
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }

                match &event.logical_key {
                    // Ctrl+F: Toggle find bar
                    Key::Character(c)
                        if c.as_str() == "f" && self.modifiers.control_key() =>
                    {
                        self.find_state.toggle();
                        self.mark_dirty();
                    }

                    // Ctrl+L: Focus URL bar
                    Key::Character(c)
                        if c.as_str() == "l" && self.modifiers.control_key() =>
                    {
                        self.chrome.focus_url();
                        self.mark_dirty();
                    }

                    // Ctrl+D: Bookmark current page
                    Key::Character(c)
                        if c.as_str() == "d" && self.modifiers.control_key() =>
                    {
                        let tab = self.tab_bar.active_tab();
                        let url = tab.url.clone();
                        let title = tab.title.clone();
                        self.bookmark_store.add_bookmark(&url, &title);
                        self.mark_dirty();
                    }

                    // Ctrl+B: Toggle bookmarks panel
                    Key::Character(c)
                        if c.as_str() == "b" && self.modifiers.control_key() =>
                    {
                        self.side_panel.toggle(SidePanelKind::Bookmarks);
                        self.mark_dirty();
                    }

                    // Ctrl+H: Toggle history panel
                    Key::Character(c)
                        if c.as_str() == "h" && self.modifiers.control_key() =>
                    {
                        self.side_panel.toggle(SidePanelKind::History);
                        self.mark_dirty();
                    }

                    // Escape: Close find bar, side panel, or unfocus URL bar
                    Key::Named(NamedKey::Escape) => {
                        if self.find_state.visible {
                            self.find_state.toggle();
                        } else if self.side_panel.is_visible() {
                            self.side_panel.kind = SidePanelKind::None;
                        } else {
                            self.chrome.url_focused = false;
                        }
                        self.mark_dirty();
                    }

                    // Enter when find bar visible: next/prev match
                    Key::Named(NamedKey::Enter) if self.find_state.visible => {
                        if self.modifiers.shift_key() {
                            self.find_state.prev_match();
                        } else {
                            self.find_state.next_match();
                        }
                        // Auto-scroll to active match
                        if let Some(y) = self.find_state.active_y() {
                            let vh = self.viewport_height();
                            self.tab_bar.active_tab_mut().scroll_y = (y - vh / 2.0).max(0.0);
                            self.clamp_scroll();
                        }
                        self.mark_dirty();
                    }

                    // Enter: Navigate when URL bar focused
                    Key::Named(NamedKey::Enter) if self.chrome.url_focused => {
                        let url = self.chrome.url_text.clone();
                        self.chrome.url_focused = false;
                        if !url.is_empty() {
                            self.navigate(&url);
                        }
                    }

                    // F5: Reload
                    Key::Named(NamedKey::F5) => {
                        self.reload();
                    }

                    // F12: Toggle OVD Inspector
                    Key::Named(NamedKey::F12) => {
                        self.inspector_state.toggle();
                        // Re-layout page at the new content width
                        let vw = self.viewport_width();
                        if let Some(text) = self.text_renderer.as_mut() {
                            let tab = self.tab_bar.active_tab_mut();
                            if let Some(page) = tab.page.as_mut() {
                                let boxes = layout::layout_document(
                                    &page.doc, text, vw, &mut self.image_cache,
                                );
                                page.content_height = layout::content_height(&boxes);
                                page.layout = boxes;
                            }
                        }
                        self.clamp_scroll();
                        self.mark_dirty();
                    }

                    // Alt+Left: Back
                    Key::Named(NamedKey::ArrowLeft) if self.modifiers.alt_key() => {
                        self.go_back();
                    }

                    // Alt+Right: Forward
                    Key::Named(NamedKey::ArrowRight) if self.modifiers.alt_key() => {
                        self.go_forward();
                    }

                    // Find bar text input
                    _ if self.find_state.visible => {
                        match &event.logical_key {
                            Key::Named(NamedKey::Backspace) => {
                                self.find_state.backspace();
                                if let Some(page) = &self.tab_bar.active_tab().page {
                                    self.find_state.search(&page.layout);
                                }
                                if let Some(y) = self.find_state.active_y() {
                                    let vh = self.viewport_height();
                                    self.tab_bar.active_tab_mut().scroll_y = (y - vh / 2.0).max(0.0);
                                    self.clamp_scroll();
                                }
                                self.mark_dirty();
                            }
                            Key::Character(c) => {
                                if !self.modifiers.control_key() {
                                    for ch in c.chars() {
                                        if !ch.is_control() {
                                            self.find_state.insert(ch);
                                        }
                                    }
                                    if let Some(page) = &self.tab_bar.active_tab().page {
                                        self.find_state.search(&page.layout);
                                    }
                                    if let Some(y) = self.find_state.active_y() {
                                        let vh = self.viewport_height();
                                        self.tab_bar.active_tab_mut().scroll_y = (y - vh / 2.0).max(0.0);
                                        self.clamp_scroll();
                                    }
                                }
                                self.mark_dirty();
                            }
                            _ => {}
                        }
                    }

                    // URL bar text input
                    _ if self.chrome.url_focused => {
                        match &event.logical_key {
                            Key::Named(NamedKey::Backspace) => {
                                self.chrome.url_backspace();
                                self.mark_dirty();
                            }
                            Key::Named(NamedKey::Delete) => {
                                self.chrome.url_delete();
                                self.mark_dirty();
                            }
                            Key::Named(NamedKey::ArrowLeft) => {
                                self.chrome.url_move_left();
                                self.mark_dirty();
                            }
                            Key::Named(NamedKey::ArrowRight) => {
                                self.chrome.url_move_right();
                                self.mark_dirty();
                            }
                            Key::Named(NamedKey::Home) => {
                                self.chrome.url_home();
                                self.mark_dirty();
                            }
                            Key::Named(NamedKey::End) => {
                                self.chrome.url_end();
                                self.mark_dirty();
                            }
                            Key::Character(c) => {
                                if self.modifiers.control_key() && c.as_str() == "a" {
                                    self.chrome.url_select_all();
                                } else if !self.modifiers.control_key() {
                                    for ch in c.chars() {
                                        if !ch.is_control() {
                                            self.chrome.url_insert(ch);
                                        }
                                    }
                                }
                                self.mark_dirty();
                            }
                            _ => {}
                        }
                    }

                    // Ctrl+T: New tab
                    Key::Character(c)
                        if c.as_str() == "t" && self.modifiers.control_key() =>
                    {
                        self.open_new_tab();
                    }

                    // Ctrl+W: Close current tab
                    Key::Character(c)
                        if c.as_str() == "w" && self.modifiers.control_key() =>
                    {
                        let active = self.tab_bar.active;
                        self.tab_bar.close_tab(active);
                        self.sync_chrome_to_active_tab();
                        self.mark_dirty();
                    }

                    // Ctrl+Tab: Next tab
                    Key::Named(NamedKey::Tab) if self.modifiers.control_key() && !self.modifiers.shift_key() => {
                        self.tab_bar.next_tab();
                        self.sync_chrome_to_active_tab();
                        self.mark_dirty();
                    }

                    // Ctrl+Shift+Tab: Previous tab
                    Key::Named(NamedKey::Tab) if self.modifiers.control_key() && self.modifiers.shift_key() => {
                        self.tab_bar.prev_tab();
                        self.sync_chrome_to_active_tab();
                        self.mark_dirty();
                    }

                    // Page scroll (when URL bar not focused)
                    Key::Named(NamedKey::Space) | Key::Named(NamedKey::PageDown) => {
                        let vh = self.viewport_height();
                        self.tab_bar.active_tab_mut().scroll_y += vh * 0.8;
                        self.clamp_scroll();
                        self.mark_dirty();
                    }
                    Key::Named(NamedKey::PageUp) => {
                        let vh = self.viewport_height();
                        self.tab_bar.active_tab_mut().scroll_y -= vh * 0.8;
                        self.clamp_scroll();
                        self.mark_dirty();
                    }
                    Key::Named(NamedKey::Home) => {
                        self.tab_bar.active_tab_mut().scroll_y = 0.0;
                        self.mark_dirty();
                    }
                    Key::Named(NamedKey::End) => {
                        let ms = self.max_scroll();
                        self.tab_bar.active_tab_mut().scroll_y = ms;
                        self.mark_dirty();
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        self.tab_bar.active_tab_mut().scroll_y += 40.0;
                        self.clamp_scroll();
                        self.mark_dirty();
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        self.tab_bar.active_tab_mut().scroll_y -= 40.0;
                        self.clamp_scroll();
                        self.mark_dirty();
                    }

                    _ => {}
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y * 40.0,
                    MouseScrollDelta::PixelDelta(pos) => -pos.y as f32,
                };
                self.tab_bar.active_tab_mut().scroll_y += scroll_amount;
                self.clamp_scroll();
                self.mark_dirty();
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_x = position.x as f32;
                self.mouse_y = position.y as f32;
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                self.handle_click(self.mouse_x, self.mouse_y);
            }

            WindowEvent::RedrawRequested => {
                // Check if background page load completed
                self.poll_pending_load();

                if self.needs_redraw {
                    self.render_full_frame();

                    // Only present to GPU when we actually rendered a new frame
                    if let (Some(ctx), Some(rp)) = (&self.vk_ctx, &self.render_pipeline) {
                        ctx.wait_idle(); // Fence: ensure previous present is done
                        match ctx.acquire_next_image() {
                            Ok((image_index, _)) => {
                                if let Err(e) = rp.render_frame(ctx, image_index) {
                                    eprintln!("Render error: {e}");
                                } else if let Err(e) = ctx.submit_and_present(image_index) {
                                    eprintln!("Present error: {e}");
                                }
                            }
                            Err(e) => {
                                eprintln!("Acquire failed: {e}");
                            }
                        }
                    }
                }

                // Only request next redraw if there's a pending load to poll
                if self.pending_load.is_some() {
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }

            _ => {}
        }
    }
}

/// Resolve a relative URL against a base URL.
fn resolve_url(base: &str, href: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") {
        return href.to_string();
    }
    if href.starts_with("//") {
        return format!("https:{href}");
    }

    // Try url crate for proper resolution
    if let Ok(base_url) = url::Url::parse(base) {
        if let Ok(resolved) = base_url.join(href) {
            return resolved.to_string();
        }
    }

    // Fallback: absolute path
    if href.starts_with('/') {
        if let Ok(base_url) = url::Url::parse(base) {
            return format!(
                "{}://{}{}",
                base_url.scheme(),
                base_url.host_str().unwrap_or(""),
                href
            );
        }
    }

    href.to_string()
}

fn main() {
    // Boa JS engine + deep render stack require more than the 1 MB Windows default.
    // Use any_thread(true) so we can run the event loop from a spawned thread with
    // a larger stack.  This is safe on Windows (same process, just a bigger stack).
    let builder = std::thread::Builder::new()
        .name("octoview-main".into())
        .stack_size(8 * 1024 * 1024); // 8 MB

    let handler = builder
        .spawn(|| {
            use winit::platform::windows::EventLoopBuilderExtWindows;
            let url = std::env::args().nth(1);
            let event_loop = winit::event_loop::EventLoop::builder()
                .with_any_thread(true)
                .build()
                .expect("Failed to create event loop");
            let mut app = App::new(url);
            event_loop.run_app(&mut app).expect("Event loop error");
        })
        .expect("Failed to spawn main thread");

    handler.join().expect("Main thread panicked");
}
