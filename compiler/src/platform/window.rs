//! Window management for OctoFlow — raw Win32 API, zero external dependencies.
//!
//! Provides window_open/close/draw/poll builtins for .flow programs.
//! Multi-window support: window_open returns an ID, all builtins accept optional window ID.
//! Backward compat: 0-arg versions operate on the active (most recently opened) window.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
#[cfg(target_os = "windows")]
use std::ffi::c_void;

#[cfg(target_os = "windows")]
use crate::platform::win32::*;

// ─── Per-window state ────────────────────────────────────────

struct WindowState {
    #[cfg(target_os = "windows")]
    hwnd: *mut c_void,
    width: u32,
    height: u32,
    alive: bool,
    pixels: Vec<u8>, // BGRA top-down
    // Event queue: (type_id, param, x, y)
    // type_id: 0=none, 1=close, 2=key_down, 3=key_up, 4=mouse_move, 5=mouse_down,
    //          6=mouse_up, 7=resize, 8=char, 9=scroll, 10=timer, 11=menu
    events: Vec<(u32, u32, i32, i32)>,
    // Last polled event (for getters)
    last_param: u32,
    last_x: i32,
    last_y: i32,
    // WM_CHAR: last typed character
    last_char: Option<char>,
    // WM_MOUSEWHEEL: last scroll delta
    last_scroll: i32,
    // WM_TIMER: last timer ID
    last_timer_id: u32,
    // Mouse button state (bitmask: bit0=left, bit1=right, bit2=middle)
    mouse_buttons: u32,
    // Cursor shape
    #[cfg(target_os = "windows")]
    cursor_id: u16,
}

#[cfg(target_os = "windows")]
impl WindowState {
    fn new(hwnd: *mut c_void, width: u32, height: u32) -> Self {
        WindowState {
            hwnd,
            width,
            height,
            alive: true,
            pixels: vec![0u8; (width * height * 4) as usize],
            events: Vec::new(),
            last_param: 0,
            last_x: 0,
            last_y: 0,
            last_char: None,
            last_scroll: 0,
            last_timer_id: 0,
            mouse_buttons: 0,
            cursor_id: IDC_ARROW,
        }
    }
}

// ─── Thread-local multi-window state ─────────────────────────

thread_local! {
    // All windows, keyed by window ID (0, 1, 2, ...)
    static WINDOWS: RefCell<HashMap<u32, WindowState>> = RefCell::new(HashMap::new());
    // Reverse mapping: hwnd (as usize) → window ID
    #[cfg(target_os = "windows")]
    static HWND_TO_ID: RefCell<HashMap<usize, u32>> = RefCell::new(HashMap::new());
    // Next available window ID
    static NEXT_WIN_ID: Cell<u32> = Cell::new(0);
    // Active window ID (for 0-arg builtin calls)
    static ACTIVE_WIN_ID: Cell<u32> = Cell::new(0);
    // Pending window ID during CreateWindowExW (wndproc fires before hwnd is mapped)
    #[cfg(target_os = "windows")]
    static PENDING_WIN_ID: Cell<Option<u32>> = Cell::new(None);
}

/// Resolve window ID: use explicit ID if Some, otherwise use active window.
fn resolve_id(explicit: Option<u32>) -> u32 {
    explicit.unwrap_or_else(|| ACTIVE_WIN_ID.with(|a| a.get()))
}

// ─── HWND → window ID lookup (used in wndproc) ──────────────

#[cfg(target_os = "windows")]
fn lookup_win_id(hwnd: *mut c_void) -> Option<u32> {
    let key = hwnd as usize;
    HWND_TO_ID.with(|m| m.borrow().get(&key).copied())
        .or_else(|| PENDING_WIN_ID.with(|p| p.get()))
}

// ─── Teal octopus icon (16×16 pixel art) ────────────────────

#[cfg(target_os = "windows")]
unsafe fn create_octopus_icon(hinstance: *mut c_void) -> *mut c_void {
    // Bright cyan octopus with clear silhouette, white eyes, smile detail
    #[rustfmt::skip]
    const M: [[u8; 16]; 16] = [
        [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,0,1,1,3,4,1,1,1,3,4,1,1,1,0,0],
        [0,0,1,1,3,4,1,1,1,3,4,1,1,1,0,0],
        [0,0,0,1,1,1,1,2,1,1,1,1,1,0,0,0],
        [0,0,0,1,1,2,2,2,2,2,1,1,0,0,0,0],
        [0,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0],
        [0,1,0,0,0,1,1,0,1,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
        [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0],
        [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ];

    let mut and_mask = [0u8; 64];
    for y in 0..16 {
        for x in 0..16 {
            if M[y][x] == 0 {
                let byte_idx = y * 4 + x / 8;
                and_mask[byte_idx] |= 1 << (7 - (x % 8));
            }
        }
    }

    let mut xor_mask = [0u8; 16 * 16 * 4];
    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 4;
            let (b, g, r, a) = match M[y][x] {
                1 => (200, 180, 0, 255),     // Body: bright cyan (BGR)
                2 => (180, 140, 0, 255),     // Mouth/detail: darker cyan
                3 => (255, 255, 255, 255),   // Eye white
                4 => (40, 40, 40, 255),      // Eye pupil
                _ => (0, 0, 0, 0),           // Transparent
            };
            xor_mask[idx] = b;
            xor_mask[idx + 1] = g;
            xor_mask[idx + 2] = r;
            xor_mask[idx + 3] = a;
        }
    }

    CreateIcon(hinstance, 16, 16, 1, 32, and_mask.as_ptr(), xor_mask.as_ptr())
}

// ─── VK code → key name ─────────────────────────────────────

#[cfg(target_os = "windows")]
fn vk_to_name(vk: u32) -> String {
    match vk {
        1 => "mouse_left".into(),
        2 => "mouse_right".into(),
        4 => "mouse_middle".into(),
        VK_BACK    => "backspace".into(),
        VK_TAB     => "tab".into(),
        VK_RETURN  => "enter".into(),
        VK_SHIFT   => "shift".into(),
        VK_CONTROL => "ctrl".into(),
        VK_ESCAPE  => "escape".into(),
        VK_SPACE   => "space".into(),
        VK_LEFT    => "left".into(),
        VK_UP      => "up".into(),
        VK_RIGHT   => "right".into(),
        VK_DOWN    => "down".into(),
        VK_DELETE  => "delete".into(),
        0x30..=0x39 => String::from(char::from(vk as u8)),
        0x41..=0x5A => String::from(char::from(vk as u8 + 32)),
        VK_F1..=VK_F12 => format!("f{}", vk - VK_F1 + 1),
        _ => format!("vk_{}", vk),
    }
}

/// Reverse of vk_to_name: key name → VK code. Returns 0 if unknown.
#[cfg(target_os = "windows")]
fn name_to_vk(name: &str) -> u32 {
    match name {
        "mouse_left" => 1,   // VK_LBUTTON
        "mouse_right" => 2,  // VK_RBUTTON
        "mouse_middle" => 4, // VK_MBUTTON
        "backspace" => VK_BACK,
        "tab" => VK_TAB,
        "enter" | "return" => VK_RETURN,
        "shift" => VK_SHIFT,
        "ctrl" | "control" => VK_CONTROL,
        "escape" | "esc" => VK_ESCAPE,
        "space" => VK_SPACE,
        "left" => VK_LEFT,
        "up" => VK_UP,
        "right" => VK_RIGHT,
        "down" => VK_DOWN,
        "delete" => VK_DELETE,
        s if s.len() == 1 => {
            let c = s.as_bytes()[0];
            match c {
                b'0'..=b'9' => c as u32,
                b'a'..=b'z' => (c - 32) as u32, // VK codes are uppercase
                b'A'..=b'Z' => c as u32,
                _ => 0,
            }
        }
        s if s.starts_with('f') || s.starts_with('F') => {
            if let Ok(n) = s[1..].parse::<u32>() {
                if (1..=12).contains(&n) { VK_F1 + n - 1 } else { 0 }
            } else { 0 }
        }
        _ => 0,
    }
}

// ─── WNDPROC callback ────────────────────────────────────────

#[cfg(target_os = "windows")]
unsafe extern "system" fn wndproc(hwnd: *mut c_void, msg: u32, wparam: usize, lparam: isize) -> isize {
    // Helper: push event to the correct window's event queue
    macro_rules! push_evt {
        ($type_id:expr, $param:expr, $x:expr, $y:expr) => {
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                        w.events.push(($type_id, $param, $x, $y));
                    }
                });
            }
        };
    }

    match msg {
        WM_CLOSE => {
            push_evt!(1, 0, 0, 0);
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                        w.alive = false;
                    }
                });
            }
            0
        }
        WM_DESTROY => {
            // Clean up hwnd → window ID mapping to prevent stale lookups
            HWND_TO_ID.with(|m| m.borrow_mut().remove(&(hwnd as usize)));
            PostQuitMessage(0);
            0
        }
        WM_PAINT => {
            let mut ps: PAINTSTRUCT = std::mem::zeroed();
            let hdc = BeginPaint(hwnd, &mut ps);
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    let windows = ws.borrow();
                    if let Some(w) = windows.get(&wid) {
                        let ww = w.width;
                        let wh = w.height;
                        let expected = (ww * wh * 4) as usize;
                        if w.pixels.len() >= expected && expected > 0 {
                            // W-01: Double buffer — render to memory DC, BitBlt to screen
                            let mem_dc = CreateCompatibleDC(hdc);
                            if !mem_dc.is_null() {
                                let mut bmi: BITMAPINFO = std::mem::zeroed();
                                bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
                                bmi.bmiHeader.biWidth = ww as i32;
                                bmi.bmiHeader.biHeight = -(wh as i32);
                                bmi.bmiHeader.biPlanes = 1;
                                bmi.bmiHeader.biBitCount = 32;
                                bmi.bmiHeader.biCompression = BI_RGB;
                                let mut bits: *mut c_void = std::ptr::null_mut();
                                let dib = CreateDIBSection(
                                    mem_dc, &bmi, DIB_RGB_COLORS, &mut bits,
                                    std::ptr::null_mut(), 0,
                                );
                                if !dib.is_null() && !bits.is_null() {
                                    let old = SelectObject(mem_dc, dib as HGDIOBJ);
                                    // Copy pixel data to DIB section
                                    std::ptr::copy_nonoverlapping(
                                        w.pixels.as_ptr(), bits as *mut u8, expected,
                                    );
                                    // Blit from memory DC to screen (flicker-free)
                                    BitBlt(hdc, 0, 0, ww as i32, wh as i32, mem_dc, 0, 0, SRCCOPY);
                                    SelectObject(mem_dc, old);
                                    DeleteObject(dib as HGDIOBJ);
                                } else {
                                    // Fallback: direct StretchDIBits (may flicker)
                                    let mut bmi_fb: BITMAPINFO = std::mem::zeroed();
                                    bmi_fb.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
                                    bmi_fb.bmiHeader.biWidth = ww as i32;
                                    bmi_fb.bmiHeader.biHeight = -(wh as i32);
                                    bmi_fb.bmiHeader.biPlanes = 1;
                                    bmi_fb.bmiHeader.biBitCount = 32;
                                    bmi_fb.bmiHeader.biCompression = BI_RGB;
                                    StretchDIBits(
                                        hdc, 0, 0, ww as i32, wh as i32,
                                        0, 0, ww as i32, wh as i32,
                                        w.pixels.as_ptr() as *const c_void,
                                        &bmi_fb, DIB_RGB_COLORS, SRCCOPY,
                                    );
                                }
                                DeleteDC(mem_dc);
                            } else {
                                // Fallback: no memory DC available
                                let mut bmi_fb: BITMAPINFO = std::mem::zeroed();
                                bmi_fb.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
                                bmi_fb.bmiHeader.biWidth = ww as i32;
                                bmi_fb.bmiHeader.biHeight = -(wh as i32);
                                bmi_fb.bmiHeader.biPlanes = 1;
                                bmi_fb.bmiHeader.biBitCount = 32;
                                bmi_fb.bmiHeader.biCompression = BI_RGB;
                                StretchDIBits(
                                    hdc, 0, 0, ww as i32, wh as i32,
                                    0, 0, ww as i32, wh as i32,
                                    w.pixels.as_ptr() as *const c_void,
                                    &bmi_fb, DIB_RGB_COLORS, SRCCOPY,
                                );
                            }
                        }
                    }
                });
            }
            EndPaint(hwnd, &ps);
            0
        }
        WM_KEYDOWN => {
            push_evt!(2, wparam as u32, 0, 0);
            DefWindowProcW(hwnd, msg, wparam, lparam)
        }
        WM_KEYUP => {
            push_evt!(3, wparam as u32, 0, 0);
            DefWindowProcW(hwnd, msg, wparam, lparam)
        }
        WM_CHAR => {
            if let Some(c) = std::char::from_u32(wparam as u32) {
                if !c.is_control() || c == '\t' {
                    if let Some(wid) = lookup_win_id(hwnd) {
                        WINDOWS.with(|ws| {
                            if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                                w.last_char = Some(c);
                                w.events.push((8, wparam as u32, 0, 0));
                            }
                        });
                    }
                }
            }
            0
        }
        WM_MOUSEMOVE => {
            let x = (lparam & 0xFFFF) as i16 as i32;
            let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
            push_evt!(4, 0, x, y);
            0
        }
        WM_LBUTTONDOWN => {
            let x = (lparam & 0xFFFF) as i16 as i32;
            let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
            SetCapture(hwnd); // capture mouse to track drag outside window
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                        w.mouse_buttons |= 1; // bit 0 = left
                    }
                });
            }
            push_evt!(5, 1, x, y);
            0
        }
        WM_LBUTTONUP => {
            let x = (lparam & 0xFFFF) as i16 as i32;
            let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
            let mut buttons_after = 0u32;
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                        w.mouse_buttons &= !1; // clear bit 0
                        buttons_after = w.mouse_buttons;
                    }
                });
            }
            if buttons_after == 0 { ReleaseCapture(); }
            push_evt!(6, 1, x, y);
            0
        }
        WM_RBUTTONDOWN => {
            let x = (lparam & 0xFFFF) as i16 as i32;
            let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
            SetCapture(hwnd);
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                        w.mouse_buttons |= 2; // bit 1 = right
                    }
                });
            }
            push_evt!(5, 2, x, y);
            0
        }
        WM_RBUTTONUP => {
            let x = (lparam & 0xFFFF) as i16 as i32;
            let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
            let mut buttons_after = 0u32;
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                        w.mouse_buttons &= !2; // clear bit 1
                        buttons_after = w.mouse_buttons;
                    }
                });
            }
            if buttons_after == 0 { ReleaseCapture(); }
            push_evt!(6, 2, x, y);
            0
        }
        WM_MBUTTONDOWN => {
            let x = (lparam & 0xFFFF) as i16 as i32;
            let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
            SetCapture(hwnd);
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                        w.mouse_buttons |= 4; // bit 2 = middle
                    }
                });
            }
            push_evt!(5, 3, x, y);
            0
        }
        WM_MBUTTONUP => {
            let x = (lparam & 0xFFFF) as i16 as i32;
            let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
            let mut buttons_after = 0u32;
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                        w.mouse_buttons &= !4; // clear bit 2
                        buttons_after = w.mouse_buttons;
                    }
                });
            }
            if buttons_after == 0 { ReleaseCapture(); }
            push_evt!(6, 3, x, y);
            0
        }
        WM_SIZE => {
            let new_w = (lparam as u32) & 0xFFFF;
            let new_h = ((lparam as u32) >> 16) & 0xFFFF;
            if new_w > 0 && new_h > 0 {
                if let Some(wid) = lookup_win_id(hwnd) {
                    WINDOWS.with(|ws| {
                        if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                            w.width = new_w;
                            w.height = new_h;
                            w.pixels.resize((new_w * new_h * 4) as usize, 0);
                            w.events.push((7, 0, new_w as i32, new_h as i32));
                        }
                    });
                }
            }
            0
        }
        WM_MOUSEWHEEL => {
            let delta = ((wparam >> 16) as i16) as i32;
            let scroll_lines = delta / 120;
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                        w.last_scroll += scroll_lines; // accumulate until read
                        w.events.push((9, 0, scroll_lines, 0));
                    }
                });
            }
            0
        }
        WM_SETCURSOR => {
            if (lparam & 0xFFFF) as u16 == HTCLIENT {
                let cursor_id = if let Some(wid) = lookup_win_id(hwnd) {
                    WINDOWS.with(|ws| {
                        ws.borrow().get(&wid).map_or(IDC_ARROW, |w| w.cursor_id)
                    })
                } else {
                    IDC_ARROW
                };
                SetCursor(LoadCursorW(std::ptr::null_mut(), MAKEINTRESOURCEW(cursor_id)));
                return 1;
            }
            DefWindowProcW(hwnd, msg, wparam, lparam)
        }
        WM_TIMER => {
            let timer_id = wparam as u32;
            if let Some(wid) = lookup_win_id(hwnd) {
                WINDOWS.with(|ws| {
                    if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                        w.last_timer_id = timer_id;
                        w.events.push((10, timer_id, 0, 0));
                    }
                });
            }
            0
        }
        WM_COMMAND => {
            let item_id = (wparam & 0xFFFF) as u32;
            push_evt!(11, item_id, 0, 0);
            0
        }
        WM_ERASEBKGND => 1,
        _ => DefWindowProcW(hwnd, msg, wparam, lparam)
    }
}

// ─── Public API ──────────────────────────────────────────────

/// Open a new window. Returns the window ID (float).
/// Backward compat: first window gets ID 0, subsequent get 1, 2, ...
#[cfg(target_os = "windows")]
pub fn window_open_impl(width: u32, height: u32, title: &str) -> Result<bool, String> {
    unsafe {
        let win_id = NEXT_WIN_ID.with(|n| {
            let id = n.get();
            n.set(id + 1);
            id
        });

        let hinstance = GetModuleHandleW(std::ptr::null());
        let class_name = to_wide("OctoFlowWindow");

        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
            lpfnWndProc: wndproc,
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: hinstance,
            hIcon: std::ptr::null_mut(),
            hCursor: LoadCursorW(std::ptr::null_mut(), MAKEINTRESOURCEW(IDC_ARROW)),
            hbrBackground: std::ptr::null_mut(),
            lpszMenuName: std::ptr::null(),
            lpszClassName: class_name.as_ptr(),
            hIconSm: std::ptr::null_mut(),
        };
        RegisterClassExW(&wc);

        // Create window state with placeholder hwnd (set after CreateWindowExW)
        let state = WindowState::new(std::ptr::null_mut(), width, height);
        WINDOWS.with(|ws| ws.borrow_mut().insert(win_id, state));
        PENDING_WIN_ID.with(|p| p.set(Some(win_id)));

        let style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        let mut rect = RECT { left: 0, top: 0, right: width as i32, bottom: height as i32 };
        AdjustWindowRectEx(&mut rect, style, 0, 0);

        let w_title = to_wide(title);

        let hwnd = CreateWindowExW(
            0,
            class_name.as_ptr(),
            w_title.as_ptr(),
            style | WS_VISIBLE,
            CW_USEDEFAULT, CW_USEDEFAULT,
            rect.right - rect.left, rect.bottom - rect.top,
            std::ptr::null_mut(), std::ptr::null_mut(),
            hinstance, std::ptr::null_mut(),
        );

        PENDING_WIN_ID.with(|p| p.set(None));

        if hwnd.is_null() {
            WINDOWS.with(|ws| ws.borrow_mut().remove(&win_id));
            return Err(format!("CreateWindowExW failed: error {}", GetLastError()));
        }

        // Map hwnd → window ID
        HWND_TO_ID.with(|m| m.borrow_mut().insert(hwnd as usize, win_id));

        // Store hwnd in window state
        WINDOWS.with(|ws| {
            if let Some(w) = ws.borrow_mut().get_mut(&win_id) {
                w.hwnd = hwnd;
            }
        });

        // Apply DWM dark mode
        let dark_mode: i32 = 1;
        DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
            &dark_mode as *const i32 as *const c_void, 4,
        );
        let corner_pref: u32 = DWMWCP_ROUND;
        DwmSetWindowAttribute(
            hwnd, DWMWA_WINDOW_CORNER_PREFERENCE,
            &corner_pref as *const u32 as *const c_void, 4,
        );

        // Set teal octopus icon
        let icon = create_octopus_icon(hinstance);
        if !icon.is_null() {
            SendMessageW(hwnd, WM_SETICON, ICON_SMALL, icon as isize);
            SendMessageW(hwnd, WM_SETICON, ICON_BIG, icon as isize);
        }

        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);

        // Set as active window and drain creation events
        ACTIVE_WIN_ID.with(|a| a.set(win_id));
        WINDOWS.with(|ws| {
            if let Some(w) = ws.borrow_mut().get_mut(&win_id) {
                w.events.clear();
            }
        });

        Ok(true)
    }
}

#[cfg(not(target_os = "windows"))]
pub fn window_open_impl(_width: u32, _height: u32, _title: &str) -> Result<bool, String> {
    Err("windowing requires Windows (ext.ui Linux support planned)".into())
}

/// Get the ID of the last opened window.
pub fn window_id_impl() -> u32 {
    ACTIVE_WIN_ID.with(|a| a.get())
}

pub fn window_close_impl_id(explicit_id: Option<u32>) {
    let wid = resolve_id(explicit_id);
    #[cfg(target_os = "windows")]
    {
        WINDOWS.with(|ws| {
            if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                if !w.hwnd.is_null() {
                    unsafe { DestroyWindow(w.hwnd); }
                    HWND_TO_ID.with(|m| m.borrow_mut().remove(&(w.hwnd as usize)));
                    w.hwnd = std::ptr::null_mut();
                }
                w.alive = false;
                w.events.clear();
            }
        });
    }
    #[cfg(not(target_os = "windows"))]
    {
        WINDOWS.with(|ws| {
            if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                w.alive = false;
                w.events.clear();
            }
        });
    }
}

pub fn window_close_impl() { window_close_impl_id(None); }

pub fn window_alive_impl_id(explicit_id: Option<u32>) -> bool {
    let wid = resolve_id(explicit_id);
    WINDOWS.with(|ws| {
        ws.borrow().get(&wid).map_or(false, |w| w.alive)
    })
}

pub fn window_alive_impl() -> bool { window_alive_impl_id(None) }

pub fn window_width_impl() -> u32 {
    let wid = resolve_id(None);
    WINDOWS.with(|ws| ws.borrow().get(&wid).map_or(0, |w| w.width))
}

pub fn window_height_impl() -> u32 {
    let wid = resolve_id(None);
    WINDOWS.with(|ws| ws.borrow().get(&wid).map_or(0, |w| w.height))
}

/// Blit R/G/B float arrays to pixel buffer and trigger repaint.
pub fn window_draw_impl(r: &[f32], g: &[f32], b: &[f32]) -> Result<(), String> {
    let wid = resolve_id(None);
    let alive = WINDOWS.with(|ws| {
        ws.borrow().get(&wid).map_or(false, |w| w.alive)
    });
    if !alive {
        return Ok(());
    }

    let (ww, wh) = WINDOWS.with(|ws| {
        ws.borrow().get(&wid).map_or((0u32, 0u32), |w| (w.width, w.height))
    });
    let total = (ww as usize) * (wh as usize);

    if r.len() < total || g.len() < total || b.len() < total {
        return Err(format!(
            "window_draw: arrays need {}*{}={} elements (got r={}, g={}, b={})",
            ww, wh, total, r.len(), g.len(), b.len()));
    }

    WINDOWS.with(|ws| {
        let mut windows = ws.borrow_mut();
        if let Some(w) = windows.get_mut(&wid) {
            if w.pixels.len() < total * 4 {
                w.pixels.resize(total * 4, 0);
            }
            for i in 0..total {
                w.pixels[i * 4]     = b[i].clamp(0.0, 255.0) as u8;
                w.pixels[i * 4 + 1] = g[i].clamp(0.0, 255.0) as u8;
                w.pixels[i * 4 + 2] = r[i].clamp(0.0, 255.0) as u8;
                w.pixels[i * 4 + 3] = 255;
            }
        }
    });

    #[cfg(target_os = "windows")]
    {
        let hwnd = WINDOWS.with(|ws| {
            ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
        });
        if !hwnd.is_null() {
            unsafe {
                InvalidateRect(hwnd, std::ptr::null(), 0);
                let mut msg: MSG = std::mem::zeroed();
                while PeekMessageW(&mut msg, hwnd, WM_PAINT, WM_PAINT, PM_REMOVE) != 0 {
                    DispatchMessageW(&msg);
                }
            }
        }
    }

    Ok(())
}

/// Process pending messages, pop first event. Returns type string.
pub fn window_poll_impl() -> String {
    let wid = resolve_id(None);

    #[cfg(target_os = "windows")]
    {
        let hwnd = WINDOWS.with(|ws| {
            ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
        });
        if !hwnd.is_null() {
            unsafe {
                let mut msg: MSG = std::mem::zeroed();
                while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
        }
    }

    let evt = WINDOWS.with(|ws| {
        let mut windows = ws.borrow_mut();
        if let Some(w) = windows.get_mut(&wid) {
            if w.events.is_empty() { None } else { Some(w.events.remove(0)) }
        } else {
            None
        }
    });

    match evt {
        None => "none".into(),
        Some((type_id, param, x, y)) => {
            WINDOWS.with(|ws| {
                if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                    w.last_param = param;
                    w.last_x = x;
                    w.last_y = y;
                }
            });
            match type_id {
                1 => "close".into(),
                2 => "key_down".into(),
                3 => "key_up".into(),
                4 => "mouse_move".into(),
                5 => "mouse_down".into(),
                6 => "mouse_up".into(),
                7 => "resize".into(),
                8 => "char".into(),
                9 => "scroll".into(),
                10 => "timer".into(),
                11 => "menu".into(),
                _ => "unknown".into(),
            }
        }
    }
}

pub fn window_event_key_impl() -> String {
    #[cfg(target_os = "windows")]
    {
        let wid = resolve_id(None);
        let param = WINDOWS.with(|ws| {
            ws.borrow().get(&wid).map_or(0, |w| w.last_param)
        });
        return vk_to_name(param);
    }
    #[cfg(not(target_os = "windows"))]
    { String::new() }
}

pub fn window_event_x_impl() -> f32 {
    let wid = resolve_id(None);
    WINDOWS.with(|ws| ws.borrow().get(&wid).map_or(0, |w| w.last_x)) as f32
}

pub fn window_event_y_impl() -> f32 {
    let wid = resolve_id(None);
    WINDOWS.with(|ws| ws.borrow().get(&wid).map_or(0, |w| w.last_y)) as f32
}

pub fn window_title_impl(title: &str) {
    #[cfg(target_os = "windows")]
    {
        let wid = resolve_id(None);
        let hwnd = WINDOWS.with(|ws| {
            ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
        });
        if !hwnd.is_null() {
            let w_title = to_wide(title);
            unsafe { SetWindowTextW(hwnd, w_title.as_ptr()); }
        }
    }
    let _ = title;
}

// ─── H-01: WM_CHAR text input ──────────────────────────────

pub fn window_event_char_impl() -> String {
    let wid = resolve_id(None);
    WINDOWS.with(|ws| {
        ws.borrow().get(&wid).and_then(|w| w.last_char).map_or(String::new(), |c| c.to_string())
    })
}

// ─── H-03: WM_MOUSEWHEEL scroll ────────────────────────────

pub fn window_event_scroll_impl() -> f32 {
    let wid = resolve_id(None);
    WINDOWS.with(|ws| ws.borrow().get(&wid).map_or(0, |w| w.last_scroll)) as f32
}

/// gui_scroll_y() — return accumulated scroll delta since last call, then reset.
/// Positive = scroll up, negative = scroll down.
pub fn gui_scroll_y_impl() -> f32 {
    let wid = resolve_id(None);
    WINDOWS.with(|ws| {
        let mut windows = ws.borrow_mut();
        if let Some(w) = windows.get_mut(&wid) {
            let delta = w.last_scroll;
            w.last_scroll = 0; // reset after reading
            delta as f32
        } else {
            0.0
        }
    })
}

// ─── H-04: Mouse capture for drag operations ───────────────

pub fn window_capture_mouse_impl() {
    #[cfg(target_os = "windows")]
    {
        let wid = resolve_id(None);
        let hwnd = WINDOWS.with(|ws| {
            ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
        });
        if !hwnd.is_null() {
            unsafe { SetCapture(hwnd); }
        }
    }
}

pub fn window_release_mouse_impl() {
    #[cfg(target_os = "windows")]
    {
        unsafe { ReleaseCapture(); }
    }
}

// ─── H-05: Clipboard support ───────────────────────────────

pub fn clipboard_get_impl() -> String {
    #[cfg(target_os = "windows")]
    {
        unsafe {
            let wid = resolve_id(None);
            let hwnd = WINDOWS.with(|ws| {
                ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
            });
            if OpenClipboard(hwnd) == 0 {
                return String::new();
            }
            let handle = GetClipboardData(CF_UNICODETEXT);
            if handle.is_null() {
                CloseClipboard();
                return String::new();
            }
            let ptr = GlobalLock(handle);
            if ptr.is_null() {
                CloseClipboard();
                return String::new();
            }
            let wptr = ptr as *const u16;
            let mut len = 0;
            while *wptr.add(len) != 0 { len += 1; }
            let wslice = std::slice::from_raw_parts(wptr, len);
            let result = String::from_utf16_lossy(wslice);
            GlobalUnlock(handle);
            CloseClipboard();
            return result;
        }
    }
    #[cfg(not(target_os = "windows"))]
    { String::new() }
}

pub fn clipboard_set_impl(text: &str) {
    #[cfg(target_os = "windows")]
    {
        unsafe {
            let wid = resolve_id(None);
            let hwnd = WINDOWS.with(|ws| {
                ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
            });
            if OpenClipboard(hwnd) == 0 {
                return;
            }
            EmptyClipboard();
            let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0u16)).collect();
            let byte_len = wide.len() * 2;
            let hmem = GlobalAlloc(GMEM_MOVEABLE, byte_len);
            if hmem.is_null() {
                CloseClipboard();
                return;
            }
            let ptr = GlobalLock(hmem);
            if !ptr.is_null() {
                std::ptr::copy_nonoverlapping(
                    wide.as_ptr() as *const u8, ptr as *mut u8, byte_len,
                );
                GlobalUnlock(hmem);
                SetClipboardData(CF_UNICODETEXT, hmem);
            }
            CloseClipboard();
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = text; }
}

// ─── H-06: File dialog builtins ─────────────────────────────

pub fn dialog_open_file_impl(title: &str, filter: &str) -> String {
    #[cfg(target_os = "windows")]
    {
        unsafe {
            let wid = resolve_id(None);
            let hwnd = WINDOWS.with(|ws| {
                ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
            });
            let mut file_buf = [0u16; 260];
            let w_title = to_wide(title);
            let filter_str = filter.replace('|', "\0");
            let mut w_filter: Vec<u16> = filter_str.encode_utf16().collect();
            w_filter.push(0);
            w_filter.push(0);

            let mut ofn: OPENFILENAMEW = std::mem::zeroed();
            ofn.lStructSize = std::mem::size_of::<OPENFILENAMEW>() as u32;
            ofn.hwndOwner = hwnd;
            ofn.lpstrFilter = w_filter.as_ptr();
            ofn.lpstrFile = file_buf.as_mut_ptr();
            ofn.nMaxFile = 260;
            ofn.lpstrTitle = w_title.as_ptr();
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

            if GetOpenFileNameW(&mut ofn) != 0 {
                let mut len = 0;
                while len < 260 && file_buf[len] != 0 { len += 1; }
                return String::from_utf16_lossy(&file_buf[..len]);
            }
            return String::new();
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = (title, filter); String::new() }
}

pub fn dialog_save_file_impl(title: &str, filter: &str) -> String {
    #[cfg(target_os = "windows")]
    {
        unsafe {
            let wid = resolve_id(None);
            let hwnd = WINDOWS.with(|ws| {
                ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
            });
            let mut file_buf = [0u16; 260];
            let w_title = to_wide(title);
            let filter_str = filter.replace('|', "\0");
            let mut w_filter: Vec<u16> = filter_str.encode_utf16().collect();
            w_filter.push(0);
            w_filter.push(0);

            let mut ofn: OPENFILENAMEW = std::mem::zeroed();
            ofn.lStructSize = std::mem::size_of::<OPENFILENAMEW>() as u32;
            ofn.hwndOwner = hwnd;
            ofn.lpstrFilter = w_filter.as_ptr();
            ofn.lpstrFile = file_buf.as_mut_ptr();
            ofn.nMaxFile = 260;
            ofn.lpstrTitle = w_title.as_ptr();
            ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;

            if GetSaveFileNameW(&mut ofn) != 0 {
                let mut len = 0;
                while len < 260 && file_buf[len] != 0 { len += 1; }
                return String::from_utf16_lossy(&file_buf[..len]);
            }
            return String::new();
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = (title, filter); String::new() }
}

pub fn dialog_message_impl(title: &str, message: &str, msg_type: &str) -> String {
    #[cfg(target_os = "windows")]
    {
        unsafe {
            let wid = resolve_id(None);
            let hwnd = WINDOWS.with(|ws| {
                ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
            });
            let w_title = to_wide(title);
            let w_msg = to_wide(message);
            let mb_type = match msg_type {
                "yesno" => MB_YESNO,
                "yesnocancel" => MB_YESNOCANCEL,
                "okcancel" => MB_OKCANCEL,
                _ => MB_OK,
            };
            let result = MessageBoxW(hwnd, w_msg.as_ptr(), w_title.as_ptr(), mb_type);
            return match result {
                IDOK => "ok".into(),
                IDYES => "yes".into(),
                IDNO => "no".into(),
                IDCANCEL => "cancel".into(),
                _ => "ok".into(),
            };
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = (title, message, msg_type); "ok".into() }
}

// ─── H-07: Cursor shape control ─────────────────────────────

pub fn window_set_cursor_impl(name: &str) {
    #[cfg(target_os = "windows")]
    {
        let cursor_id = match name {
            "ibeam" => IDC_IBEAM,
            "hand" => IDC_HAND,
            "crosshair" => IDC_CROSS,
            "wait" => IDC_WAIT,
            "resize_ns" => IDC_SIZENS,
            "resize_ew" => IDC_SIZEWE,
            "move" => IDC_SIZEALL,
            _ => IDC_ARROW,
        };
        let wid = resolve_id(None);
        WINDOWS.with(|ws| {
            if let Some(w) = ws.borrow_mut().get_mut(&wid) {
                w.cursor_id = cursor_id;
            }
        });
        unsafe {
            SetCursor(LoadCursorW(std::ptr::null_mut(), MAKEINTRESOURCEW(cursor_id)));
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = name; }
}

// ─── H-08: Timer support ───────────────────────────────────

pub fn window_set_timer_impl(id: u32, ms: u32) {
    #[cfg(target_os = "windows")]
    {
        let wid = resolve_id(None);
        let hwnd = WINDOWS.with(|ws| {
            ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
        });
        if !hwnd.is_null() {
            unsafe { SetTimer(hwnd, id as usize, ms, std::ptr::null()); }
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = (id, ms); }
}

pub fn window_kill_timer_impl(id: u32) {
    #[cfg(target_os = "windows")]
    {
        let wid = resolve_id(None);
        let hwnd = WINDOWS.with(|ws| {
            ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
        });
        if !hwnd.is_null() {
            unsafe { KillTimer(hwnd, id as usize); }
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = id; }
}

pub fn window_event_timer_id_impl() -> f32 {
    let wid = resolve_id(None);
    WINDOWS.with(|ws| ws.borrow().get(&wid).map_or(0, |w| w.last_timer_id)) as f32
}

// ─── H-09: Menu bar support ────────────────────────────────

pub fn window_create_menu_impl() -> f64 {
    #[cfg(target_os = "windows")]
    {
        unsafe {
            let hmenu = CreateMenu();
            return hmenu as usize as f64;
        }
    }
    #[cfg(not(target_os = "windows"))]
    { 0.0 }
}

pub fn menu_add_item_impl(menu_handle: f64, id: u32, label: &str) {
    #[cfg(target_os = "windows")]
    {
        unsafe {
            let hmenu = menu_handle as usize as *mut c_void;
            let w_label = to_wide(label);
            AppendMenuW(hmenu, MF_STRING, id as usize, w_label.as_ptr());
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = (menu_handle, id, label); }
}

pub fn menu_add_submenu_impl(menu_handle: f64, label: &str, submenu_handle: f64) {
    #[cfg(target_os = "windows")]
    {
        unsafe {
            let hmenu = menu_handle as usize as *mut c_void;
            let hsub = submenu_handle as usize as *mut c_void;
            let w_label = to_wide(label);
            AppendMenuW(hmenu, MF_POPUP, hsub as usize, w_label.as_ptr());
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = (menu_handle, label, submenu_handle); }
}

pub fn window_set_menu_impl(menu_handle: f64) {
    #[cfg(target_os = "windows")]
    {
        let wid = resolve_id(None);
        let hwnd = WINDOWS.with(|ws| {
            ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
        });
        if !hwnd.is_null() {
            unsafe {
                let hmenu = menu_handle as usize as *mut c_void;
                SetMenu(hwnd, hmenu);
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = menu_handle; }
}

// ─── H-11: DPI awareness ───────────────────────────────────

pub fn window_dpi_impl() -> f32 {
    #[cfg(target_os = "windows")]
    {
        let wid = resolve_id(None);
        let hwnd = WINDOWS.with(|ws| {
            ws.borrow().get(&wid).map_or(std::ptr::null_mut(), |w| w.hwnd)
        });
        if hwnd.is_null() {
            return 96.0;
        }
        unsafe {
            let dpi = GetDpiForWindow(hwnd);
            if dpi == 0 { 96.0 } else { dpi as f32 }
        }
    }
    #[cfg(not(target_os = "windows"))]
    { 96.0 }
}

// ─── R-06: Mouse button state query ─────────────────────────

/// Returns 1.0 if left mouse button is currently pressed, 0.0 otherwise.
pub fn gui_mouse_down_impl() -> f32 {
    let wid = resolve_id(None);
    WINDOWS.with(|ws| {
        ws.borrow().get(&wid).map_or(0.0, |w| if w.mouse_buttons & 1 != 0 { 1.0 } else { 0.0 })
    })
}

/// Returns bitmask of pressed mouse buttons (1=left, 2=right, 4=middle).
pub fn gui_mouse_buttons_impl() -> f32 {
    let wid = resolve_id(None);
    WINDOWS.with(|ws| {
        ws.borrow().get(&wid).map_or(0.0, |w| w.mouse_buttons as f32)
    })
}

// ─── R-08: Continuous key-held state ─────────────────────────

/// Returns 1.0 if the named key is currently held down. Uses GetAsyncKeyState.
pub fn window_key_held_impl(key_name: &str) -> f32 {
    #[cfg(target_os = "windows")]
    {
        let vk = name_to_vk(key_name);
        if vk == 0 { return 0.0; }
        unsafe {
            let state = GetAsyncKeyState(vk as i32);
            if state & (0x8000u16 as i16) != 0 { 1.0 } else { 0.0 }
        }
    }
    #[cfg(not(target_os = "windows"))]
    { let _ = key_name; 0.0 }
}
