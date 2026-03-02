//! Raw Win32 API bindings — zero external dependencies.
//!
//! Direct `extern "system"` declarations against:
//!   user32.dll   — window management, input events, message processing
//!   gdi32.dll    — device contexts, DIB pixel rendering
//!   kernel32.dll — module handles, error codes

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;

// ─── Type aliases ────────────────────────────────────────────────

pub type HWND = *mut c_void;
pub type HINSTANCE = *mut c_void;
pub type HDC = *mut c_void;
pub type HMENU = *mut c_void;
pub type HBRUSH = *mut c_void;
pub type HICON = *mut c_void;
pub type HCURSOR = *mut c_void;
pub type HFONT = *mut c_void;
pub type HBITMAP = *mut c_void;
pub type HGDIOBJ = *mut c_void;
pub type COLORREF = u32;
pub type BOOL = i32;
pub type LRESULT = isize;
pub type WPARAM = usize;
pub type LPARAM = isize;
pub type UINT = u32;
pub type DWORD = u32;
pub type WORD = u16;
pub type ATOM = u16;
pub type LONG = i32;
pub type BYTE = u8;
pub type LPCSTR = *const i8;
pub type LPCWSTR = *const u16;
pub type LPVOID = *mut c_void;

pub type WNDPROC = unsafe extern "system" fn(HWND, UINT, WPARAM, LPARAM) -> LRESULT;

// ─── Constants ───────────────────────────────────────────────────

// Window styles
pub const WS_OVERLAPPED: u32      = 0x00000000;
pub const WS_CAPTION: u32         = 0x00C00000;
pub const WS_SYSMENU: u32         = 0x00080000;
pub const WS_THICKFRAME: u32      = 0x00040000;
pub const WS_MINIMIZEBOX: u32     = 0x00020000;
pub const WS_MAXIMIZEBOX: u32     = 0x00010000;
pub const WS_OVERLAPPEDWINDOW: u32 = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU
    | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX;
pub const WS_VISIBLE: u32         = 0x10000000;

// Class styles
pub const CS_HREDRAW: u32 = 0x0002;
pub const CS_VREDRAW: u32 = 0x0001;
pub const CS_OWNDC: u32   = 0x0020;

// Window position
pub const CW_USEDEFAULT: i32 = 0x80000000u32 as i32;

// ShowWindow commands
pub const SW_SHOW: i32 = 5;

// Icon messages
pub const WM_SETICON: u32       = 0x0080;
pub const ICON_SMALL: usize     = 0;
pub const ICON_BIG: usize       = 1;

// Messages
pub const WM_DESTROY: u32       = 0x0002;
pub const WM_SIZE: u32          = 0x0005;
pub const WM_PAINT: u32         = 0x000F;
pub const WM_CLOSE: u32         = 0x0010;
pub const WM_ERASEBKGND: u32    = 0x0014;
pub const WM_SETCURSOR: u32     = 0x0020;
pub const WM_COMMAND: u32       = 0x0111;
pub const WM_TIMER: u32         = 0x0113;
pub const WM_KEYDOWN: u32       = 0x0100;
pub const WM_KEYUP: u32         = 0x0101;
pub const WM_CHAR: u32          = 0x0102;
pub const WM_MOUSEMOVE: u32     = 0x0200;
pub const WM_LBUTTONDOWN: u32   = 0x0201;
pub const WM_LBUTTONUP: u32     = 0x0202;
pub const WM_RBUTTONDOWN: u32   = 0x0204;
pub const WM_RBUTTONUP: u32     = 0x0205;
pub const WM_MBUTTONDOWN: u32   = 0x0207;
pub const WM_MBUTTONUP: u32     = 0x0208;
pub const WM_MOUSEWHEEL: u32    = 0x020A;
pub const WM_DPICHANGED: u32    = 0x02E0;

// PeekMessage flags
pub const PM_REMOVE: u32 = 0x0001;

// GDI constants
pub const DIB_RGB_COLORS: u32 = 0;
pub const SRCCOPY: u32        = 0x00CC0020;
pub const BI_RGB: u32         = 0;

// GDI font constants
pub const FW_NORMAL: i32          = 400;
pub const DEFAULT_CHARSET: u32    = 1;
pub const OUT_TT_PRECIS: u32      = 4;
pub const CLIP_DEFAULT_PRECIS: u32 = 0;
pub const CLEARTYPE_QUALITY: u32  = 5;
pub const DEFAULT_PITCH: u32      = 0;
pub const TRANSPARENT_BK: i32     = 1;

// Cursor IDs
pub const IDC_ARROW: u16     = 32512;
pub const IDC_IBEAM: u16     = 32513;
pub const IDC_WAIT: u16      = 32514;
pub const IDC_CROSS: u16     = 32515;
pub const IDC_SIZEALL: u16   = 32646;
pub const IDC_SIZENS: u16    = 32645;
pub const IDC_SIZEWE: u16    = 32644;
pub const IDC_HAND: u16      = 32649;

// Clipboard formats
pub const CF_TEXT: u32 = 1;
pub const CF_UNICODETEXT: u32 = 13;

// GlobalAlloc flags
pub const GMEM_MOVEABLE: u32 = 0x0002;

// MessageBox types
pub const MB_OK: u32            = 0x00000000;
pub const MB_OKCANCEL: u32      = 0x00000001;
pub const MB_YESNO: u32         = 0x00000004;
pub const MB_YESNOCANCEL: u32   = 0x00000003;

// MessageBox return values
pub const IDOK: i32     = 1;
pub const IDCANCEL: i32 = 2;
pub const IDYES: i32    = 6;
pub const IDNO: i32     = 7;

// Menu flags
pub const MF_STRING: u32    = 0x00000000;
pub const MF_POPUP: u32     = 0x00000010;

// HTCLIENT for WM_SETCURSOR
pub const HTCLIENT: u16 = 1;

// Virtual key codes
pub const VK_BACK: u32    = 0x08;
pub const VK_TAB: u32     = 0x09;
pub const VK_RETURN: u32  = 0x0D;
pub const VK_SHIFT: u32   = 0x10;
pub const VK_CONTROL: u32 = 0x11;
pub const VK_ESCAPE: u32  = 0x1B;
pub const VK_SPACE: u32   = 0x20;
pub const VK_LEFT: u32    = 0x25;
pub const VK_UP: u32      = 0x26;
pub const VK_RIGHT: u32   = 0x27;
pub const VK_DOWN: u32    = 0x28;
pub const VK_DELETE: u32  = 0x2E;
pub const VK_F1: u32      = 0x70;
pub const VK_F12: u32     = 0x7B;

// ─── Structures ──────────────────────────────────────────────────

#[repr(C)]
pub struct POINT {
    pub x: LONG,
    pub y: LONG,
}

#[repr(C)]
pub struct SIZE {
    pub cx: LONG,
    pub cy: LONG,
}

#[repr(C)]
pub struct RECT {
    pub left: LONG,
    pub top: LONG,
    pub right: LONG,
    pub bottom: LONG,
}

#[repr(C)]
pub struct WNDCLASSEXA {
    pub cbSize: UINT,
    pub style: UINT,
    pub lpfnWndProc: WNDPROC,
    pub cbClsExtra: i32,
    pub cbWndExtra: i32,
    pub hInstance: HINSTANCE,
    pub hIcon: HICON,
    pub hCursor: HCURSOR,
    pub hbrBackground: HBRUSH,
    pub lpszMenuName: LPCSTR,
    pub lpszClassName: LPCSTR,
    pub hIconSm: HICON,
}

#[repr(C)]
pub struct WNDCLASSEXW {
    pub cbSize: UINT,
    pub style: UINT,
    pub lpfnWndProc: WNDPROC,
    pub cbClsExtra: i32,
    pub cbWndExtra: i32,
    pub hInstance: HINSTANCE,
    pub hIcon: HICON,
    pub hCursor: HCURSOR,
    pub hbrBackground: HBRUSH,
    pub lpszMenuName: LPCWSTR,
    pub lpszClassName: LPCWSTR,
    pub hIconSm: HICON,
}

#[repr(C)]
pub struct MSG {
    pub hwnd: HWND,
    pub message: UINT,
    pub wParam: WPARAM,
    pub lParam: LPARAM,
    pub time: DWORD,
    pub pt: POINT,
}

#[repr(C)]
pub struct PAINTSTRUCT {
    pub hdc: HDC,
    pub fErase: BOOL,
    pub rcPaint: RECT,
    pub fRestore: BOOL,
    pub fIncUpdate: BOOL,
    pub rgbReserved: [BYTE; 32],
}

#[repr(C)]
pub struct BITMAPINFOHEADER {
    pub biSize: DWORD,
    pub biWidth: LONG,
    pub biHeight: LONG,
    pub biPlanes: WORD,
    pub biBitCount: WORD,
    pub biCompression: DWORD,
    pub biSizeImage: DWORD,
    pub biXPelsPerMeter: LONG,
    pub biYPelsPerMeter: LONG,
    pub biClrUsed: DWORD,
    pub biClrImportant: DWORD,
}

#[repr(C)]
pub struct RGBQUAD {
    pub rgbBlue: BYTE,
    pub rgbGreen: BYTE,
    pub rgbRed: BYTE,
    pub rgbReserved: BYTE,
}

#[repr(C)]
pub struct BITMAPINFO {
    pub bmiHeader: BITMAPINFOHEADER,
    pub bmiColors: [RGBQUAD; 1],
}

// OPENFILENAMEA for file dialogs (comdlg32)
#[repr(C)]
pub struct OPENFILENAMEA {
    pub lStructSize: DWORD,
    pub hwndOwner: HWND,
    pub hInstance: HINSTANCE,
    pub lpstrFilter: LPCSTR,
    pub lpstrCustomFilter: *mut i8,
    pub nMaxCustFilter: DWORD,
    pub nFilterIndex: DWORD,
    pub lpstrFile: *mut i8,
    pub nMaxFile: DWORD,
    pub lpstrFileTitle: *mut i8,
    pub nMaxFileTitle: DWORD,
    pub lpstrInitialDir: LPCSTR,
    pub lpstrTitle: LPCSTR,
    pub Flags: DWORD,
    pub nFileOffset: WORD,
    pub nFileExtension: WORD,
    pub lpstrDefExt: LPCSTR,
    pub lCustData: LPARAM,
    pub lpfnHook: *const c_void,
    pub lpTemplateName: LPCSTR,
    pub pvReserved: *mut c_void,
    pub dwReserved: DWORD,
    pub FlagsEx: DWORD,
}

// OPENFILENAMEW for Unicode file dialogs (comdlg32)
#[repr(C)]
pub struct OPENFILENAMEW {
    pub lStructSize: DWORD,
    pub hwndOwner: HWND,
    pub hInstance: HINSTANCE,
    pub lpstrFilter: LPCWSTR,
    pub lpstrCustomFilter: *mut u16,
    pub nMaxCustFilter: DWORD,
    pub nFilterIndex: DWORD,
    pub lpstrFile: *mut u16,
    pub nMaxFile: DWORD,
    pub lpstrFileTitle: *mut u16,
    pub nMaxFileTitle: DWORD,
    pub lpstrInitialDir: LPCWSTR,
    pub lpstrTitle: LPCWSTR,
    pub Flags: DWORD,
    pub nFileOffset: WORD,
    pub nFileExtension: WORD,
    pub lpstrDefExt: LPCWSTR,
    pub lCustData: LPARAM,
    pub lpfnHook: *const c_void,
    pub lpTemplateName: LPCWSTR,
    pub pvReserved: *mut c_void,
    pub dwReserved: DWORD,
    pub FlagsEx: DWORD,
}

pub const OFN_FILEMUSTEXIST: DWORD = 0x00001000;
pub const OFN_PATHMUSTEXIST: DWORD = 0x00000800;
pub const OFN_OVERWRITEPROMPT: DWORD = 0x00000002;

// ─── Helpers ─────────────────────────────────────────────────────

pub fn MAKEINTRESOURCEA(id: u16) -> LPCSTR {
    id as usize as LPCSTR
}

pub fn MAKEINTRESOURCEW(id: u16) -> LPCWSTR {
    id as usize as LPCWSTR
}

/// Convert a Rust &str to a null-terminated UTF-16 Vec for W-suffix Win32 APIs.
pub fn to_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0u16)).collect()
}

// ─── Function imports ────────────────────────────────────────────

#[link(name = "kernel32")]
extern "system" {
    pub fn GetModuleHandleA(lpModuleName: LPCSTR) -> HINSTANCE;
    pub fn GetModuleHandleW(lpModuleName: LPCWSTR) -> HINSTANCE;
    pub fn GetLastError() -> DWORD;
}

#[link(name = "user32")]
extern "system" {
    pub fn RegisterClassExA(lpwcx: *const WNDCLASSEXA) -> ATOM;
    pub fn CreateWindowExA(
        dwExStyle: DWORD, lpClassName: LPCSTR, lpWindowName: LPCSTR,
        dwStyle: DWORD, x: i32, y: i32, nWidth: i32, nHeight: i32,
        hWndParent: HWND, hMenu: HMENU, hInstance: HINSTANCE, lpParam: LPVOID,
    ) -> HWND;
    pub fn ShowWindow(hWnd: HWND, nCmdShow: i32) -> BOOL;
    pub fn UpdateWindow(hWnd: HWND) -> BOOL;
    pub fn DestroyWindow(hWnd: HWND) -> BOOL;
    pub fn DefWindowProcA(hWnd: HWND, Msg: UINT, wParam: WPARAM, lParam: LPARAM) -> LRESULT;
    pub fn PeekMessageA(
        lpMsg: *mut MSG, hWnd: HWND,
        wMsgFilterMin: UINT, wMsgFilterMax: UINT, wRemoveMsg: UINT,
    ) -> BOOL;
    pub fn TranslateMessage(lpMsg: *const MSG) -> BOOL;
    pub fn DispatchMessageA(lpMsg: *const MSG) -> LRESULT;
    pub fn PostQuitMessage(nExitCode: i32);
    pub fn InvalidateRect(hWnd: HWND, lpRect: *const RECT, bErase: BOOL) -> BOOL;
    pub fn LoadCursorA(hInstance: HINSTANCE, lpCursorName: LPCSTR) -> HCURSOR;
    pub fn SetWindowTextA(hWnd: HWND, lpString: LPCSTR) -> BOOL;
    pub fn AdjustWindowRectEx(
        lpRect: *mut RECT, dwStyle: DWORD, bMenu: BOOL, dwExStyle: DWORD,
    ) -> BOOL;
    pub fn BeginPaint(hWnd: HWND, lpPaint: *mut PAINTSTRUCT) -> HDC;
    pub fn EndPaint(hWnd: HWND, lpPaint: *const PAINTSTRUCT) -> BOOL;
    pub fn SendMessageA(hWnd: HWND, Msg: UINT, wParam: WPARAM, lParam: LPARAM) -> LRESULT;
    pub fn CreateIcon(
        hInstance: HINSTANCE, nWidth: i32, nHeight: i32,
        cPlanes: u8, cBitsPixel: u8,
        lpbANDbitsMask: *const u8, lpbXORbitsColor: *const u8,
    ) -> HICON;
    // Mouse capture
    pub fn SetCapture(hWnd: HWND) -> HWND;
    pub fn ReleaseCapture() -> BOOL;
    // Key state
    pub fn GetAsyncKeyState(vKey: i32) -> i16;
    // Clipboard
    pub fn OpenClipboard(hWndNewOwner: HWND) -> BOOL;
    pub fn CloseClipboard() -> BOOL;
    pub fn EmptyClipboard() -> BOOL;
    pub fn GetClipboardData(uFormat: UINT) -> *mut c_void;
    pub fn SetClipboardData(uFormat: UINT, hMem: *mut c_void) -> *mut c_void;
    // Cursor
    pub fn SetCursor(hCursor: HCURSOR) -> HCURSOR;
    // Timers
    pub fn SetTimer(hWnd: HWND, nIDEvent: usize, uElapse: UINT, lpTimerFunc: *const c_void) -> usize;
    pub fn KillTimer(hWnd: HWND, uIDEvent: usize) -> BOOL;
    // Message box
    pub fn MessageBoxA(hWnd: HWND, lpText: LPCSTR, lpCaption: LPCSTR, uType: UINT) -> i32;
    // Menus
    pub fn CreateMenu() -> HMENU;
    pub fn CreatePopupMenu() -> HMENU;
    pub fn AppendMenuA(hMenu: HMENU, uFlags: UINT, uIDNewItem: usize, lpNewItem: LPCSTR) -> BOOL;
    pub fn SetMenu(hWnd: HWND, hMenu: HMENU) -> BOOL;
    // DPI
    pub fn GetDpiForWindow(hWnd: HWND) -> UINT;
    // Unicode (W-suffix) variants
    pub fn RegisterClassExW(lpwcx: *const WNDCLASSEXW) -> ATOM;
    pub fn CreateWindowExW(
        dwExStyle: DWORD, lpClassName: LPCWSTR, lpWindowName: LPCWSTR,
        dwStyle: DWORD, x: i32, y: i32, nWidth: i32, nHeight: i32,
        hWndParent: HWND, hMenu: HMENU, hInstance: HINSTANCE, lpParam: LPVOID,
    ) -> HWND;
    pub fn DefWindowProcW(hWnd: HWND, Msg: UINT, wParam: WPARAM, lParam: LPARAM) -> LRESULT;
    pub fn PeekMessageW(
        lpMsg: *mut MSG, hWnd: HWND,
        wMsgFilterMin: UINT, wMsgFilterMax: UINT, wRemoveMsg: UINT,
    ) -> BOOL;
    pub fn DispatchMessageW(lpMsg: *const MSG) -> LRESULT;
    pub fn LoadCursorW(hInstance: HINSTANCE, lpCursorName: LPCWSTR) -> HCURSOR;
    pub fn SetWindowTextW(hWnd: HWND, lpString: LPCWSTR) -> BOOL;
    pub fn SendMessageW(hWnd: HWND, Msg: UINT, wParam: WPARAM, lParam: LPARAM) -> LRESULT;
    pub fn MessageBoxW(hWnd: HWND, lpText: LPCWSTR, lpCaption: LPCWSTR, uType: UINT) -> i32;
    pub fn AppendMenuW(hMenu: HMENU, uFlags: UINT, uIDNewItem: usize, lpNewItem: LPCWSTR) -> BOOL;
}

#[link(name = "kernel32")]
extern "system" {
    pub fn GlobalAlloc(uFlags: UINT, dwBytes: usize) -> *mut c_void;
    pub fn GlobalLock(hMem: *mut c_void) -> *mut c_void;
    pub fn GlobalUnlock(hMem: *mut c_void) -> BOOL;
}

#[link(name = "comdlg32")]
extern "system" {
    pub fn GetOpenFileNameA(lpofn: *mut OPENFILENAMEA) -> BOOL;
    pub fn GetSaveFileNameA(lpofn: *mut OPENFILENAMEA) -> BOOL;
    pub fn GetOpenFileNameW(lpofn: *mut OPENFILENAMEW) -> BOOL;
    pub fn GetSaveFileNameW(lpofn: *mut OPENFILENAMEW) -> BOOL;
}

#[link(name = "gdi32")]
extern "system" {
    pub fn StretchDIBits(
        hdc: HDC,
        xDest: i32, yDest: i32, DestWidth: i32, DestHeight: i32,
        xSrc: i32, ySrc: i32, SrcWidth: i32, SrcHeight: i32,
        lpBits: *const c_void, lpbmi: *const BITMAPINFO,
        iUsage: UINT, rop: DWORD,
    ) -> i32;
    pub fn CreateCompatibleDC(hdc: HDC) -> HDC;
    pub fn CreateDIBSection(
        hdc: HDC, pbmi: *const BITMAPINFO, usage: UINT,
        ppvBits: *mut *mut c_void, hSection: *mut c_void, offset: DWORD,
    ) -> HBITMAP;
    pub fn SelectObject(hdc: HDC, h: HGDIOBJ) -> HGDIOBJ;
    pub fn DeleteDC(hdc: HDC) -> BOOL;
    pub fn DeleteObject(ho: HGDIOBJ) -> BOOL;
    pub fn BitBlt(
        hdc: HDC, x: i32, y: i32, cx: i32, cy: i32,
        hdcSrc: HDC, x1: i32, y1: i32, rop: DWORD,
    ) -> BOOL;
    pub fn CreateFontA(
        cHeight: i32, cWidth: i32, cEscapement: i32, cOrientation: i32,
        cWeight: i32, bItalic: DWORD, bUnderline: DWORD, bStrikeOut: DWORD,
        iCharSet: DWORD, iOutPrecision: DWORD, iClipPrecision: DWORD,
        iQuality: DWORD, iPitchAndFamily: DWORD, pszFaceName: LPCSTR,
    ) -> HFONT;
    pub fn SetTextColor(hdc: HDC, color: COLORREF) -> COLORREF;
    pub fn SetBkMode(hdc: HDC, mode: i32) -> i32;
    pub fn TextOutA(hdc: HDC, x: i32, y: i32, lpString: LPCSTR, c: i32) -> BOOL;
    pub fn GetTextExtentPoint32A(
        hdc: HDC, lpString: LPCSTR, c: i32, psizl: *mut SIZE,
    ) -> BOOL;
    // Unicode (W-suffix) variants
    pub fn CreateFontW(
        cHeight: i32, cWidth: i32, cEscapement: i32, cOrientation: i32,
        cWeight: i32, bItalic: DWORD, bUnderline: DWORD, bStrikeOut: DWORD,
        iCharSet: DWORD, iOutPrecision: DWORD, iClipPrecision: DWORD,
        iQuality: DWORD, iPitchAndFamily: DWORD, pszFaceName: LPCWSTR,
    ) -> HFONT;
    pub fn TextOutW(hdc: HDC, x: i32, y: i32, lpString: LPCWSTR, c: i32) -> BOOL;
    pub fn GetTextExtentPoint32W(
        hdc: HDC, lpString: LPCWSTR, c: i32, psizl: *mut SIZE,
    ) -> BOOL;
}

// ─── DWM (Desktop Window Manager) — modern window styling ────
//
// dwmapi.dll provides dark mode title bars, rounded corners,
// and custom border/caption colors on Windows 10 1903+ and 11.

pub const DWMWA_USE_IMMERSIVE_DARK_MODE: u32 = 20;
pub const DWMWA_WINDOW_CORNER_PREFERENCE: u32 = 33;
pub const DWMWCP_ROUND: u32 = 2;

#[link(name = "dwmapi")]
extern "system" {
    pub fn DwmSetWindowAttribute(
        hwnd: HWND, dwAttribute: DWORD,
        pvAttribute: *const c_void, cbAttribute: DWORD,
    ) -> i32; // HRESULT
}

// ─── Audio: winmm.dll (MME API) ─────────────────────────────

/// WAVEFORMATEX — audio format descriptor for waveOut.
#[repr(C)]
#[derive(Clone)]
pub struct WAVEFORMATEX {
    pub wFormatTag: u16,
    pub nChannels: u16,
    pub nSamplesPerSec: u32,
    pub nAvgBytesPerSec: u32,
    pub nBlockAlign: u16,
    pub wBitsPerSample: u16,
    pub cbSize: u16,
}

/// WAVEHDR — buffer header for waveOut.
#[repr(C)]
pub struct WAVEHDR {
    pub lpData: *mut i8,
    pub dwBufferLength: u32,
    pub dwBytesRecorded: u32,
    pub dwUser: usize,
    pub dwFlags: u32,
    pub dwLoops: u32,
    pub lpNext: *mut WAVEHDR,
    pub reserved: usize,
}

pub const WAVE_FORMAT_PCM: u16 = 1;
pub const WAVE_MAPPER: u32 = 0xFFFFFFFF;
pub const WHDR_DONE: u32 = 0x01;
pub const WHDR_PREPARED: u32 = 0x02;
pub const CALLBACK_NULL: u32 = 0;
pub type HWAVEOUT = *mut c_void;

// PlaySound flags
pub const SND_FILENAME: u32 = 0x00020000;
pub const SND_ASYNC: u32 = 0x0001;
pub const SND_NODEFAULT: u32 = 0x0002;

#[link(name = "winmm")]
extern "system" {
    pub fn waveOutOpen(
        phwo: *mut HWAVEOUT, uDeviceID: u32,
        pwfx: *const WAVEFORMATEX, dwCallback: usize,
        dwInstance: usize, fdwOpen: u32,
    ) -> u32; // MMRESULT
    pub fn waveOutPrepareHeader(
        hwo: HWAVEOUT, pwh: *mut WAVEHDR, cbwh: u32,
    ) -> u32;
    pub fn waveOutWrite(
        hwo: HWAVEOUT, pwh: *mut WAVEHDR, cbwh: u32,
    ) -> u32;
    pub fn waveOutUnprepareHeader(
        hwo: HWAVEOUT, pwh: *mut WAVEHDR, cbwh: u32,
    ) -> u32;
    pub fn waveOutClose(hwo: HWAVEOUT) -> u32;
    pub fn waveOutReset(hwo: HWAVEOUT) -> u32;
    pub fn PlaySoundW(
        pszSound: *const u16, hmod: *mut c_void, fdwSound: u32,
    ) -> BOOL;
}
