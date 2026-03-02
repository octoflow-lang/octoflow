//! GDI text rendering with per-string caching and batch packing.
//!
//! Renders text via Win32 GDI (ClearType quality), extracts green channel
//! as alpha, caches results, and packs into a flat atlas for GPU upload.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;

use crate::platform::win32::*;

// ─── Types ──────────────────────────────────────────────────────

struct CachedRender {
    alpha: Vec<f32>,
    width: u32,
    height: u32,
}

struct TextEntry {
    alpha: Vec<f32>,
    width: u32,
    height: u32,
    offset: usize,
}

struct TextBatch {
    entries: Vec<TextEntry>,
}

// ─── Thread-local state ─────────────────────────────────────────

thread_local! {
    static TEXT_BATCH: RefCell<TextBatch> = RefCell::new(TextBatch { entries: Vec::new() });
    static RENDER_CACHE: RefCell<HashMap<(String, u32), CachedRender>> = RefCell::new(HashMap::new());
    static FONT_HEIGHT_CACHE: RefCell<HashMap<u32, u32>> = RefCell::new(HashMap::new());
    static CACHE_ORDER: RefCell<Vec<(String, u32)>> = RefCell::new(Vec::new());
    static GDI_DC: Cell<*mut c_void> = Cell::new(std::ptr::null_mut());
}

const MAX_CACHE_ENTRIES: usize = 1000;

// ─── Core GDI rendering ────────────────────────────────────────

/// Render text to alpha bitmap via GDI. Returns (alpha, width, height).
fn render_to_alpha(text: &str, font_size: u32) -> (Vec<f32>, u32, u32) {
    if text.is_empty() {
        return (vec![], 0, 0);
    }

    unsafe {
        let mem_dc = CreateCompatibleDC(std::ptr::null_mut());
        if mem_dc.is_null() {
            return (vec![], 0, 0);
        }

        // Create font — negative height = pixel size
        let w_face = to_wide("Segoe UI");
        let hfont = CreateFontW(
            -(font_size as i32), 0, 0, 0,
            FW_NORMAL, 0, 0, 0,
            DEFAULT_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH, w_face.as_ptr(),
        );
        if hfont.is_null() {
            DeleteDC(mem_dc);
            return (vec![], 0, 0);
        }
        let old_font = SelectObject(mem_dc, hfont as HGDIOBJ);

        // Measure text (UTF-16)
        let w_text = to_wide(text);
        let w_len = (w_text.len() - 1) as i32; // exclude null terminator
        let mut size = SIZE { cx: 0, cy: 0 };
        GetTextExtentPoint32W(
            mem_dc, w_text.as_ptr(), w_len, &mut size,
        );
        let w = if size.cx > 0 { size.cx as u32 } else { 1 };
        let h = if size.cy > 0 { size.cy as u32 } else { 1 };

        // Create 32bpp top-down DIB
        let mut bmi: BITMAPINFO = std::mem::zeroed();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = w as i32;
        bmi.bmiHeader.biHeight = -(h as i32); // top-down
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        let mut bits: *mut c_void = std::ptr::null_mut();
        let dib = CreateDIBSection(
            mem_dc, &bmi, DIB_RGB_COLORS, &mut bits,
            std::ptr::null_mut(), 0,
        );
        if dib.is_null() || bits.is_null() {
            SelectObject(mem_dc, old_font);
            DeleteObject(hfont as HGDIOBJ);
            DeleteDC(mem_dc);
            return (vec![], 0, 0);
        }
        let old_bmp = SelectObject(mem_dc, dib as HGDIOBJ);

        // Clear DIB to black
        let pixel_bytes = (w * h * 4) as usize;
        std::ptr::write_bytes(bits as *mut u8, 0, pixel_bytes);

        // Render white text on black background
        SetBkMode(mem_dc, TRANSPARENT_BK);
        SetTextColor(mem_dc, 0x00FFFFFF); // white (BGR)
        TextOutW(mem_dc, 0, 0, w_text.as_ptr(), w_len);

        // Extract green channel as alpha (0.0-255.0)
        let pixel_count = (w * h) as usize;
        let pixel_data = std::slice::from_raw_parts(bits as *const u8, pixel_bytes);
        let mut alpha = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            // BGRA layout: [0]=B, [1]=G, [2]=R, [3]=A
            // Green channel is best single-channel approximation of ClearType coverage
            let g = pixel_data[i * 4 + 1] as f32;
            alpha.push(g);
        }

        // Cleanup
        SelectObject(mem_dc, old_bmp);
        SelectObject(mem_dc, old_font);
        DeleteObject(dib as HGDIOBJ);
        DeleteObject(hfont as HGDIOBJ);
        DeleteDC(mem_dc);

        (alpha, w, h)
    }
}

/// Measure text width without caching the full render.
fn measure_text(text: &str, font_size: u32) -> (u32, u32) {
    if text.is_empty() {
        return (0, font_size);
    }

    unsafe {
        let mem_dc = CreateCompatibleDC(std::ptr::null_mut());
        if mem_dc.is_null() {
            return (0, font_size);
        }

        let w_face = to_wide("Segoe UI");
        let hfont = CreateFontW(
            -(font_size as i32), 0, 0, 0,
            FW_NORMAL, 0, 0, 0,
            DEFAULT_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH, w_face.as_ptr(),
        );
        if hfont.is_null() {
            DeleteDC(mem_dc);
            return (0, font_size);
        }
        let old_font = SelectObject(mem_dc, hfont as HGDIOBJ);

        let w_text = to_wide(text);
        let w_len = (w_text.len() - 1) as i32;
        let mut size = SIZE { cx: 0, cy: 0 };
        GetTextExtentPoint32W(
            mem_dc, w_text.as_ptr(), w_len, &mut size,
        );

        SelectObject(mem_dc, old_font);
        DeleteObject(hfont as HGDIOBJ);
        DeleteDC(mem_dc);

        (size.cx.max(0) as u32, size.cy.max(1) as u32)
    }
}

// ─── Public API ─────────────────────────────────────────────────

/// Clear the text batch for a new frame.
pub fn text_begin() {
    TEXT_BATCH.with(|b| {
        b.borrow_mut().entries.clear();
    });
}

/// Render text via GDI (cached), add to current batch. Returns entry index.
pub fn text_add(text: &str, font_size: u32) -> usize {
    let key = (text.to_string(), font_size);

    // Check cache
    let cached = RENDER_CACHE.with(|c| {
        c.borrow().get(&key).map(|cr| (cr.alpha.clone(), cr.width, cr.height))
    });

    let (alpha, width, height) = if let Some((a, w, h)) = cached {
        (a, w, h)
    } else {
        let (a, w, h) = render_to_alpha(text, font_size);

        // Evict oldest if cache is full
        CACHE_ORDER.with(|co| {
            let mut order = co.borrow_mut();
            if order.len() >= MAX_CACHE_ENTRIES {
                let evict_key = order.remove(0);
                RENDER_CACHE.with(|c| c.borrow_mut().remove(&evict_key));
            }
            order.push(key.clone());
        });

        RENDER_CACHE.with(|c| {
            c.borrow_mut().insert(key, CachedRender {
                alpha: a.clone(), width: w, height: h,
            });
        });
        (a, w, h)
    };

    TEXT_BATCH.with(|b| {
        let mut batch = b.borrow_mut();
        let idx = batch.entries.len();
        batch.entries.push(TextEntry {
            alpha, width, height, offset: 0, // set during text_atlas()
        });
        idx
    })
}

/// Pack all batch entries into a flat alpha array. Sets offsets.
pub fn text_atlas() -> Vec<f32> {
    TEXT_BATCH.with(|b| {
        let mut batch = b.borrow_mut();
        let total_size: usize = batch.entries.iter().map(|e| e.alpha.len()).sum();
        let mut atlas = Vec::with_capacity(total_size.max(1));

        // Ensure at least one element (empty heap may cause issues)
        if total_size == 0 {
            atlas.push(0.0);
            return atlas;
        }

        let mut offset = 0usize;
        for entry in batch.entries.iter_mut() {
            entry.offset = offset;
            atlas.extend_from_slice(&entry.alpha);
            offset += entry.alpha.len();
        }
        atlas
    })
}

/// Get width of batch entry at index.
pub fn text_w(idx: usize) -> u32 {
    TEXT_BATCH.with(|b| {
        b.borrow().entries.get(idx).map_or(0, |e| e.width)
    })
}

/// Get height of batch entry at index.
pub fn text_h(idx: usize) -> u32 {
    TEXT_BATCH.with(|b| {
        b.borrow().entries.get(idx).map_or(0, |e| e.height)
    })
}

/// Get offset of batch entry in packed atlas.
pub fn text_off(idx: usize) -> usize {
    TEXT_BATCH.with(|b| {
        b.borrow().entries.get(idx).map_or(0, |e| e.offset)
    })
}

/// Standalone text width measurement (pixels). Uses cache.
pub fn text_width(text: &str, font_size: u32) -> u32 {
    let key = (text.to_string(), font_size);

    // Check render cache first
    let cached_w = RENDER_CACHE.with(|c| {
        c.borrow().get(&key).map(|cr| cr.width)
    });
    if let Some(w) = cached_w {
        return w;
    }

    let (w, _h) = measure_text(text, font_size);
    w
}

/// Get line height for a font size (pixels). Cached.
pub fn text_height(font_size: u32) -> u32 {
    let cached = FONT_HEIGHT_CACHE.with(|c| {
        c.borrow().get(&font_size).copied()
    });
    if let Some(h) = cached {
        return h;
    }

    let (_w, h) = measure_text("Hg", font_size);
    FONT_HEIGHT_CACHE.with(|c| {
        c.borrow_mut().insert(font_size, h);
    });
    h
}
