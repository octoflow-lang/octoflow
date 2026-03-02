//! CPU metrics via Win32 GetSystemTimes — OS-boundary concern.
//!
//! Provides: overall CPU utilization %, logical core count.
//! Uses delta between two samples (idle vs total time).

use std::cell::Cell;

// ─── Win32 types ───────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone)]
struct Filetime {
    lo: u32,
    hi: u32,
}

impl Filetime {
    fn to_u64(self) -> u64 {
        (self.hi as u64) << 32 | self.lo as u64
    }
}

extern "system" {
    fn GetSystemTimes(
        idle: *mut Filetime,
        kernel: *mut Filetime,
        user: *mut Filetime,
    ) -> i32;
}

// ─── SYSTEM_INFO for core count ────────────────────────────

#[repr(C)]
struct SystemInfo {
    _arch: u16,
    _reserved: u16,
    _page_size: u32,
    _min_app_addr: usize,
    _max_app_addr: usize,
    _active_mask: usize,
    num_processors: u32,
    _proc_type: u32,
    _alloc_gran: u32,
    _proc_level: u16,
    _proc_rev: u16,
}

extern "system" {
    fn GetNativeSystemInfo(info: *mut SystemInfo);
}

// ─── Cached previous sample ────────────────────────────────

thread_local! {
    static PREV_IDLE: Cell<u64> = Cell::new(0);
    static PREV_TOTAL: Cell<u64> = Cell::new(0);
    static INITIALIZED: Cell<bool> = Cell::new(false);
}

fn sample() -> (u64, u64) {
    let mut idle = Filetime { lo: 0, hi: 0 };
    let mut kernel = Filetime { lo: 0, hi: 0 };
    let mut user = Filetime { lo: 0, hi: 0 };
    let ret = unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) };
    if ret == 0 {
        return (0, 0);
    }
    let idle_t = idle.to_u64();
    // kernel time includes idle time
    let total = kernel.to_u64() + user.to_u64();
    (idle_t, total)
}

// ─── Public API ────────────────────────────────────────────

/// CPU utilization % (0-100). Returns delta since last call.
/// First call seeds the baseline and returns 0.
pub fn cpu_util() -> f32 {
    let (idle, total) = sample();
    if total == 0 {
        return 0.0;
    }

    let prev_idle = PREV_IDLE.with(|c| c.get());
    let prev_total = PREV_TOTAL.with(|c| c.get());
    let was_init = INITIALIZED.with(|c| c.get());

    PREV_IDLE.with(|c| c.set(idle));
    PREV_TOTAL.with(|c| c.set(total));
    INITIALIZED.with(|c| c.set(true));

    if !was_init {
        return 0.0;
    }

    let d_idle = idle.wrapping_sub(prev_idle) as f64;
    let d_total = total.wrapping_sub(prev_total) as f64;
    if d_total <= 0.0 {
        return 0.0;
    }

    let usage = (1.0 - d_idle / d_total) * 100.0;
    usage.clamp(0.0, 100.0) as f32
}

/// Number of logical CPU cores.
pub fn cpu_count() -> f32 {
    let mut info: SystemInfo = unsafe { std::mem::zeroed() };
    unsafe { GetNativeSystemInfo(&mut info) };
    info.num_processors as f32
}
