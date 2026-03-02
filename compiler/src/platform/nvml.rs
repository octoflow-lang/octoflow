//! NVML bindings — dynamic loading of nvml.dll for real GPU metrics.
//!
//! Provides: GPU utilization, memory utilization, temperature, VRAM usage,
//! power draw, GPU name. Loaded lazily on first call.

use std::cell::Cell;
use std::ffi::c_void;

// ─── NVML types ──────────────────────────────────────────────

#[repr(C)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

#[repr(C)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

const NVML_SUCCESS: u32 = 0;
const NVML_TEMPERATURE_GPU: u32 = 0;

// ─── Function pointer types ──────────────────────────────────

type FnInit = unsafe extern "C" fn() -> u32;
#[allow(dead_code)]
type FnShutdown = unsafe extern "C" fn() -> u32;
type FnGetHandle = unsafe extern "C" fn(u32, *mut *mut c_void) -> u32;
type FnGetUtil = unsafe extern "C" fn(*mut c_void, *mut NvmlUtilization) -> u32;
type FnGetTemp = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> u32;
type FnGetMem = unsafe extern "C" fn(*mut c_void, *mut NvmlMemory) -> u32;
type FnGetName = unsafe extern "C" fn(*mut c_void, *mut u8, u32) -> u32;
type FnGetPower = unsafe extern "C" fn(*mut c_void, *mut u32) -> u32;
type FnGetClock = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> u32;

// ─── Cached state ────────────────────────────────────────────

extern "system" {
    fn LoadLibraryA(name: *const u8) -> usize;
    fn GetProcAddress(module: usize, name: *const u8) -> usize;
}

thread_local! {
    static NVML_DLL: Cell<usize> = Cell::new(0);
    static NVML_DEV: Cell<*mut c_void> = Cell::new(std::ptr::null_mut());
    static NVML_READY: Cell<bool> = Cell::new(false);
}

fn get_proc(dll: usize, name: &str) -> usize {
    let mut s = name.as_bytes().to_vec();
    s.push(0);
    unsafe { GetProcAddress(dll, s.as_ptr()) }
}

// ─── Init ────────────────────────────────────────────────────

fn ensure_init() -> bool {
    if NVML_READY.with(|r| r.get()) {
        return true;
    }

    let dll_name = b"nvml.dll\0";
    let dll = unsafe { LoadLibraryA(dll_name.as_ptr()) };
    if dll == 0 {
        return false;
    }
    NVML_DLL.with(|d| d.set(dll));

    let fn_init = get_proc(dll, "nvmlInit_v2");
    if fn_init == 0 {
        return false;
    }
    let ret = unsafe { (std::mem::transmute::<usize, FnInit>(fn_init))() };
    if ret != NVML_SUCCESS {
        return false;
    }

    let fn_handle = get_proc(dll, "nvmlDeviceGetHandleByIndex_v2");
    if fn_handle == 0 {
        return false;
    }
    let mut dev: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { (std::mem::transmute::<usize, FnGetHandle>(fn_handle))(0, &mut dev) };
    if ret != NVML_SUCCESS {
        return false;
    }
    NVML_DEV.with(|d| d.set(dev));
    NVML_READY.with(|r| r.set(true));
    true
}

// ─── Public API ──────────────────────────────────────────────

/// Initialize NVML. Returns 1.0 on success, 0.0 on failure.
pub fn nvml_init() -> f32 {
    if ensure_init() { 1.0 } else { 0.0 }
}

/// GPU utilization % (0-100).
pub fn nvml_gpu_util() -> f32 {
    if !ensure_init() { return 0.0; }
    let dll = NVML_DLL.with(|d| d.get());
    let dev = NVML_DEV.with(|d| d.get());
    let fp = get_proc(dll, "nvmlDeviceGetUtilizationRates");
    if fp == 0 { return 0.0; }
    let mut util = NvmlUtilization { gpu: 0, memory: 0 };
    let ret = unsafe { (std::mem::transmute::<usize, FnGetUtil>(fp))(dev, &mut util) };
    if ret == NVML_SUCCESS { util.gpu as f32 } else { 0.0 }
}

/// Memory utilization % (0-100).
pub fn nvml_mem_util() -> f32 {
    if !ensure_init() { return 0.0; }
    let dll = NVML_DLL.with(|d| d.get());
    let dev = NVML_DEV.with(|d| d.get());
    let fp = get_proc(dll, "nvmlDeviceGetUtilizationRates");
    if fp == 0 { return 0.0; }
    let mut util = NvmlUtilization { gpu: 0, memory: 0 };
    let ret = unsafe { (std::mem::transmute::<usize, FnGetUtil>(fp))(dev, &mut util) };
    if ret == NVML_SUCCESS { util.memory as f32 } else { 0.0 }
}

/// GPU temperature in Celsius.
pub fn nvml_temperature() -> f32 {
    if !ensure_init() { return 0.0; }
    let dll = NVML_DLL.with(|d| d.get());
    let dev = NVML_DEV.with(|d| d.get());
    let fp = get_proc(dll, "nvmlDeviceGetTemperature");
    if fp == 0 { return 0.0; }
    let mut temp: u32 = 0;
    let ret = unsafe { (std::mem::transmute::<usize, FnGetTemp>(fp))(dev, NVML_TEMPERATURE_GPU, &mut temp) };
    if ret == NVML_SUCCESS { temp as f32 } else { 0.0 }
}

/// VRAM used in MB.
pub fn nvml_vram_used() -> f32 {
    if !ensure_init() { return 0.0; }
    let dll = NVML_DLL.with(|d| d.get());
    let dev = NVML_DEV.with(|d| d.get());
    let fp = get_proc(dll, "nvmlDeviceGetMemoryInfo");
    if fp == 0 { return 0.0; }
    let mut mem = NvmlMemory { total: 0, free: 0, used: 0 };
    let ret = unsafe { (std::mem::transmute::<usize, FnGetMem>(fp))(dev, &mut mem) };
    if ret == NVML_SUCCESS { (mem.used / (1024 * 1024)) as f32 } else { 0.0 }
}

/// VRAM total in MB.
pub fn nvml_vram_total() -> f32 {
    if !ensure_init() { return 0.0; }
    let dll = NVML_DLL.with(|d| d.get());
    let dev = NVML_DEV.with(|d| d.get());
    let fp = get_proc(dll, "nvmlDeviceGetMemoryInfo");
    if fp == 0 { return 0.0; }
    let mut mem = NvmlMemory { total: 0, free: 0, used: 0 };
    let ret = unsafe { (std::mem::transmute::<usize, FnGetMem>(fp))(dev, &mut mem) };
    if ret == NVML_SUCCESS { (mem.total / (1024 * 1024)) as f32 } else { 0.0 }
}

/// Power draw in watts.
pub fn nvml_power() -> f32 {
    if !ensure_init() { return 0.0; }
    let dll = NVML_DLL.with(|d| d.get());
    let dev = NVML_DEV.with(|d| d.get());
    let fp = get_proc(dll, "nvmlDeviceGetPowerUsage");
    if fp == 0 { return 0.0; }
    let mut mw: u32 = 0;
    let ret = unsafe { (std::mem::transmute::<usize, FnGetPower>(fp))(dev, &mut mw) };
    if ret == NVML_SUCCESS { mw as f32 / 1000.0 } else { 0.0 }
}

/// GPU name (e.g., "NVIDIA GeForce GTX 1660 SUPER").
pub fn nvml_gpu_name() -> String {
    if !ensure_init() { return "Unknown GPU".to_string(); }
    let dll = NVML_DLL.with(|d| d.get());
    let dev = NVML_DEV.with(|d| d.get());
    let fp = get_proc(dll, "nvmlDeviceGetName");
    if fp == 0 { return "Unknown GPU".to_string(); }
    let mut buf = [0u8; 128];
    let ret = unsafe { (std::mem::transmute::<usize, FnGetName>(fp))(dev, buf.as_mut_ptr(), 128) };
    if ret == NVML_SUCCESS {
        let len = buf.iter().position(|&b| b == 0).unwrap_or(128);
        String::from_utf8_lossy(&buf[..len]).to_string()
    } else {
        "Unknown GPU".to_string()
    }
}

/// GPU clock speed in MHz.
pub fn nvml_clock_gpu() -> f32 {
    if !ensure_init() { return 0.0; }
    let dll = NVML_DLL.with(|d| d.get());
    let dev = NVML_DEV.with(|d| d.get());
    let fp = get_proc(dll, "nvmlDeviceGetClockInfo");
    if fp == 0 { return 0.0; }
    let mut mhz: u32 = 0;
    // NVML_CLOCK_GRAPHICS = 0
    let ret = unsafe { (std::mem::transmute::<usize, FnGetClock>(fp))(dev, 0, &mut mhz) };
    if ret == NVML_SUCCESS { mhz as f32 } else { 0.0 }
}
