//! Compiler: AST → SPIR-V pattern selection and GPU/CPU dispatch.

pub mod grammar;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use octoflow_parser::ast::{Arg, BinOp, CompareOp, Expr, ExternFn, PrintSegment, Program, ScalarExpr, StageCall, Statement};
use octoflow_vulkan::{MapOp, BinaryOp, ReduceOp, FusedOp, TemporalOp};

// ── CPU Thread Pool (Primitive 7) ──

/// Task sent to the thread pool workers.
enum PoolTask {
    FileRead { path: String, result_id: u32 },
    FileWrite { path: String, data: Vec<u8> },
    Shutdown,
}

/// Result returned from pool workers.
enum AsyncResult {
    FileData(Vec<u8>),
    Done,
    Error(String),
}

/// OctoPress compression state.
struct OctoPressState {
    block_size: u32,
}

/// OctoPress streaming decompression handle.
struct OctoPressStream {
    compressed: Vec<f32>,
    method: u32,
    original_count: u32,
    block_size: u32,
    cursor: usize,
}

/// CPU thread pool for async file I/O and SPIR-V compilation.
struct LoomPool {
    workers: Vec<std::thread::JoinHandle<()>>,
    task_tx: std::sync::mpsc::Sender<PoolTask>,
    result_rx: std::sync::mpsc::Receiver<(u32, AsyncResult)>,
}

thread_local! {
    /// Session-level stream cache: maps user key → GPU output Vec<f32>.
    /// `cache("key") expr` hits this on second evaluation — zero GPU dispatch.
    static STREAM_CACHE: std::cell::RefCell<std::collections::HashMap<String, Vec<f32>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());

    /// Monotonic seed counter for random_stream().
    /// Each call gets a unique seed so multiple random_stream calls in one
    /// script produce independent sequences (xorshift64* CPU RNG).
    static RANDOM_SEED: std::cell::Cell<u32> = std::cell::Cell::new(0x9e3779b9);

    /// Side-channel for returning arrays from user functions.
    /// `return arr_name` inside a ScalarFnDecl sets this if arr_name is an array.
    /// The LetDecl handler checks it after eval_scalar to capture the array.
    static RETURNED_ARRAY: std::cell::RefCell<Option<Vec<crate::Value>>> =
        std::cell::RefCell::new(None);

    /// Side-channel for returning hashmaps from user functions.
    /// `return map_name` inside a ScalarFnDecl sets this if map_name is a hashmap.
    /// The LetDecl handler checks it after eval_scalar to capture the hashmap.
    static RETURNED_MAP: std::cell::RefCell<Option<std::collections::HashMap<String, crate::Value>>> =
        std::cell::RefCell::new(None);

    /// R-05: Side-channel for propagating mutable scalar writes back from user functions.
    /// After execute_user_fn completes, modified mutable scalars are placed here.
    /// The statement-level caller (execute_loop_body) applies them to its mutable scalars.
    static SCALAR_WRITEBACK: std::cell::RefCell<Option<Vec<(String, crate::Value)>>> =
        std::cell::RefCell::new(None);
}

use crate::{CliError, Value};

/// Maximum recursion depth for user-defined function calls.
/// Prevents stack overflow from unbounded/runaway recursion.
const MAX_RECURSION_DEPTH: usize = 50;

// Security flags — set per execution run, deny by default.
thread_local! {
    static ALLOW_READ: RefCell<crate::PermScope> = RefCell::new(crate::PermScope::Deny);
    static ALLOW_WRITE: RefCell<crate::PermScope> = RefCell::new(crate::PermScope::Deny);
    static ALLOW_NET: RefCell<crate::PermScope> = RefCell::new(crate::PermScope::Deny);
    static ALLOW_EXEC: RefCell<crate::PermScope> = RefCell::new(crate::PermScope::Deny);
    static ALLOW_FFI: Cell<bool> = Cell::new(false);
    /// GPU memory quota in bytes (0 = unlimited).
    static GPU_MAX_BYTES: Cell<u64> = Cell::new(0);
    /// Running total of GPU memory allocated in this execution.
    static GPU_ALLOCATED_BYTES: Cell<u64> = Cell::new(0);
    /// Current recursion depth for user-defined function calls.
    static RECURSION_DEPTH: Cell<usize> = Cell::new(0);
    /// Transitive import guard — canonical paths of already-imported modules.
    static IMPORTED_PATHS: RefCell<std::collections::HashSet<String>> = RefCell::new(std::collections::HashSet::new());
    /// Registered extern functions — populated by ExternBlock, used by call_extern_fn.
    static EXTERN_REGISTRY: RefCell<HashMap<String, ExternFnDecl>> = RefCell::new(HashMap::new());

    /// GPU-native array storage: either raw f32 on CPU or VkBuffer resident in VRAM.
    /// Phase 78: raw f32 bypass. Phase 79: GPU-resident VkBuffer handles.
    static GPU_ARRAYS: RefCell<HashMap<String, GpuArrayStorage>> = RefCell::new(HashMap::new());

    /// Raw pointer to the VulkanCompute device for the current execution run.
    /// Set in execute(), cleared on exit. Enables fast GPU→CPU downloads via
    /// HOST_CACHED staging instead of slow HOST_VISIBLE direct reads.
    static GPU_DEVICE_PTR: Cell<usize> = Cell::new(0);

    /// Whether the current GPU supports f16 arithmetic + 16-bit storage buffers.
    /// Set during execute() from VulkanCompute::supports_f16.
    static GPU_SUPPORTS_F16: Cell<bool> = Cell::new(false);

    /// File byte cache for GGUF model files — avoids re-reading from disk.
    /// gguf_cache_file(path) loads once, gguf_load_tensor uses cached bytes.
    static FILE_CACHE: RefCell<HashMap<String, Vec<u8>>> = RefCell::new(HashMap::new());

    /// Dequantized tensor cache — avoids re-dequantizing the same weight tensor.
    /// Key: "path:tensor_name", Value: dequantized Vec<f32>.
    /// Only caches full tensor loads (not row extractions).
    static TENSOR_CACHE: RefCell<HashMap<String, Vec<f32>>> = RefCell::new(HashMap::new());

    /// KV cache for transformer inference — stored in Rust to avoid .flow overhead.
    static KV_CACHE: RefCell<Option<InferKvCache>> = RefCell::new(None);

    /// GPU buffer cache — dequantized weight tensors uploaded to VRAM.
    /// Key: "path:tensor_name", Value: GpuBuffer (VkBuffer in HOST_VISIBLE memory).
    /// Weight data stays resident across tokens — only input vectors are re-uploaded.
    static GPU_BUFFER_CACHE: RefCell<HashMap<String, octoflow_vulkan::GpuBuffer>> =
        RefCell::new(HashMap::new());

    /// Staging buffer handles for double-buffered streaming.
    /// Each handle owns: HOST_VISIBLE VkBuffer + VkDeviceMemory + VkFence + mapped ptr + size.
    static STAGING_HANDLES: RefCell<HashMap<u32, StagingHandle>> =
        RefCell::new(HashMap::new());
    static STAGING_NEXT_ID: Cell<u32> = Cell::new(1);

    /// Background prefetch thread for async layer weight loading.
    /// Returns Vec<(cache_key, dequanted_data, is_small)>.
    static PREFETCH_THREAD: RefCell<Option<std::thread::JoinHandle<
        Result<Vec<(String, Vec<f32>, bool)>, String>
    >>> = RefCell::new(None);

    /// Verbose inference logging (--verbose flag). When false, gguf_infer_layer
    /// suppresses per-weight debug output (1,120+ eprintln per token).
    static VERBOSE_INFER: Cell<bool> = Cell::new(false);

    /// Tracks which layers are fully GPU-resident (all large tensors in GPU_BUFFER_CACHE).
    /// Key: layer_idx, Value: true = all on GPU, false = all on CPU/TENSOR_CACHE.
    /// Set by decomposed_load_layer/gguf_prefetch_complete, cleared by gguf_evict_layer.
    static LAYER_RESIDENCY: RefCell<HashMap<usize, bool>> = RefCell::new(HashMap::new());

    /// Total bytes currently in GPU_BUFFER_CACHE (for VRAM budget decisions from .flow).
    /// Incremented on GPU_BUFFER_CACHE insert, decremented on remove.
    static GPU_CACHE_BYTES: Cell<u64> = Cell::new(0);

    /// R-23: GPU dispatch timer (CPU-side Instant).
    static GPU_TIMER_START: Cell<Option<std::time::Instant>> = Cell::new(None);

    /// GPU VM instances. Key = VM ID, Value = VmHandle.
    static GPU_VMS: RefCell<HashMap<u32, octoflow_vulkan::vm::VmHandle>> = RefCell::new(HashMap::new());
    static VM_NEXT_ID: Cell<u32> = Cell::new(1);
    /// Mailbox VM IDs (VMs created via loom_mailbox, used for ring buffer protocol).
    static MAILBOX_VMS: RefCell<std::collections::HashSet<u32>> = RefCell::new(std::collections::HashSet::new());
    /// Parked VMs — pooled for reuse by loom_auto_spawn. Key = VM ID, Value = VmHandle.
    static PARKED_VMS: RefCell<HashMap<u32, octoflow_vulkan::vm::VmHandle>> = RefCell::new(HashMap::new());
    /// Resource budget: max active + parked VMs (default 64, 0 = unlimited).
    static LOOM_MAX_VMS: Cell<u32> = Cell::new(64);
    /// Resource budget: VRAM budget in bytes (0 = unlimited).
    static LOOM_VRAM_BUDGET: Cell<u64> = Cell::new(0);
    /// Resource budget: estimated VRAM usage in bytes across all active + parked VMs.
    static LOOM_VRAM_USED: Cell<u64> = Cell::new(0);
    /// Wall-clock time (microseconds) of the last loom_run/loom_dispatch call.
    static LAST_DISPATCH_US: Cell<u64> = Cell::new(0);

    /// GPU VM staged dispatch ops. Key = VM ID, Value = list of VmOp.
    static VM_STAGED_OPS: RefCell<HashMap<u32, Vec<octoflow_vulkan::vm::VmOp>>> = RefCell::new(HashMap::new());

    /// GPU VM compiled programs. Key = program ID, Value = VmProgram.
    static VM_PROGRAMS: RefCell<HashMap<u32, octoflow_vulkan::vm::VmProgram>> = RefCell::new(HashMap::new());
    static VM_PROG_NEXT_ID: Cell<u32> = Cell::new(1);
    /// Maps program ID → VM ID so vm_shutdown can drop owned programs while device is alive.
    static VM_PROG_OWNERS: RefCell<HashMap<u32, u32>> = RefCell::new(HashMap::new());

    /// Loom homeostasis: pace delay in microseconds per VM (0 = no pacing).
    static LOOM_PACE_US: RefCell<HashMap<u32, u64>> = RefCell::new(HashMap::new());
    /// Loom homeostasis: baseline dispatch time in microseconds per VM.
    static LOOM_BASELINE_US: RefCell<HashMap<u32, u64>> = RefCell::new(HashMap::new());
    /// Loom homeostasis: total dispatch count per VM.
    static LOOM_DISPATCH_COUNT: RefCell<HashMap<u32, u64>> = RefCell::new(HashMap::new());
    /// Loom homeostasis: total paced dispatch count per VM.
    static LOOM_PACED_COUNT: RefCell<HashMap<u32, u64>> = RefCell::new(HashMap::new());

    /// Loom homeostasis: accumulated pacing debt (microseconds) settled at vm_present.
    static LOOM_PACE_DEBT_US: Cell<u64> = Cell::new(0);

    /// Async present: pending framebuffer read from previous frame.
    static PENDING_FB_READ: RefCell<Option<octoflow_vulkan::vm::PendingFbRead>> = RefCell::new(None);
    /// Async present: cached RGB data from previous frame (ready to blit).
    static PENDING_FB_RGB: RefCell<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> = RefCell::new(None);

    /// SPIR-V file cache — avoids re-reading the same .spv files from disk every dispatch.
    static SPIRV_FILE_CACHE: RefCell<HashMap<String, Vec<u8>>> = RefCell::new(HashMap::new());

    /// Batched uploads: pending staging DMA copies per VM, consumed at loom_build.
    static VM_PENDING_UPLOADS: RefCell<HashMap<u32, Vec<octoflow_vulkan::vm::PendingUpload>>> = RefCell::new(HashMap::new());

    /// SPIR-V prefetch: background threads reading .spv files from disk.
    static SPIRV_PREFETCH: RefCell<HashMap<String, std::thread::JoinHandle<Result<Vec<u8>, String>>>> = RefCell::new(HashMap::new());

    /// Output capture sink — when active, print() output goes here instead of stdout.
    /// Used by `octoflow chat` to capture generated program output.
    static CAPTURED_OUTPUT: RefCell<Option<String>> = RefCell::new(None);

    /// Token streaming callback — when set, chat_emit_token() calls this to stream
    /// tokens to stderr for real-time display during LLM generation.
    static TOKEN_CALLBACK: RefCell<Option<Box<dyn FnMut(&str)>>> = RefCell::new(None);

    /// CPU thread pool (Primitive 7). None until loom_threads() is called.
    static LOOM_THREAD_POOL: RefCell<Option<LoomPool>> = RefCell::new(None);
    /// Async task results collected from pool workers.
    static LOOM_ASYNC_RESULTS: RefCell<HashMap<u32, AsyncResult>> = RefCell::new(HashMap::new());
    /// Monotonic ID for async task handles.
    static LOOM_ASYNC_NEXT_ID: Cell<u32> = Cell::new(1);
    /// OctoPress compression state. None until octopress_init() is called.
    static OCTOPRESS_STATE: RefCell<Option<OctoPressState>> = RefCell::new(None);
    /// OctoPress streaming handles. Key = stream ID, Value = OctoPressStream.
    static OCTOPRESS_STREAMS: RefCell<HashMap<u32, OctoPressStream>> = RefCell::new(HashMap::new());
    /// Monotonic ID for OctoPress stream handles.
    static OCTOPRESS_STREAM_NEXT_ID: Cell<u32> = Cell::new(1);
}

/// Start capturing print() output. Subsequent print() calls append to internal buffer.
pub fn capture_output_start() {
    CAPTURED_OUTPUT.with(|co| {
        *co.borrow_mut() = Some(String::new());
    });
}

/// Take the captured output and stop capturing. Returns the captured string.
pub fn capture_output_take() -> String {
    CAPTURED_OUTPUT.with(|co| {
        co.borrow_mut().take().unwrap_or_default()
    })
}

/// Stop capturing without returning the output.
pub fn capture_output_stop() {
    CAPTURED_OUTPUT.with(|co| {
        *co.borrow_mut() = None;
    });
}

/// Set the token streaming callback for chat_emit_token().
pub fn set_token_callback(cb: Box<dyn FnMut(&str)>) {
    TOKEN_CALLBACK.with(|tc| {
        *tc.borrow_mut() = Some(cb);
    });
}

/// Clear the token streaming callback.
pub fn clear_token_callback() {
    TOKEN_CALLBACK.with(|tc| {
        *tc.borrow_mut() = None;
    });
}

/// Internal: write output to capture buffer or stdout.
fn runtime_println(s: &str) {
    let captured = CAPTURED_OUTPUT.with(|co| {
        let mut borrow = co.borrow_mut();
        if let Some(ref mut buf) = *borrow {
            buf.push_str(s);
            buf.push('\n');
            true
        } else {
            false
        }
    });
    if !captured {
        println!("{}", s);
    }
}

/// KV cache for transformer inference.
#[allow(dead_code)]
struct InferKvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    n_layer: usize,
    max_seq: usize,
    kv_dim: usize,
}

/// Staging buffer for double-buffered streaming.
/// Holds CPU-side data ready to be uploaded to GPU_BUFFER_CACHE.
#[allow(dead_code)]
struct StagingHandle {
    data: Vec<f32>,
    size_bytes: usize,
    /// Background thread for async file I/O (None = idle or completed).
    io_thread: Option<std::thread::JoinHandle<Result<Vec<f32>, String>>>,
}

/// Ensure a tensor is loaded into TENSOR_CACHE. Returns Ok(()) if already cached.
fn ensure_tensor_cached(
    path: &str,
    model_map: &HashMap<String, Value>,
    tname: &str,
) -> Result<(), CliError> {
    let cache_key = format!("{}:{}", path, tname);
    let in_cache = TENSOR_CACHE.with(|tc| tc.borrow().contains_key(&cache_key));
    if in_cache {
        return Ok(());
    }
    let prefix = format!("t.{}", tname);
    let total_count = model_map.get(&format!("{}.count", prefix))
        .and_then(|v| v.as_float().ok())
        .unwrap_or(0.0) as usize;
    if total_count == 0 {
        return Err(CliError::Runtime(format!("ensure_tensor_cached: tensor '{}' not found or zero count", tname)));
    }
    let tensor_type = model_map.get(&format!("{}.type", prefix))
        .and_then(|v| v.as_float().ok())
        .unwrap_or(0.0) as u32;
    let byte_size = match tensor_type {
        0 => total_count * 4,
        1 => total_count * 2,
        12 => (total_count / 256) * 144,
        13 => (total_count / 256) * 176,
        14 => (total_count / 256) * 210,
        _ => total_count * 4,
    };
    let ds_buf = model_map.get("_ds_buf")
        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
    let hdr_buf = model_map.get("_hdr_buf")
        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
    let off_pos = model_map.get(&format!("{}.off_pos", prefix))
        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
    let ds_ptr = mem_table_get_ptr(ds_buf)?;
    let data_start = unsafe { (ds_ptr as *const u64).read_unaligned() };
    let hdr_ptr = mem_table_get_ptr(hdr_buf)?;
    let tensor_offset = unsafe { (hdr_ptr.add(off_pos) as *const u64).read_unaligned() };
    let file_offset = data_start + tensor_offset;

    let raw: Vec<u8> = FILE_CACHE.with(|cache| -> Result<Vec<u8>, CliError> {
        let c = cache.borrow();
        if let Some(cached_bytes) = c.get(path) {
            let start = file_offset as usize;
            let end = start + byte_size;
            if end <= cached_bytes.len() {
                return Ok(cached_bytes[start..end].to_vec());
            }
            return Err(CliError::Runtime(format!(
                "ensure_tensor_cached: offset {} + size {} exceeds file size {}",
                start, byte_size, cached_bytes.len()
            )));
        }
        drop(c);
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(path)
            .map_err(|e| CliError::Io(format!("ensure_tensor_cached: {}", e)))?;
        file.seek(SeekFrom::Start(file_offset))
            .map_err(|e| CliError::Io(format!("ensure_tensor_cached: seek: {}", e)))?;
        let mut buf = vec![0u8; byte_size];
        file.read_exact(&mut buf)
            .map_err(|e| CliError::Io(format!("ensure_tensor_cached: read: {}", e)))?;
        Ok(buf)
    })?;

    let dequanted: Vec<f32> = match tensor_type {
        0 => raw.chunks_exact(4).take(total_count).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        1 => raw.chunks_exact(2).take(total_count).map(|c| gguf_f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        12 => {
            // Q4_K: try GPU first, fall back to CPU
            match gpu_dequant_q4k_batch(&raw, total_count) {
                Ok(mut dequanted) => { dequanted.truncate(total_count); dequanted }
                Err(_) => {
                    let mut out = Vec::with_capacity(total_count + 256);
                    for block in raw.chunks(144) {
                        if block.len() < 144 { break; }
                        gguf_dequant_q4k_block(block, &mut out);
                    }
                    out.truncate(total_count);
                    out
                }
            }
        }
        13 => {
            // Q5_K: CPU dequantization
            let mut out = Vec::with_capacity(total_count + 256);
            for block in raw.chunks(176) {
                if block.len() < 176 { break; }
                gguf_dequant_q5k_block(block, &mut out);
            }
            out.truncate(total_count);
            out
        }
        14 => {
            // Q6_K: try GPU first, fall back to CPU
            match gpu_dequant_q6k_batch(&raw, total_count) {
                Ok(mut dequanted) => { dequanted.truncate(total_count); dequanted }
                Err(_) => {
                    let mut out = Vec::with_capacity(total_count + 256);
                    for block in raw.chunks(210) {
                        if block.len() < 210 { break; }
                        gguf_dequant_q6k_block(block, &mut out);
                    }
                    out.truncate(total_count);
                    out
                }
            }
        }
        _ => return Err(CliError::Runtime(format!("ensure_tensor_cached: unsupported type {}", tensor_type))),
    };
    TENSOR_CACHE.with(|tc| tc.borrow_mut().insert(cache_key, dequanted));
    Ok(())
}

/// Fast CPU matvec: result[i] = dot(w[i*K..(i+1)*K], input).
/// Uses unsafe pointer arithmetic to eliminate bounds checks.
#[inline(never)]
fn fast_matvec(weight: &[f32], input: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; m];
    unsafe {
        let wp = weight.as_ptr();
        let ip = input.as_ptr();
        let rp = result.as_mut_ptr();
        for i in 0..m {
            let row = wp.add(i * k);
            let mut sum = 0.0f32;
            let mut j = 0;
            // Process 4 elements at a time for better vectorization
            let k4 = k & !3;
            while j < k4 {
                sum += *row.add(j) * *ip.add(j)
                     + *row.add(j + 1) * *ip.add(j + 1)
                     + *row.add(j + 2) * *ip.add(j + 2)
                     + *row.add(j + 3) * *ip.add(j + 3);
                j += 4;
            }
            while j < k {
                sum += *row.add(j) * *ip.add(j);
                j += 1;
            }
            *rp.add(i) = sum;
        }
    }
    result
}

/// Fast CPU RMSNorm: out[i] = (x[i] / rms) * weight[i]
fn fast_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        sum_sq += x[i] * x[i];
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
    out
}

/// In-place RoPE (Rotary Position Embedding).
fn fast_rope(vec: &mut [f32], pos: usize, head_dim: usize, n_heads: usize, rope_theta: f64) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let half = head_dim / 2;
        for i in 0..half {
            let theta = pos as f64 / rope_theta.powf(2.0 * i as f64 / head_dim as f64);
            let cos_t = theta.cos() as f32;
            let sin_t = theta.sin() as f32;
            let x = vec[base + 2 * i];
            let y = vec[base + 2 * i + 1];
            vec[base + 2 * i] = x * cos_t - y * sin_t;
            vec[base + 2 * i + 1] = x * sin_t + y * cos_t;
        }
    }
}

/// Pre-compiled matvec SPIR-V kernel (emitted by emit_matvec.flow).
/// Bindings: 0=A (M*K matrix), 1=B (K vector), 2=C (M result).
/// Push constants: pc[0]=M (rows as float), pc[1]=K (cols as float).
static MATVEC_SPV: &[u8] = include_bytes!("../../../stdlib/loom/kernels/nn/matvec.spv");

/// GPU-accelerated matvec with buffer caching.
///
/// Weight tensor is uploaded to GPU VRAM on first call and cached in GPU_BUFFER_CACHE.
/// Input vector is uploaded fresh each call (small — typically 896-4864 floats = 3-19KB).
/// Falls back to fast_matvec if no GPU device is available.
///
/// Returns result[i] = dot(weight[i*k..(i+1)*k], input) for i in 0..m.
fn gpu_cached_matvec(
    cache_key: &str,
    weight: &[f32],
    input: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
    if device_ptr == 0 || m < 4 {
        return fast_matvec(weight, input, m, k);
    }
    let gpu = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };

    // Check/populate GPU weight buffer cache
    let weight_ref = GPU_BUFFER_CACHE.with(|gc| {
        let cache = gc.borrow();
        if let Some(buf) = cache.get(cache_key) {
            return Some(buf.as_ref());
        }
        None
    });

    let weight_buf_ref = if let Some(r) = weight_ref {
        r
    } else {
        // Upload weight to GPU and cache
        match octoflow_vulkan::upload_buffer(gpu, weight) {
            Ok(buf) => {
                let r = buf.as_ref();
                GPU_BUFFER_CACHE.with(|gc| {
                    gc.borrow_mut().insert(cache_key.to_string(), buf);
                });
                r
            }
            Err(_) => return fast_matvec(weight, input, m, k),
        }
    };

    // Upload input vector (small, fresh each call)
    let input_buf = match octoflow_vulkan::upload_buffer(gpu, input) {
        Ok(b) => b,
        Err(_) => return fast_matvec(weight, input, m, k),
    };

    // Dispatch matvec kernel: binding 0=weight(M*K), 1=input(K), 2=output(M)
    let pc = [m as f32, k as f32];
    let out_size = (m * 4) as u64;
    let wg_x = ((m + 255) / 256) as u32;
    match octoflow_vulkan::dispatch_resident_pc(
        gpu,
        MATVEC_SPV,
        &[weight_buf_ref, input_buf.as_ref()],
        &[(out_size, m)],
        wg_x,
        &pc,
    ) {
        Ok(mut out_bufs) => {
            let out_buf = out_bufs.remove(0);
            octoflow_vulkan::download_buffer(gpu, &out_buf).unwrap_or_else(|_| fast_matvec(weight, input, m, k))
        }
        Err(_) => fast_matvec(weight, input, m, k),
    }
}

/// Evict a tensor from both TENSOR_CACHE and GPU_BUFFER_CACHE.
fn evict_tensor(cache_key: &str) {
    TENSOR_CACHE.with(|tc| tc.borrow_mut().remove(cache_key));
    GPU_BUFFER_CACHE.with(|gc| {
        if let Some(buf) = gc.borrow_mut().remove(cache_key) {
            GPU_CACHE_BYTES.with(|c| {
                let prev = c.get();
                c.set(prev.saturating_sub(buf.len() as u64 * 4));
            });
        }
    });
}

/// Evict a tensor from TENSOR_CACHE only. GPU_BUFFER_CACHE stays resident in VRAM.
/// Use this when you want to free system RAM but keep weights on GPU for reuse.
fn evict_tensor_cache_only(cache_key: &str) {
    TENSOR_CACHE.with(|tc| tc.borrow_mut().remove(cache_key));
}

/// Ensure a weight tensor is in GPU_BUFFER_CACHE (VRAM only, no TENSOR_CACHE copy).
/// Dequants from GGUF on disk → uploads to GPU → drops CPU data.
/// This saves system RAM by not double-storing weights in TENSOR_CACHE + GPU_BUFFER_CACHE.
fn ensure_gpu_buffer_cached(
    path: &str,
    model_map: &HashMap<String, Value>,
    tname: &str,
) -> Result<(), CliError> {
    let cache_key = format!("{}:{}", path, tname);
    // Already in GPU cache?
    let in_gpu = GPU_BUFFER_CACHE.with(|gc| gc.borrow().contains_key(&cache_key));
    if in_gpu {
        return Ok(());
    }
    // Already in TENSOR_CACHE? Upload to GPU and remove CPU copy.
    let from_tensor_cache = TENSOR_CACHE.with(|tc| {
        let cache = tc.borrow();
        cache.get(&cache_key).map(|w| w.clone())
    });
    if let Some(weight) = from_tensor_cache {
        let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
        if device_ptr != 0 {
            let gpu = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
            if let Ok(buf) = octoflow_vulkan::upload_buffer(gpu, &weight) {
                GPU_CACHE_BYTES.with(|c| c.set(c.get() + buf.len() as u64 * 4));
                GPU_BUFFER_CACHE.with(|gc| gc.borrow_mut().insert(cache_key.clone(), buf));
                // Keep TENSOR_CACHE backup — prevents zero-output if GPU dispatch fails later.
                // RAM is cheap; garbage output from split-brain is catastrophic.
                return Ok(());
            }
        }
        return Ok(()); // Fallback: keep in TENSOR_CACHE
    }
    // Not cached anywhere — dequant from disk, upload to GPU, drop CPU data
    let prefix = format!("t.{}", tname);
    let total_count = model_map.get(&format!("{}.count", prefix))
        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
    if total_count == 0 {
        return Err(CliError::Runtime(format!("ensure_gpu_buffer_cached: tensor '{}' not found", tname)));
    }
    let tensor_type = model_map.get(&format!("{}.type", prefix))
        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as u32;
    let byte_size = match tensor_type {
        0 => total_count * 4,
        1 => total_count * 2,
        12 => (total_count / 256) * 144,
        13 => (total_count / 256) * 176,
        14 => (total_count / 256) * 210,
        _ => total_count * 4,
    };
    let ds_buf = model_map.get("_ds_buf")
        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
    let hdr_buf = model_map.get("_hdr_buf")
        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
    let off_pos = model_map.get(&format!("{}.off_pos", prefix))
        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
    let ds_ptr = mem_table_get_ptr(ds_buf)?;
    let data_start = unsafe { (ds_ptr as *const u64).read_unaligned() };
    let hdr_ptr = mem_table_get_ptr(hdr_buf)?;
    let tensor_offset = unsafe { (hdr_ptr.add(off_pos) as *const u64).read_unaligned() };
    let file_offset = data_start + tensor_offset;
    let raw: Vec<u8> = FILE_CACHE.with(|cache| -> Result<Vec<u8>, CliError> {
        let c = cache.borrow();
        if let Some(cached_bytes) = c.get(path) {
            let start = file_offset as usize;
            let end = start + byte_size;
            if end <= cached_bytes.len() {
                return Ok(cached_bytes[start..end].to_vec());
            }
        }
        drop(c);
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(path)
            .map_err(|e| CliError::Io(format!("ensure_gpu_buffer_cached: {}", e)))?;
        file.seek(SeekFrom::Start(file_offset))
            .map_err(|e| CliError::Io(format!("ensure_gpu_buffer_cached: seek: {}", e)))?;
        let mut buf = vec![0u8; byte_size];
        file.read_exact(&mut buf)
            .map_err(|e| CliError::Io(format!("ensure_gpu_buffer_cached: read: {}", e)))?;
        Ok(buf)
    })?;
    let dequanted: Vec<f32> = match tensor_type {
        0 => raw.chunks_exact(4).take(total_count).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        1 => raw.chunks_exact(2).take(total_count).map(|c| gguf_f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        12 => {
            match gpu_dequant_q4k_batch(&raw, total_count) {
                Ok(mut d) => { d.truncate(total_count); d }
                Err(_) => {
                    let mut out = Vec::with_capacity(total_count + 256);
                    for block in raw.chunks(144) {
                        if block.len() < 144 { break; }
                        gguf_dequant_q4k_block(block, &mut out);
                    }
                    out.truncate(total_count); out
                }
            }
        }
        13 => {
            // Q5_K: CPU dequantization
            let mut out = Vec::with_capacity(total_count + 256);
            for block in raw.chunks(176) {
                if block.len() < 176 { break; }
                gguf_dequant_q5k_block(block, &mut out);
            }
            out.truncate(total_count); out
        }
        14 => {
            match gpu_dequant_q6k_batch(&raw, total_count) {
                Ok(mut d) => { d.truncate(total_count); d }
                Err(_) => {
                    let mut out = Vec::with_capacity(total_count + 256);
                    for block in raw.chunks(210) {
                        if block.len() < 210 { break; }
                        gguf_dequant_q6k_block(block, &mut out);
                    }
                    out.truncate(total_count); out
                }
            }
        }
        _ => return Err(CliError::Runtime(format!("ensure_gpu_buffer_cached: unsupported type {}", tensor_type))),
    };
    // Upload to GPU VRAM and cache — keep CPU backup in TENSOR_CACHE
    let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
    if device_ptr != 0 {
        let gpu = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
        if let Ok(buf) = octoflow_vulkan::upload_buffer(gpu, &dequanted) {
            GPU_CACHE_BYTES.with(|c| c.set(c.get() + buf.len() as u64 * 4));
            GPU_BUFFER_CACHE.with(|gc| gc.borrow_mut().insert(cache_key.clone(), buf));
        }
    }
    // Always keep CPU backup in TENSOR_CACHE (prevents split-brain zero-output)
    TENSOR_CACHE.with(|tc| tc.borrow_mut().insert(cache_key, dequanted));
    Ok(())
}

/// Dispatch matvec from GPU_BUFFER_CACHE (weight already in VRAM).
/// Returns None if weight not in GPU cache (caller should fall back to CPU).
fn gpu_matvec_from_cache(cache_key: &str, input: &[f32], m: usize, k: usize) -> Option<Vec<f32>> {
    let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
    if device_ptr == 0 || m < 4 {
        return None;
    }
    let gpu = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
    let weight_ref = GPU_BUFFER_CACHE.with(|gc| {
        gc.borrow().get(cache_key).map(|b| b.as_ref())
    })?;
    let input_buf = octoflow_vulkan::upload_buffer(gpu, input).ok()?;
    let pc = [m as f32, k as f32];
    let out_size = (m * 4) as u64;
    let wg_x = ((m + 255) / 256) as u32;
    match octoflow_vulkan::dispatch_resident_pc(
        gpu, MATVEC_SPV, &[weight_ref, input_buf.as_ref()],
        &[(out_size, m)], wg_x, &pc,
    ) {
        Ok(mut out_bufs) => {
            let out_buf = out_bufs.remove(0);
            Some(octoflow_vulkan::download_buffer(gpu, &out_buf).unwrap_or_else(|_| vec![0.0; m]))
        }
        Err(_) => None,
    }
}

/// CPU fallback matvec when GPU dispatch is unavailable.
/// Loads weight from TENSOR_CACHE (must already be cached).
fn fast_matvec_fallback(path: &str, tname: &str, input: &[f32], m: usize, k: usize) -> Vec<f32> {
    let cache_key = format!("{}:{}", path, tname);
    TENSOR_CACHE.with(|tc| {
        let cache = tc.borrow();
        if let Some(w) = cache.get(&cache_key) {
            return fast_matvec(w, input, m, k);
        }
        // Weight not in TENSOR_CACHE (GPU-only path) — download from GPU
        GPU_BUFFER_CACHE.with(|gc| {
            let gc_cache = gc.borrow();
            if let Some(buf) = gc_cache.get(&cache_key) {
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr != 0 {
                    let gpu = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                    if let Ok(weight) = octoflow_vulkan::download_buffer(gpu, buf) {
                        return fast_matvec(&weight, input, m, k);
                    }
                }
            }
            if VERBOSE_INFER.with(|c| c.get()) {
                eprintln!("[WARNING] fast_matvec_fallback: weight '{}' not in TENSOR_CACHE or GPU_BUFFER_CACHE — returning zeros ({})", cache_key, m);
            }
            vec![0.0f32; m]
        })
    })
}

/// Storage for GPU arrays: either CPU-side f32 or GPU-resident VkBuffer.
enum GpuArrayStorage {
    Cpu(Vec<f32>),
    Resident(octoflow_vulkan::GpuBuffer),
}

/// Result from eval_array_fn — either Value-based, GPU-native f32, or GPU-resident VkBuffer.
enum ArrayResult {
    Values(Vec<Value>),
    GpuFloats(Vec<f32>),
    Resident(octoflow_vulkan::GpuBuffer),
}

/// Flush pending deferred GPU dispatches using the thread-local device pointer.
/// No-op if no device or no pending dispatches.
fn flush_pending_gpu() {
    GPU_DEVICE_PTR.with(|c| {
        let ptr = c.get();
        if ptr != 0 {
            let gpu = unsafe { &*(ptr as *const octoflow_vulkan::VulkanCompute) };
            let _ = octoflow_vulkan::flush_pending(gpu);
        }
    });
}

// GPU array helpers — thread-local access without parameter threading.
fn gpu_array_get(name: &str) -> Option<Vec<f32>> {
    flush_pending_gpu(); // Ensure deferred dispatches are complete before download
    GPU_ARRAYS.with(|ga| {
        ga.borrow().get(name).map(|storage| match storage {
            GpuArrayStorage::Cpu(v) => v.clone(),
            GpuArrayStorage::Resident(buf) => gpu_download_fast(buf),
        })
    })
}
fn gpu_array_insert(name: String, data: Vec<f32>) {
    // Check GPU quota (CPU-resident arrays still count toward the limit)
    let bytes = (data.len() * std::mem::size_of::<f32>()) as u64;
    if let Err(_) = check_gpu_quota(bytes) {
        // Quota check failure is best-effort for CPU arrays — log but don't block
    }
    GPU_ARRAYS.with(|ga| ga.borrow_mut().insert(name, GpuArrayStorage::Cpu(data)));
}
fn gpu_array_insert_resident(name: String, buf: octoflow_vulkan::GpuBuffer) {
    GPU_ARRAYS.with(|ga| ga.borrow_mut().insert(name, GpuArrayStorage::Resident(buf)));
}
/// Get a GpuBufferRef if the array is GPU-resident (for zero-copy chaining).
fn gpu_array_get_resident(name: &str) -> Option<octoflow_vulkan::GpuBufferRef> {
    GPU_ARRAYS.with(|ga| {
        ga.borrow().get(name).and_then(|s| match s {
            GpuArrayStorage::Resident(buf) => Some(buf.as_ref()),
            GpuArrayStorage::Cpu(_) => None,
        })
    })
}
fn gpu_array_len(name: &str) -> Option<usize> {
    GPU_ARRAYS.with(|ga| ga.borrow().get(name).map(|s| match s {
        GpuArrayStorage::Cpu(v) => v.len(),
        GpuArrayStorage::Resident(buf) => buf.len(),
    }))
}
fn gpu_array_index(name: &str, idx: usize) -> Option<f32> {
    flush_pending_gpu(); // Ensure deferred dispatches are complete before element read
    GPU_ARRAYS.with(|ga| ga.borrow().get(name).and_then(|s| match s {
        GpuArrayStorage::Cpu(v) => v.get(idx).copied(),
        GpuArrayStorage::Resident(buf) => buf.read_element(idx),
    }))
}
fn gpu_array_has(name: &str) -> bool {
    GPU_ARRAYS.with(|ga| ga.borrow().contains_key(name))
}
fn gpu_array_clear() {
    flush_pending_gpu(); // Ensure deferred dispatches are complete before clearing
    GPU_ARRAYS.with(|ga| ga.borrow_mut().clear());
}
/// Materialize a GPU array into the Value-based arrays map (for non-GPU operations).
fn gpu_array_materialize(name: &str, arrays: &mut HashMap<String, Vec<Value>>) -> bool {
    flush_pending_gpu(); // Ensure deferred dispatches are complete before materialize
    let data = GPU_ARRAYS.with(|ga| ga.borrow_mut().remove(name));
    if let Some(storage) = data {
        let floats = match storage {
            GpuArrayStorage::Cpu(v) => v,
            GpuArrayStorage::Resident(buf) => gpu_download_fast(&buf),
        };
        arrays.insert(name.to_string(), floats.into_iter().map(Value::Float).collect());
        true
    } else {
        false
    }
}
// ── Shared LetDecl helpers (extracted from 5 duplicate handlers) ─────────────

/// Open a video/image from raw bytes. Returns (handle, width, height, frame_count, fps).
/// Used by video_open() in all execution contexts.
fn video_open_from_bytes(bytes: Vec<u8>) -> Result<(VideoHandle, u32, u32, usize, f32), CliError> {
    let handle = if bytes.len() >= 6 && (&bytes[0..3] == b"GIF") {
        let gif = crate::image_io::gif::decode(&bytes)
            .map_err(|e| CliError::Compile(format!("video_open(): {}", e)))?;
        let fc = gif.frames.len();
        let fps = if fc > 0 && gif.frames[0].delay_ms > 0 {
            1000.0 / gif.frames[0].delay_ms as f32
        } else { 10.0 };
        let decoded: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = gif.frames.iter().map(|f| {
            let total = gif.width as usize * gif.height as usize;
            let mut r = Vec::with_capacity(total);
            let mut g = Vec::with_capacity(total);
            let mut b = Vec::with_capacity(total);
            for i in 0..total {
                r.push(f.rgb[i*3] as f32);
                g.push(f.rgb[i*3+1] as f32);
                b.push(f.rgb[i*3+2] as f32);
            }
            (r, g, b)
        }).collect();
        VideoHandle::Gif { width: gif.width, height: gif.height, frame_count: fc, fps, decoded_frames: decoded }
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"AVI " {
        let info = crate::image_io::avi::parse(&bytes)
            .map_err(|e| CliError::Compile(format!("video_open(): {}", e)))?;
        VideoHandle::Avi { data: bytes, info }
    } else {
        let (r, g, b, w, h) = crate::image_io::decode_image_bytes(&bytes)
            .map_err(|e| CliError::Compile(format!("video_open(): {}", e)))?;
        VideoHandle::Gif { width: w, height: h, frame_count: 1, fps: 0.0, decoded_frames: vec![(r, g, b)] }
    };
    let (w, h, fc, fps) = match &handle {
        VideoHandle::Gif { width, height, frame_count, fps, .. } => (*width, *height, *frame_count, *fps),
        VideoHandle::Avi { info, .. } => (info.width, info.height, info.frame_count, info.fps),
    };
    Ok((handle, w, h, fc, fps))
}

/// Insert video_open scalars and register handle. Returns the handle ID.
/// Open a video directly from a file path — avoids loading bytes into Vec<Value>.
fn video_open_file_bytes(path: &str) -> Result<(VideoHandle, u32, u32, usize, f32), CliError> {
    let bytes = std::fs::read(path)
        .map_err(|e| CliError::Io(format!("video_open_file(\"{}\"): {}", path, e)))?;
    video_open_from_bytes(bytes)
}

fn video_open_register(handle: VideoHandle, name: &str, w: u32, h: u32, fc: usize, fps: f32,
                       scalars: &mut HashMap<String, Value>) -> u32 {
    let id = NEXT_VIDEO_ID.with(|c| { let v = c.get(); c.set(v + 1); v });
    VIDEO_HANDLES.with(|vh| vh.borrow_mut().insert(id, handle));
    scalars.insert(name.to_string(), Value::Float(id as f32));
    scalars.insert(format!("{}.width", name), Value::Float(w as f32));
    scalars.insert(format!("{}.height", name), Value::Float(h as f32));
    scalars.insert(format!("{}.frames", name), Value::Float(fc as f32));
    scalars.insert(format!("{}.fps", name), Value::Float(fps));
    id
}

/// Decode a video frame by handle ID and frame index. Returns (R, G, B) channel arrays.
fn video_frame_decode(handle_id: u32, frame_idx: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), CliError> {
    VIDEO_HANDLES.with(|vh| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), CliError> {
        let handles = vh.borrow();
        let handle = handles.get(&handle_id)
            .ok_or_else(|| CliError::Compile(format!("video_frame(): invalid handle {}", handle_id)))?;
        match handle {
            VideoHandle::Gif { decoded_frames, frame_count, .. } => {
                if frame_idx >= *frame_count {
                    return Err(CliError::Compile(format!("video_frame(): frame {} out of range (0..{})", frame_idx, frame_count)));
                }
                let (r, g, b) = &decoded_frames[frame_idx];
                Ok((r.clone(), g.clone(), b.clone()))
            }
            VideoHandle::Avi { data, info } => {
                if frame_idx >= info.frame_count {
                    return Err(CliError::Compile(format!("video_frame(): frame {} out of range (0..{})", frame_idx, info.frame_count)));
                }
                let (offset, len) = info.frame_offsets[frame_idx];
                let frame_data = &data[offset..offset + len];
                let (r, g, b, _w, _h) = crate::image_io::decode_image_bytes(frame_data)
                    .map_err(|e| CliError::Compile(format!("video_frame(): {}", e)))?;
                Ok((r, g, b))
            }
        }
    })
}

/// Extract byte array for video_open from arrays map.
fn video_open_extract_bytes(arr_name: &str, arrays: &HashMap<String, Vec<Value>>) -> Result<Vec<u8>, CliError> {
    let arr = arrays.get(arr_name)
        .ok_or_else(|| CliError::Compile(format!("video_open(): array '{}' not found", arr_name)))?;
    Ok(arr.iter().map(|v| v.as_float().unwrap_or(0.0) as u8).collect())
}

/// Format a GPU array for print interpolation.
fn gpu_array_format(name: &str) -> Option<String> {
    flush_pending_gpu(); // Ensure deferred dispatches are complete before format
    GPU_ARRAYS.with(|ga| {
        ga.borrow().get(name).map(|storage| {
            let arr = match storage {
                GpuArrayStorage::Cpu(v) => v.clone(),
                GpuArrayStorage::Resident(buf) => gpu_download_fast(buf),
            };
            let items: Vec<String> = arr.iter().map(|v| format!("{}", v)).collect();
            format!("[{}]", items.join(", "))
        })
    })
}

/// Fast GPU→CPU download using HOST_CACHED staging (DMA copy → cached read).
/// Falls back to direct HOST_VISIBLE read if no GPU device is available.
fn gpu_download_fast(buf: &octoflow_vulkan::GpuBuffer) -> Vec<f32> {
    let ptr = GPU_DEVICE_PTR.with(|c| c.get());
    if ptr != 0 {
        let gpu = unsafe { &*(ptr as *const octoflow_vulkan::VulkanCompute) };
        octoflow_vulkan::download_buffer_fast(gpu, buf).unwrap_or_else(|_| buf.download().unwrap_or_default())
    } else {
        buf.download().unwrap_or_default()
    }
}

// ── MEM_TABLE: Handle-based raw memory for FFI struct construction ────
// Slot 0 = null sentinel (never allocated). External pointers use size=0.

struct MemBlock {
    ptr: *mut u8,
    size: usize,
    layout: std::alloc::Layout,
}

thread_local! {
    static MEM_TABLE: RefCell<Vec<Option<MemBlock>>> = RefCell::new(vec![None]); // slot 0 = null
}

fn mem_table_alloc(size: usize) -> Result<f32, CliError> {
    if size == 0 {
        return Err(CliError::Runtime("mem_alloc: size must be > 0".into()));
    }
    let layout = std::alloc::Layout::from_size_align(size, 16)
        .map_err(|_| CliError::Runtime(format!("mem_alloc: invalid size {}", size)))?;
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(CliError::Runtime(format!("mem_alloc: allocation failed for {} bytes", size)));
    }
    MEM_TABLE.with(|t| {
        let mut table = t.borrow_mut();
        // Find a free slot or push a new one
        for (i, slot) in table.iter_mut().enumerate().skip(1) {
            if slot.is_none() {
                *slot = Some(MemBlock { ptr, size, layout });
                return Ok(i as f32);
            }
        }
        let idx = table.len();
        table.push(Some(MemBlock { ptr, size, layout }));
        Ok(idx as f32)
    })
}

fn mem_table_free(handle: usize) -> Result<(), CliError> {
    MEM_TABLE.with(|t| {
        let mut table = t.borrow_mut();
        if handle == 0 || handle >= table.len() {
            return Err(CliError::Runtime(format!("mem_free: invalid handle {}", handle)));
        }
        match table[handle].take() {
            Some(block) => {
                if block.size == 0 {
                    return Err(CliError::Runtime("mem_free: cannot free external pointer".into()));
                }
                unsafe { std::alloc::dealloc(block.ptr, block.layout); }
                Ok(())
            }
            None => Err(CliError::Runtime(format!("mem_free: handle {} already freed", handle))),
        }
    })
}

fn mem_table_get_ptr(handle: usize) -> Result<*mut u8, CliError> {
    if handle == 0 { return Ok(std::ptr::null_mut()); }
    MEM_TABLE.with(|t| {
        let table = t.borrow();
        match &table.get(handle) {
            Some(Some(block)) => Ok(block.ptr),
            _ => Err(CliError::Runtime(format!("mem: invalid handle {}", handle))),
        }
    })
}

fn mem_table_get_size(handle: usize) -> Result<usize, CliError> {
    MEM_TABLE.with(|t| {
        let table = t.borrow();
        match &table.get(handle) {
            Some(Some(block)) => Ok(block.size),
            _ => Err(CliError::Runtime(format!("mem_size: invalid handle {}", handle))),
        }
    })
}

/// Store an external pointer in the table (size=0 means not owned).
fn mem_table_store_external(ptr: *mut u8) -> f32 {
    if ptr.is_null() { return 0.0; }
    MEM_TABLE.with(|t| {
        let mut table = t.borrow_mut();
        let layout = std::alloc::Layout::from_size_align(1, 1).unwrap(); // dummy
        let idx = table.len();
        table.push(Some(MemBlock { ptr, size: 0, layout }));
        idx as f32
    })
}

fn mem_table_clear() {
    MEM_TABLE.with(|t| {
        let mut table = t.borrow_mut();
        for slot in table.iter_mut().skip(1) {
            if let Some(block) = slot.take() {
                if block.size > 0 {
                    unsafe { std::alloc::dealloc(block.ptr, block.layout); }
                }
            }
        }
        table.truncate(1); // keep null sentinel
    });
}

// ── LIB_CACHE: Cached library handles for FFI (Phase 72b) ────────────
// Avoids load/free per call — critical for 45+ Vulkan API calls per dispatch.

thread_local! {
    static LIB_CACHE: RefCell<HashMap<String, usize>> = RefCell::new(HashMap::new());
}

fn lib_cache_clear() {
    LIB_CACHE.with(|c| {
        let cache = c.borrow();
        for &handle in cache.values() {
            ffi_os::free_library(handle);
        }
        drop(cache);
        c.borrow_mut().clear();
    });
}

#[allow(dead_code)]
fn check_read_permission() -> Result<(), CliError> {
    ALLOW_READ.with(|c| {
        match &*c.borrow() {
            crate::PermScope::AllowAll => Ok(()),
            crate::PermScope::Deny => Err(CliError::Security("file read not permitted — use --allow-read to enable".into())),
            crate::PermScope::AllowScoped(_) => Err(CliError::Security(
                "file read requires path validation — use --allow-read (without path) or call with specific path".into()
            )),
        }
    })
}

fn check_read_permission_for(path: &str) -> Result<(), CliError> {
    ALLOW_READ.with(|c| {
        let scope = c.borrow();
        if scope.allows_path(path) { Ok(()) }
        else {
            match &*scope {
                crate::PermScope::Deny => Err(CliError::Security("file read not permitted — use --allow-read to enable".into())),
                crate::PermScope::AllowScoped(_) => Err(CliError::Security(format!(
                    "file read not permitted for '{}' — path is outside allowed directories", path
                ))),
                crate::PermScope::AllowAll => Ok(()),
            }
        }
    })
}

#[allow(dead_code)]
fn check_write_permission() -> Result<(), CliError> {
    ALLOW_WRITE.with(|c| {
        match &*c.borrow() {
            crate::PermScope::AllowAll => Ok(()),
            crate::PermScope::Deny => Err(CliError::Security("file write not permitted — use --allow-write to enable".into())),
            crate::PermScope::AllowScoped(_) => Err(CliError::Security(
                "file write requires path validation — use --allow-write (without path) or call with specific path".into()
            )),
        }
    })
}

fn check_write_permission_for(path: &str) -> Result<(), CliError> {
    ALLOW_WRITE.with(|c| {
        let scope = c.borrow();
        if scope.allows_path(path) { Ok(()) }
        else {
            match &*scope {
                crate::PermScope::Deny => Err(CliError::Security("file write not permitted — use --allow-write to enable".into())),
                crate::PermScope::AllowScoped(_) => Err(CliError::Security(format!(
                    "file write not permitted for '{}' — path is outside allowed directories", path
                ))),
                crate::PermScope::AllowAll => Ok(()),
            }
        }
    })
}

fn check_net_permission() -> Result<(), CliError> {
    ALLOW_NET.with(|c| {
        if c.borrow().is_allowed() { Ok(()) }
        else { Err(CliError::Security("network access not permitted — use --allow-net to enable".into())) }
    })
}

fn check_exec_permission() -> Result<(), CliError> {
    ALLOW_EXEC.with(|c| {
        if c.borrow().is_allowed() { Ok(()) }
        else { Err(CliError::Security("command execution not permitted — use --allow-exec to enable".into())) }
    })
}

fn check_ffi_permission() -> Result<(), CliError> {
    if !ALLOW_FFI.with(|c| c.get()) {
        return Err(CliError::Security("FFI calls not permitted — use --allow-ffi to enable".into()));
    }
    Ok(())
}

/// Check GPU memory quota before allocating `bytes` on the GPU.
/// Returns Ok(()) if within quota or no quota is set, Err otherwise.
fn check_gpu_quota(bytes: u64) -> Result<(), CliError> {
    let max = GPU_MAX_BYTES.with(|c| c.get());
    if max == 0 { return Ok(()); }
    let current = GPU_ALLOCATED_BYTES.with(|c| c.get());
    if current + bytes > max {
        return Err(CliError::Gpu(format!(
            "GPU memory quota exceeded: requesting {} bytes, already allocated {} of {} bytes (--gpu-max-mb {})",
            bytes, current, max, max / (1024 * 1024)
        )));
    }
    GPU_ALLOCATED_BYTES.with(|c| c.set(current + bytes));
    Ok(())
}

/// Write array contents as raw bytes to a binary file.
/// Each f32 element is truncated to u8 (0-255).
fn do_write_bytes(path: &str, arr: &[Value]) -> Result<(), CliError> {
    let bytes: Vec<u8> = arr.iter().map(|v| {
        let f = v.as_float().unwrap_or(0.0);
        f as u8
    }).collect();

    // SPIR-V batch emit: cache immediately + async write via pool if available
    if path.ends_with(".spv") {
        SPIRV_FILE_CACHE.with(|cache| {
            cache.borrow_mut().insert(path.to_string(), bytes.clone());
        });
        let async_written = LOOM_THREAD_POOL.with(|pool| {
            if let Some(ref pool) = *pool.borrow() {
                pool.task_tx.send(PoolTask::FileWrite {
                    path: path.to_string(),
                    data: bytes.clone(),
                }).ok();
                true
            } else {
                false
            }
        });
        if async_written {
            return Ok(()); // file write submitted to pool
        }
    }

    std::fs::write(path, &bytes)
        .map_err(|e| CliError::Io(format!("write_bytes(\"{}\"): {}", path, e)))?;
    Ok(())
}

/// Registered extern function declaration (for FFI calls at runtime).
#[derive(Debug, Clone)]
struct ExternFnDecl {
    library: String,
    params: Vec<String>,
    return_type: Option<String>,
}

/// Resolve a library name to the OS-appropriate path.
fn resolve_library_name(name: &str) -> String {
    // Already has an extension — use as-is
    if name.ends_with(".dll") || name.ends_with(".so") || name.contains(".so.")
        || name.ends_with(".dylib")
    {
        return name.to_string();
    }
    // Standard C library special-case
    if name == "c" {
        if cfg!(target_os = "windows") { return "msvcrt.dll".to_string(); }
        if cfg!(target_os = "macos")   { return "libc.dylib".to_string(); }
        return "libc.so.6".to_string(); // Linux
    }
    // Add platform extension
    if cfg!(target_os = "windows") {
        format!("{}.dll", name)
    } else if cfg!(target_os = "macos") {
        format!("lib{}.dylib", name)
    } else {
        format!("lib{}.so", name)
    }
}

// ── Raw OS dynamic-loading (no external crates) ──────────────────────────────

#[cfg(target_os = "windows")]
mod ffi_os {
    extern "system" {
        pub fn LoadLibraryA(lp_lib_file_name: *const u8) -> usize;
        pub fn GetProcAddress(h_module: usize, lp_proc_name: *const u8) -> usize;
        pub fn FreeLibrary(h_lib_module: usize) -> i32;
    }

    pub fn load_library(name: &str) -> Option<usize> {
        let mut s = name.as_bytes().to_vec();
        s.push(0);
        let handle = unsafe { LoadLibraryA(s.as_ptr()) };
        if handle == 0 { None } else { Some(handle) }
    }

    pub fn get_proc_address(handle: usize, sym: &str) -> Option<usize> {
        let mut s = sym.as_bytes().to_vec();
        s.push(0);
        let addr = unsafe { GetProcAddress(handle, s.as_ptr()) };
        if addr == 0 { None } else { Some(addr) }
    }

    pub fn free_library(handle: usize) {
        unsafe { FreeLibrary(handle); }
    }
}

#[cfg(not(target_os = "windows"))]
mod ffi_os {
    extern "C" {
        pub fn dlopen(filename: *const u8, flags: i32) -> usize;
        pub fn dlsym(handle: usize, symbol: *const u8) -> usize;
        pub fn dlclose(handle: usize) -> i32;
    }
    const RTLD_NOW: i32 = 0x0002;
    const RTLD_GLOBAL: i32 = 0x0100;

    pub fn load_library(name: &str) -> Option<usize> {
        let mut s = name.as_bytes().to_vec();
        s.push(0);
        let handle = unsafe { dlopen(s.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if handle == 0 { None } else { Some(handle) }
    }

    pub fn get_proc_address(handle: usize, sym: &str) -> Option<usize> {
        let mut s = sym.as_bytes().to_vec();
        s.push(0);
        let addr = unsafe { dlsym(handle, s.as_ptr()) };
        if addr == 0 { None } else { Some(addr) }
    }

    pub fn free_library(handle: usize) {
        unsafe { dlclose(handle); }
    }
}

/// Call an extern C function by address with up to 4 u64 arguments → u64 result.
///
/// All OctoFlow FFI values are transmitted as u64:
///   - float → f64 bit-cast or truncated to u64
///   - string → pointer to null-terminated bytes
///   - ptr/handle → raw u64 address
unsafe fn call_fn_ptr(addr: usize, args: &[u64]) -> u64 {
    // We use transmute to call with the correct arity.
    // The callee must actually match — UB if signature mismatches.
    type F0 = unsafe extern "C" fn() -> u64;
    type F1 = unsafe extern "C" fn(u64) -> u64;
    type F2 = unsafe extern "C" fn(u64, u64) -> u64;
    type F3 = unsafe extern "C" fn(u64, u64, u64) -> u64;
    type F4 = unsafe extern "C" fn(u64, u64, u64, u64) -> u64;
    type F5 = unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64;
    type F6 = unsafe extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64;
    type F7 = unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64;
    type F8  = unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64;
    type F9  = unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64;
    type F10 = unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64;
    type F11 = unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64;
    type F12 = unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64;
    if args.len() > 12 {
        panic!("FFI call_fn_ptr: too many arguments ({}, max 12)", args.len());
    }
    match args.len() {
        0 => std::mem::transmute::<usize, F0>(addr)(),
        1 => std::mem::transmute::<usize, F1>(addr)(args[0]),
        2 => std::mem::transmute::<usize, F2>(addr)(args[0], args[1]),
        3 => std::mem::transmute::<usize, F3>(addr)(args[0], args[1], args[2]),
        4 => std::mem::transmute::<usize, F4>(addr)(args[0], args[1], args[2], args[3]),
        5 => std::mem::transmute::<usize, F5>(addr)(args[0], args[1], args[2], args[3], args[4]),
        6 => std::mem::transmute::<usize, F6>(addr)(args[0], args[1], args[2], args[3], args[4], args[5]),
        7 => std::mem::transmute::<usize, F7>(addr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6]),
        8 => std::mem::transmute::<usize, F8>(addr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]),
        9 => std::mem::transmute::<usize, F9>(addr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]),
        10 => std::mem::transmute::<usize, F10>(addr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]),
        11 => std::mem::transmute::<usize, F11>(addr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]),
        _ => std::mem::transmute::<usize, F12>(addr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]),
    }
}

/// Register extern fn declarations into the thread-local registry.
/// Called when an ExternBlock statement is executed.
fn register_extern_block(library: &str, functions: &[ExternFn]) {
    EXTERN_REGISTRY.with(|r| {
        let mut reg = r.borrow_mut();
        for f in functions {
            reg.insert(f.name.clone(), ExternFnDecl {
                library: library.to_string(),
                params: f.params.iter().map(|p| p.type_name.clone()).collect(),
                return_type: f.return_type.clone(),
            });
        }
    });
}

/// Look up whether a function name is a registered extern fn.
fn is_extern_fn(fn_name: &str) -> bool {
    EXTERN_REGISTRY.with(|r| r.borrow().contains_key(fn_name))
}

/// Call a registered extern function with OctoFlow Values (looks up in EXTERN_REGISTRY).
///
/// String arguments are encoded as *const u8 pointers (caller must keep buffer
/// alive across the call — this is safe because the buffer is a local Vec<u8>).
fn call_extern_fn(fn_name: &str, arg_vals: &[Value]) -> Result<Value, CliError> {
    check_ffi_permission()?;

    let decl = EXTERN_REGISTRY.with(|r| r.borrow().get(fn_name).cloned())
        .ok_or_else(|| CliError::Runtime(format!("FFI: '{}' not in extern registry", fn_name)))?;

    if arg_vals.len() > 12 {
        return Err(CliError::Runtime(format!(
            "FFI '{}': maximum 12 arguments supported, got {}",
            fn_name, arg_vals.len()
        )));
    }

    let lib_path = resolve_library_name(&decl.library);
    let handle = LIB_CACHE.with(|c| {
        let cache = c.borrow();
        if let Some(&h) = cache.get(&lib_path) { return Ok(h); }
        drop(cache);
        let h = ffi_os::load_library(&lib_path).ok_or_else(|| {
            CliError::Runtime(format!("FFI: cannot load library '{}'", lib_path))
        })?;
        c.borrow_mut().insert(lib_path.clone(), h);
        Ok(h)
    })?;

    let addr = ffi_os::get_proc_address(handle, fn_name).ok_or_else(|| {
        CliError::Runtime(format!("FFI: symbol '{}' not found in '{}'", fn_name, lib_path))
    })?;

    // Encode args → u64 using declared param types for correct encoding.
    let mut str_bufs: Vec<Vec<u8>> = Vec::new();
    let mut c_args: Vec<u64> = Vec::with_capacity(arg_vals.len());
    for (i, v) in arg_vals.iter().enumerate() {
        let param_type = decl.params.get(i).map(|s| s.as_str()).unwrap_or("");
        match v {
            Value::None => c_args.push(0),
            Value::Float(f) => {
                match param_type {
                    "f32" | "float" => c_args.push(((*f) as f64).to_bits()),
                    "f64" | "double" => c_args.push(((*f) as f64).to_bits()),
                    "ptr" | "handle" => {
                        // MEM_TABLE handle → raw pointer
                        let h = *f as isize;
                        if h <= 0 {
                            c_args.push(0); // null pointer
                        } else {
                            let ptr = mem_table_get_ptr(h as usize)?;
                            c_args.push(ptr as u64);
                        }
                    }
                    // Integer types + default: direct cast (not f64.to_bits!)
                    _ => c_args.push(*f as u64),
                }
            }
            Value::Str(s) => {
                let mut buf = s.as_bytes().to_vec();
                buf.push(0);
                c_args.push(buf.as_ptr() as u64);
                str_bufs.push(buf); // keep alive
            }
            Value::Int(i) => {
                match param_type {
                    "f32" | "float" => c_args.push(((*i as f32) as f64).to_bits()),
                    "f64" | "double" => c_args.push((*i as f64).to_bits()),
                    "ptr" | "handle" => {
                        if *i <= 0 {
                            c_args.push(0);
                        } else {
                            let ptr = mem_table_get_ptr(*i as usize)?;
                            c_args.push(ptr as u64);
                        }
                    }
                    _ => c_args.push(*i as u64),
                }
            }
            Value::Map(_) => c_args.push(0),
        }
    }

    let result = unsafe { call_fn_ptr(addr, &c_args) };

    // str_bufs kept alive until here
    drop(str_bufs);
    // Library handle cached in LIB_CACHE — freed at session end via lib_cache_clear()

    // Convert result based on declared return type
    let ret_val = match decl.return_type.as_deref() {
        Some("void") | None => Value::Float(0.0),
        Some("f32") | Some("float") => Value::Float(f32::from_bits(result as u32)),
        Some("f64") | Some("double") => Value::Float(f64::from_bits(result) as f32),
        Some("string") => {
            // result is a *const u8 — read null-terminated string
            if result == 0 {
                Value::Str(String::new())
            } else {
                let mut bytes = Vec::new();
                let mut p = result as *const u8;
                unsafe {
                    while *p != 0 {
                        bytes.push(*p);
                        p = p.add(1);
                    }
                }
                Value::Str(String::from_utf8_lossy(&bytes).into_owned())
            }
        }
        Some("ptr") | Some("handle") => {
            // Store raw pointer in MEM_TABLE, return handle
            Value::Float(mem_table_store_external(result as *mut u8))
        }
        // integer types — truncate to declared width before converting to f32
        Some("u32") => Value::Float((result as u32) as f32),
        Some("i32") => Value::Float((result as i32) as f32),
        Some("u16") => Value::Float((result as u16) as f32),
        Some("i16") => Value::Float((result as i16) as f32),
        Some("u8") => Value::Float((result as u8) as f32),
        Some("i8") => Value::Float((result as i8) as f32),
        // u64, i64, or unknown → direct cast (may lose precision for large values)
        _ => Value::Float(result as f32),
    };

    Ok(ret_val)
}

// ── Pure-Rust base64 (Phase 45 — replaces base64 crate) ──────────────

const B64_ALPHABET: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode_pure(input: &[u8]) -> String {
    let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 { out.push(B64_ALPHABET[((n >> 6) & 63) as usize] as char); }
        else                { out.push('='); }
        if chunk.len() > 2 { out.push(B64_ALPHABET[(n & 63) as usize] as char); }
        else                { out.push('='); }
    }
    out
}

fn base64_decode_pure(input: &str) -> Result<Vec<u8>, String> {
    fn decode_char(c: u8) -> Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+'         => Ok(62),
            b'/'         => Ok(63),
            _            => Err(format!("invalid base64 character '{}'", c as char)),
        }
    }
    let input = input.as_bytes();
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut i = 0;
    while i < input.len() {
        // Skip whitespace
        while i < input.len() && matches!(input[i], b' '|b'\t'|b'\n'|b'\r') { i += 1; }
        if i >= input.len() { break; }
        let a = decode_char(input[i])?;                          i += 1;
        while i < input.len() && matches!(input[i], b' '|b'\t'|b'\n'|b'\r') { i += 1; }
        let b = decode_char(input[i])?;                          i += 1;
        while i < input.len() && matches!(input[i], b' '|b'\t'|b'\n'|b'\r') { i += 1; }
        let c = if i < input.len() && input[i] != b'=' { let v = decode_char(input[i])?; i += 1; Some(v) } else { i += 1; None };
        while i < input.len() && matches!(input[i], b' '|b'\t'|b'\n'|b'\r') { i += 1; }
        let d = if i < input.len() && input[i] != b'=' { let v = decode_char(input[i])?; i += 1; Some(v) } else { if i < input.len() { i += 1; } None };
        out.push((a << 2) | (b >> 4));
        if let Some(cv) = c { out.push((b << 4) | (cv >> 2)); }
        if let Some(dv) = d { if let Some(cv) = c { out.push((cv << 6) | dv); } }
    }
    Ok(out)
}

// ── Phase 83: Terminal Pixel Graphics — Kitty / Sixel / Halfblock ──

/// Detect which terminal graphics protocol the current terminal supports.
fn term_supports_graphics_impl() -> &'static str {
    // Kitty: only detect when running inside actual Kitty terminal
    if std::env::var("KITTY_PID").is_ok() || std::env::var("KITTY_WINDOW_ID").is_ok() {
        return "kitty";
    }
    if let Ok(term) = std::env::var("TERM") {
        if term.contains("kitty") {
            return "kitty";
        }
    }
    // WezTerm supports Kitty graphics protocol
    if let Ok(tp) = std::env::var("TERM_PROGRAM") {
        let tp_lower = tp.to_lowercase();
        if tp_lower.contains("kitty") {
            return "kitty";
        }
        if tp_lower.contains("wezterm") {
            return "kitty";
        }
    }
    // iTerm2 supports Kitty protocol
    if let Ok(lc) = std::env::var("LC_TERMINAL") {
        if lc.contains("iTerm2") {
            return "kitty";
        }
    }
    // Sixel: only for terminals known to handle it properly
    if let Ok(term) = std::env::var("TERM") {
        if term.contains("vt340") || term.contains("mlterm") {
            return "sixel";
        }
    }
    // Default: halfblock (ANSI truecolor) — works everywhere
    // Users can opt into sixel/kitty with --set mode=sixel or --set mode=kitty
    "halfblock"
}

/// Emit image via Kitty Graphics Protocol (base64 chunked RGB).
fn term_image_kitty(width: usize, height: usize, rgb_data: &[u8], image_id: Option<u8>) {
    use std::io::Write;
    let b64 = base64_encode_pure(rgb_data);
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let chunk_size = 4096;
    let b64_bytes = b64.as_bytes();
    let total_chunks = (b64_bytes.len() + chunk_size - 1) / chunk_size;

    for (i, chunk) in b64_bytes.chunks(chunk_size).enumerate() {
        let more = if i == total_chunks - 1 { 0 } else { 1 };
        let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk) };
        if i == 0 {
            let id_str = match image_id {
                Some(id) => format!(",i={}", id),
                None => String::new(),
            };
            let _ = write!(out, "\x1b_Ga=T,f=24,s={},v={},t=d,m={},q=2{};{}\x1b\\",
                width, height, more, id_str, chunk_str);
        } else {
            let _ = write!(out, "\x1b_Gm={};{}\x1b\\", more, chunk_str);
        }
    }
    let _ = write!(out, "\n");
    let _ = out.flush();
}

/// Emit image via Sixel protocol — adaptive 255-color palette + Bayer dithering.
fn term_image_sixel(width: usize, height: usize, rgb_data: &[u8]) {
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let npix = width * height;

    // Bayer 4×4 ordered dither matrix (normalized to ±3.5 for 5-bit quantization step=8)
    const BAYER: [[i32; 4]; 4] = [
        [-4,  0, -3,  1],
        [ 2, -2,  3, -1],
        [-2,  2, -4,  0],
        [ 3, -1,  1, -3],
    ];

    // 1. Quantize each pixel to 5-bit RGB (32 levels/channel), count popularity
    let mut pop = vec![0u32; 32768];
    let mut quant_keys = Vec::with_capacity(npix);
    for i in 0..npix {
        let y = i / width;
        let x = i % width;
        let d = BAYER[y % 4][x % 4];
        let r = ((rgb_data[i * 3] as i32 + d).clamp(0, 255) as usize) >> 3;
        let g = ((rgb_data[i * 3 + 1] as i32 + d).clamp(0, 255) as usize) >> 3;
        let b = ((rgb_data[i * 3 + 2] as i32 + d).clamp(0, 255) as usize) >> 3;
        let key = (r << 10) | (g << 5) | b;
        pop[key] += 1;
        quant_keys.push(key as u16);
    }

    // 2. Select top 255 most popular 5-bit colors
    let mut ranked: Vec<u16> = (0..32768u16).filter(|&k| pop[k as usize] > 0).collect();
    ranked.sort_by(|a, b| pop[*b as usize].cmp(&pop[*a as usize]));
    let pal_size = ranked.len().min(255);

    // 3. Build palette + direct lookup table (32KB, cache-friendly)
    let mut pal_r5 = vec![0u8; pal_size];
    let mut pal_g5 = vec![0u8; pal_size];
    let mut pal_b5 = vec![0u8; pal_size];
    let mut lut = vec![0u8; 32768]; // 15-bit key → palette index
    let mut in_pal = vec![false; 32768];
    for i in 0..pal_size {
        let key = ranked[i] as usize;
        pal_r5[i] = ((key >> 10) & 31) as u8;
        pal_g5[i] = ((key >> 5) & 31) as u8;
        pal_b5[i] = (key & 31) as u8;
        lut[key] = i as u8;
        in_pal[key] = true;
    }

    // 4. Map non-palette colors to nearest palette entry
    for key in 0..32768usize {
        if pop[key] > 0 && !in_pal[key] {
            let r = ((key >> 10) & 31) as i32;
            let g = ((key >> 5) & 31) as i32;
            let b = (key & 31) as i32;
            let mut best = 0u8;
            let mut best_d = i32::MAX;
            for j in 0..pal_size {
                let dr = r - pal_r5[j] as i32;
                let dg = g - pal_g5[j] as i32;
                let db = b - pal_b5[j] as i32;
                let dist = dr * dr + dg * dg + db * db;
                if dist < best_d { best_d = dist; best = j as u8; }
            }
            lut[key] = best;
        }
    }

    // 5. Build per-pixel palette index array
    let mut pixel_idx = Vec::with_capacity(npix);
    for i in 0..npix {
        pixel_idx.push(lut[quant_keys[i] as usize]);
    }

    // 6. Emit sixel: DCS q " 1;1;width;height
    let _ = write!(out, "\x1bP0;0;0q\"1;1;{};{}", width, height);

    // Define palette (adaptive entries, full 0-100% precision)
    for i in 0..pal_size {
        let rp = (pal_r5[i] as u16 * 8 + 4) as u16 * 100 / 255;
        let gp = (pal_g5[i] as u16 * 8 + 4) as u16 * 100 / 255;
        let bp = (pal_b5[i] as u16 * 8 + 4) as u16 * 100 / 255;
        let _ = write!(out, "#{};2;{};{};{}", i, rp, gp, bp);
    }

    // Render sixel bands (6 pixel rows each)
    let six_rows = (height + 5) / 6;
    for six_row in 0..six_rows {
        let y_start = six_row * 6;

        let mut color_patterns: Vec<(u8, Vec<u8>)> = Vec::new();
        let mut color_index = vec![-1i16; pal_size];

        for x in 0..width {
            for dy in 0..6usize {
                let y = y_start + dy;
                if y >= height { continue; }
                let cidx = pixel_idx[y * width + x];
                let ci = cidx as usize;
                if color_index[ci] < 0 {
                    color_index[ci] = color_patterns.len() as i16;
                    color_patterns.push((cidx, vec![0u8; width]));
                }
                color_patterns[color_index[ci] as usize].1[x] |= 1 << dy;
            }
        }

        let mut first = true;
        for (cidx, pattern) in &color_patterns {
            if !first {
                let _ = write!(out, "$");
            }
            let _ = write!(out, "#{}", cidx);
            for x in 0..width {
                let _ = out.write_all(&[pattern[x] + 63]);
            }
            first = false;
        }
        if six_row < six_rows - 1 {
            let _ = write!(out, "-");
        }
    }

    let _ = write!(out, "\x1b\\\n");
    let _ = out.flush();
}

/// Emit image via ANSI truecolor halfblock characters (universal fallback).
fn term_image_halfblock_raw(width: usize, height: usize, rgb_data: &[u8]) {
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    let text_rows = (height + 1) / 2;
    for row in 0..text_rows {
        let top_y = row * 2;
        let bot_y = row * 2 + 1;
        for x in 0..width {
            let ti = (top_y * width + x) * 3;
            let tr = rgb_data[ti]; let tg = rgb_data[ti + 1]; let tb = rgb_data[ti + 2];
            if bot_y < height {
                let bi = (bot_y * width + x) * 3;
                let br = rgb_data[bi]; let bg = rgb_data[bi + 1]; let bb = rgb_data[bi + 2];
                let _ = write!(out, "\x1b[38;2;{};{};{};48;2;{};{};{}m\u{2580}",
                    tr, tg, tb, br, bg, bb);
            } else {
                let _ = write!(out, "\x1b[38;2;{};{};{}m\u{2580}", tr, tg, tb);
            }
        }
        let _ = write!(out, "\x1b[0m\n");
    }
    let _ = out.flush();
}

/// Dispatch terminal image rendering using best available protocol.
/// If `mode_override` is Some, use that protocol instead of auto-detection.
fn term_image_dispatch(width: usize, height: usize, rgb_data: &[u8], image_id: Option<u8>, mode_override: Option<&str>) {
    let mode = match mode_override {
        Some(m) => m,
        None => term_supports_graphics_impl(),
    };
    match mode {
        "kitty" => term_image_kitty(width, height, rgb_data, image_id),
        "sixel" => term_image_sixel(width, height, rgb_data),
        _ => term_image_halfblock_raw(width, height, rgb_data),
    }
}

// ── Pure-Rust ISO8601 / datetime helpers (Phase 45 — replaces time crate) ──

/// Convert Unix timestamp (seconds) to (year, month, day, hour, min, sec).
fn unix_to_calendar(secs: i64) -> (i32, u8, u8, u8, u8, u8) {
    // Days since 1970-01-01
    let days = secs.div_euclid(86400) as i32;
    let day_secs = secs.rem_euclid(86400) as u32;
    let hour = (day_secs / 3600) as u8;
    let min  = ((day_secs % 3600) / 60) as u8;
    let sec  = (day_secs % 60) as u8;

    // Gregorian calendar: days since epoch → (year, month, day)
    // Using the algorithm from https://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u8;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u8;
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d, hour, min, sec)
}

/// Parse ISO8601 date/time string to Unix timestamp.
/// Supports: "2024-01-15", "2024-01-15T10:30:00", "2024-01-15T10:30:00Z",
///           "2024-01-15T10:30:00+05:30", "2024-01-15 10:30:00"
fn parse_iso8601(s: &str) -> Result<i64, String> {
    let s = s.trim();
    if s.len() < 10 { return Err(format!("too short for ISO8601: '{}'", s)); }

    // Parse date portion
    let year: i32 = s[0..4].parse().map_err(|_| "bad year")?;
    if s.as_bytes()[4] != b'-' { return Err("expected '-' after year".into()); }
    let month: u32 = s[5..7].parse().map_err(|_| "bad month")?;
    if s.as_bytes()[7] != b'-' { return Err("expected '-' after month".into()); }
    let day: u32 = s[8..10].parse().map_err(|_| "bad day")?;

    // Parse time portion
    let (hour, min, sec, tz_offset_secs) = if s.len() > 10 {
        let sep = s.as_bytes()[10];
        if sep != b'T' && sep != b' ' { return Err(format!("expected T or space at position 10, got '{}'", sep as char)); }
        if s.len() < 19 { return Err("too short for time portion".into()); }
        let h: i64 = s[11..13].parse().map_err(|_| "bad hour")?;
        let m: i64 = s[14..16].parse().map_err(|_| "bad min")?;
        let sc: i64 = s[17..19].parse().map_err(|_| "bad sec")?;

        // Parse timezone
        let tz = if s.len() > 19 {
            let tz_part = &s[19..];
            if tz_part.starts_with('Z') || tz_part.starts_with('z') {
                0i64
            } else if tz_part.starts_with('+') || tz_part.starts_with('-') {
                let sign: i64 = if tz_part.starts_with('+') { 1 } else { -1 };
                let tz_str = &tz_part[1..];
                if tz_str.len() >= 5 {
                    let th: i64 = tz_str[0..2].parse().map_err(|_| "bad tz hour")?;
                    let tm: i64 = tz_str[3..5].parse().map_err(|_| "bad tz min")?;
                    sign * (th * 3600 + tm * 60)
                } else { 0 }
            } else { 0 }
        } else { 0 };
        (h, m, sc, tz)
    } else {
        (0, 0, 0, 0)
    };

    // Days from 1970-01-01 using Gregorian calendar algorithm
    let y = year as i64;
    let mo = month as i64;
    let d = day as i64;
    // Adjust for months (March = month 3 in standard)
    let (adj_y, adj_m) = if mo <= 2 { (y - 1, mo + 9) } else { (y, mo - 3) };
    let era = adj_y.div_euclid(400);
    let yoe = adj_y.rem_euclid(400);
    let doy = (153 * adj_m + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days_since_epoch = era * 146097 + doe - 719468;

    let unix = days_since_epoch * 86400 + hour * 3600 + min * 60 + sec - tz_offset_secs;
    Ok(unix)
}

/// Format Unix timestamp with a strftime-like format string.
/// Supports: %Y %m %d %H %M %S
fn format_unix_timestamp(unix_secs: i64, fmt: &str) -> Result<String, String> {
    let (year, month, day, hour, min, sec) = unix_to_calendar(unix_secs);
    let mut out = String::new();
    let bytes = fmt.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 1 < bytes.len() {
            i += 1;
            match bytes[i] {
                b'Y' => out.push_str(&format!("{:04}", year)),
                b'm' => out.push_str(&format!("{:02}", month)),
                b'd' => out.push_str(&format!("{:02}", day)),
                b'H' => out.push_str(&format!("{:02}", hour)),
                b'M' => out.push_str(&format!("{:02}", min)),
                b'S' => out.push_str(&format!("{:02}", sec)),
                c    => { out.push('%'); out.push(c as char); }
            }
        } else {
            out.push(bytes[i] as char);
        }
        i += 1;
    }
    Ok(out)
}

/// Execute an HTTP request over plain TCP (Phase 46 — ureq removed).
fn do_http_request(method: &str, url: &str, body_opt: Option<&str>) -> (f32, String, f32, String) {
    crate::http_io::do_request(method, url, body_opt)
}

// ── HTTP server — thread_local request cache (Phase 46) ──────────────
// Accepted connections' parsed requests are stored here so that
// http_method(), http_path(), http_body() can look them up by fd.

thread_local! {
    static HTTP_REQUESTS: RefCell<HashMap<i64, crate::http_io::HttpRequest>> =
        RefCell::new(HashMap::new());
}

// ─── Video handle storage for video_open / video_frame ──────────────────────
enum VideoHandle {
    Gif {
        width: u32,
        height: u32,
        frame_count: usize,
        fps: f32,
        decoded_frames: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>, // R, G, B per frame
    },
    Avi {
        data: Vec<u8>,
        info: crate::image_io::avi::AviInfo,
    },
}

thread_local! {
    static VIDEO_HANDLES: RefCell<HashMap<u32, VideoHandle>> = RefCell::new(HashMap::new());
    static NEXT_VIDEO_ID: Cell<u32> = Cell::new(1);
}

/// A user-defined function (pipeline fragment).
#[derive(Debug, Clone)]
struct FnDef {
    params: Vec<String>,
    body: Vec<StageCall>,
}

/// A user-defined scalar function (imperative body with return).
#[derive(Debug, Clone)]
struct ScalarFnDef {
    params: Vec<String>,
    body: Vec<(Statement, octoflow_parser::ast::Span)>,
    /// R-04: Captured module-level scalars (closure-like environment for nested imports)
    captured_scalars: HashMap<String, Value>,
    /// R-04: Captured module-level arrays (closure-like environment for nested imports)
    captured_arrays: HashMap<String, Vec<Value>>,
}

/// Result from evaluating a statement in the REPL context.
pub enum StmtResult {
    /// Statement executed silently (let, assign).
    Silent,
    /// A function or struct was defined.
    FnDefined(String),
    /// A print() statement already produced output.
    Printed,
    /// A stream was created (name, element count).
    StreamCreated(String, usize),
    /// An expression produced a value.
    Value(Value),
}

/// Persistent REPL context — holds all interpreter state across inputs.
pub struct Context {
    streams: HashMap<String, Vec<f32>>,
    scalars: HashMap<String, Value>,
    functions: HashMap<String, FnDef>,
    struct_defs: HashMap<String, Vec<String>>,
    arrays: HashMap<String, Vec<Value>>,
    hashmaps: HashMap<String, HashMap<String, Value>>,
    scalar_fns: HashMap<String, ScalarFnDef>,
    mutable_scalars: std::collections::HashSet<String>,
    image_dims: HashMap<String, (u32, u32)>,
    /// Registered extern functions from `extern "lib" { fn ... }` blocks.
    extern_fns: HashMap<String, ExternFnDecl>,
    rng: Cell<u64>,
    gpu: Option<octoflow_vulkan::VulkanCompute>,
    base_dir: String,
    total_elements: usize,
    max_iters: usize,
}

impl Context {
    /// Create a new REPL context, optionally initializing GPU.
    pub fn new() -> Self {
        // REPL defaults to allowing file I/O and network
        ALLOW_READ.with(|c| *c.borrow_mut() = crate::PermScope::AllowAll);
        ALLOW_WRITE.with(|c| *c.borrow_mut() = crate::PermScope::AllowAll);
        ALLOW_NET.with(|c| *c.borrow_mut() = crate::PermScope::AllowAll);
        ALLOW_EXEC.with(|c| *c.borrow_mut() = crate::PermScope::AllowAll);

        let seed: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);

        let gpu = match octoflow_vulkan::VulkanCompute::new() {
            Ok(g) => Some(g),
            Err(_) => None,
        };

        Context {
            streams: HashMap::new(),
            scalars: HashMap::new(),
            functions: HashMap::new(),
            struct_defs: HashMap::new(),
            arrays: HashMap::new(),
            hashmaps: HashMap::new(),
            scalar_fns: HashMap::new(),
            mutable_scalars: std::collections::HashSet::new(),
            image_dims: HashMap::new(),
            extern_fns: HashMap::new(),
            rng: Cell::new(seed),
            gpu,
            base_dir: ".".to_string(),
            total_elements: 0,
            max_iters: 10_000_000,
        }
    }

    /// Set the base directory for resolving module/file paths.
    pub fn set_base_dir(&mut self, dir: &str) {
        self.base_dir = dir.to_string();
    }

    /// Set the `_` variable to the last expression result.
    pub fn set_underscore(&mut self, val: Value) {
        self.mutable_scalars.insert("_".to_string());
        self.scalars.insert("_".to_string(), val);
    }

    /// Whether GPU is available.
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// GPU device name, if available.
    pub fn gpu_name(&self) -> Option<String> {
        self.gpu.as_ref().map(|g| g.device_name().to_string())
    }

    /// Evaluate a single statement and return the result.
    pub fn eval_statement(&mut self, stmt: &Statement, span: &octoflow_parser::ast::Span) -> Result<StmtResult, CliError> {
        (match stmt {
            Statement::FnDecl { name, params, body } => {
                self.functions.insert(name.clone(), FnDef {
                    params: params.clone(),
                    body: body.clone(),
                });
                Ok(StmtResult::FnDefined(name.clone()))
            }
            Statement::UseDecl { module } => {
                import_module(&self.base_dir, module, &mut self.functions, &mut self.scalar_fns,
                              &mut self.struct_defs, &mut self.scalars, &mut self.arrays, &mut self.hashmaps, &mut self.mutable_scalars)?;
                Ok(StmtResult::Silent)
            }
            Statement::StructDef { name, fields } => {
                self.struct_defs.insert(name.clone(), fields.clone());
                Ok(StmtResult::FnDefined(format!("struct {}", name)))
            }
            Statement::ArrayDecl { name, elements, mutable } => {
                let mut values = Vec::new();
                for elem in elements {
                    let val = eval_scalar(elem, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                    values.push(val);
                }
                self.arrays.insert(name.clone(), values);
                if *mutable { self.mutable_scalars.insert(name.clone()); }
                Ok(StmtResult::Silent)
            }
            Statement::StreamDecl { name, expr } => {
                let data = eval_expr(expr, &self.streams, &self.scalars, &self.gpu, &self.base_dir, &self.functions, &mut self.image_dims)?;
                let count = data.len();
                self.total_elements += count;
                if let Some(dims) = find_source_dims(expr, &self.image_dims) {
                    self.image_dims.insert(name.clone(), dims);
                }
                self.streams.insert(name.clone(), data);
                Ok(StmtResult::StreamCreated(name.clone(), count))
            }
            Statement::LetDecl { name, value, mutable } => {
                // vec constructors
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    if let Some(dim) = match fn_name.as_str() {
                        "vec2" => Some(2usize), "vec3" => Some(3usize), "vec4" => Some(4usize), _ => None,
                    } {
                        if args.len() != dim {
                            return Err(CliError::Compile(format!("{}() requires {} components, got {}", fn_name, dim, args.len())));
                        }
                        let components = ["x", "y", "z", "w"];
                        for (i, arg) in args.iter().enumerate() {
                            let val = eval_scalar(arg, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?.as_float()?;
                            self.scalars.insert(format!("{}.{}", name, components[i]), Value::Float(val));
                        }
                        if *mutable { self.mutable_scalars.insert(name.clone()); }
                        return Ok(StmtResult::Silent);
                    }
                    // struct constructors
                    if let Some(fields) = self.struct_defs.get(fn_name).cloned() {
                        if args.len() != fields.len() {
                            return Err(CliError::Compile(format!("{}() requires {} fields, got {}", fn_name, fields.len(), args.len())));
                        }
                        for (i, arg) in args.iter().enumerate() {
                            let val = eval_scalar(arg, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?.as_float()?;
                            self.scalars.insert(format!("{}.{}", name, &fields[i]), Value::Float(val));
                        }
                        if *mutable { self.mutable_scalars.insert(name.clone()); }
                        return Ok(StmtResult::Silent);
                    }
                    // try() error handling
                    if fn_name == "try" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("try() requires exactly 1 argument".into()));
                        }
                        match eval_scalar(&args[0], &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars) {
                            Ok(val) => {
                                self.scalars.insert(format!("{}.value", name), val);
                                self.scalars.insert(format!("{}.ok", name), Value::Float(1.0));
                                self.scalars.insert(format!("{}.error", name), Value::Str(String::new()));
                            }
                            Err(e) => {
                                self.scalars.insert(format!("{}.value", name), Value::Str(String::new()));
                                self.scalars.insert(format!("{}.ok", name), Value::Float(0.0));
                                self.scalars.insert(format!("{}.error", name), Value::Str(format!("{}", e)));
                            }
                        }
                        return Ok(StmtResult::Silent);
                    }
                    // read_image(path) → decompose into name.width, name.height + channel arrays
                    if fn_name == "read_image" && args.len() == 1 {
                        let path_val = eval_scalar(&args[0], &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                        let path = path_val.as_str().map_err(|_| CliError::Compile("read_image() path must be a string".into()))?.to_string();
                        check_read_permission_for(&path)?;
                        let (pixels, w, h) = crate::image_io::read_image(&path)?;
                        let n = (w as usize) * (h as usize);
                        let mut r_arr = Vec::with_capacity(n);
                        let mut g_arr = Vec::with_capacity(n);
                        let mut b_arr = Vec::with_capacity(n);
                        for i in 0..n {
                            r_arr.push(Value::Float(pixels[i * 3]));
                            g_arr.push(Value::Float(pixels[i * 3 + 1]));
                            b_arr.push(Value::Float(pixels[i * 3 + 2]));
                        }
                        let r_name = format!("{}.r", name);
                        let g_name = format!("{}.g", name);
                        let b_name = format!("{}.b", name);
                        self.arrays.insert(r_name.clone(), r_arr);
                        self.arrays.insert(g_name.clone(), g_arr);
                        self.arrays.insert(b_name.clone(), b_arr);
                        self.scalars.insert(format!("{}.width", name), Value::Float(w as f32));
                        self.scalars.insert(format!("{}.height", name), Value::Float(h as f32));
                        self.scalars.insert(format!("{}.r", name), Value::Str(r_name));
                        self.scalars.insert(format!("{}.g", name), Value::Str(g_name));
                        self.scalars.insert(format!("{}.b", name), Value::Str(b_name));
                        if *mutable { self.mutable_scalars.insert(name.clone()); }
                        return Ok(StmtResult::Silent);
                    }
                    // HTTP client functions
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        check_net_permission()?;
                        let (method, expected_args) = match fn_name.as_str() {
                            "http_get" => ("GET", 1), "http_post" => ("POST", 2),
                            "http_put" => ("PUT", 2), _ => ("DELETE", 1),
                        };
                        if args.len() != expected_args {
                            return Err(CliError::Compile(format!("{}() requires {} argument(s), got {}", fn_name, expected_args, args.len())));
                        }
                        let url_val = eval_scalar(&args[0], &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                        let url = url_val.as_str().map_err(|_| CliError::Compile(format!("{}() URL must be a string", fn_name)))?.to_string();
                        let body_str = if expected_args == 2 {
                            let b = eval_scalar(&args[1], &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                            Some(b.as_str().map_err(|_| CliError::Compile(format!("{}() body must be a string", fn_name)))?.to_string())
                        } else { None };
                        let (status, body, ok, error) = do_http_request(method, &url, body_str.as_deref());
                        self.scalars.insert(format!("{}.status", name), Value::Float(status));
                        self.scalars.insert(format!("{}.body", name), Value::Str(body));
                        self.scalars.insert(format!("{}.ok", name), Value::Float(ok));
                        self.scalars.insert(format!("{}.error", name), Value::Str(error));
                        return Ok(StmtResult::Silent);
                    }
                    // Command execution: exec(cmd, ...args)
                    if fn_name == "exec" {
                        check_exec_permission()?;
                        if args.is_empty() {
                            return Err(CliError::Compile("exec() requires at least 1 argument (command)".into()));
                        }
                        // Evaluate all args to strings
                        let mut cmd_args = Vec::new();
                        for arg in args {
                            let val = eval_scalar(arg, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                            cmd_args.push(val.to_string());
                        }
                        let command = &cmd_args[0];
                        let command_args = &cmd_args[1..];

                        use std::process::Command;
                        let output = Command::new(command)
                            .args(command_args)
                            .output()
                            .map_err(|e| CliError::Io(format!("exec(\"{}\"): {}", command, e)))?;

                        let status_code = output.status.code().unwrap_or(-1) as f32;
                        let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
                        let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();
                        let ok = if output.status.success() { 1.0 } else { 0.0 };

                        self.scalars.insert(format!("{}.status", name), Value::Float(status_code));
                        self.scalars.insert(format!("{}.output", name), Value::Str(stdout_str));
                        self.scalars.insert(format!("{}.ok", name), Value::Float(ok));
                        self.scalars.insert(format!("{}.error", name), Value::Str(stderr_str));
                        return Ok(StmtResult::Silent);
                    }
                    // video_open(byte_array) → handle + .width .height .frames .fps scalars
                    if fn_name == "video_open" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("video_open() requires 1 argument (byte array name)".into()));
                        }
                        let arr_name = match &args[0] {
                            ScalarExpr::Ref(n) => n.clone(),
                            _ => return Err(CliError::Compile("video_open() argument must be an array name".into())),
                        };
                        let bytes = video_open_extract_bytes(&arr_name, &self.arrays)?;
                        let (handle, w, h, fc, fps) = video_open_from_bytes(bytes)?;
                        video_open_register(handle, name, w, h, fc, fps, &mut self.scalars);
                        return Ok(StmtResult::Silent);
                    }
                    // video_open_file(path) → same as video_open but reads file directly (no Vec<Value> overhead)
                    if fn_name == "video_open_file" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("video_open_file() requires 1 argument (file path)".into()));
                        }
                        let path_val = eval_scalar(&args[0], &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                        let path = path_val.as_str().map_err(|_| CliError::Compile("video_open_file() path must be a string".into()))?.to_string();
                        check_read_permission_for(&path)?;
                        let (handle, w, h, fc, fps) = video_open_file_bytes(&path)?;
                        video_open_register(handle, name, w, h, fc, fps, &mut self.scalars);
                        return Ok(StmtResult::Silent);
                    }
                    // video_frame(handle, index) → .r .g .b arrays
                    if fn_name == "video_frame" {
                        if args.len() != 2 {
                            return Err(CliError::Compile("video_frame() requires 2 arguments (handle, frame_index)".into()));
                        }
                        let handle_val = eval_scalar(&args[0], &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                        let idx_val = eval_scalar(&args[1], &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                        let handle_id = handle_val.as_float().map_err(|_| CliError::Compile("video_frame(): handle must be numeric".into()))? as u32;
                        let frame_idx = idx_val.as_float().map_err(|_| CliError::Compile("video_frame(): index must be numeric".into()))? as usize;
                        let (r, g, b) = video_frame_decode(handle_id, frame_idx)?;
                        gpu_array_insert(format!("{}.r", name), r);
                        gpu_array_insert(format!("{}.g", name), g);
                        gpu_array_insert(format!("{}.b", name), b);
                        return Ok(StmtResult::Silent);
                    }
                    // Extern FFI function call: let r = SomeCFn(args...)
                    if is_extern_fn(fn_name) {
                        let mut arg_vals = Vec::new();
                        for arg in args {
                            arg_vals.push(eval_scalar(arg, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?);
                        }
                        let result = call_extern_fn(fn_name, &arg_vals)?;
                        self.scalars.insert(name.clone(), result);
                        if *mutable { self.mutable_scalars.insert(name.clone()); }
                        return Ok(StmtResult::Silent);
                    }
                    // Array-returning functions: read_lines, list_dir, split
                    if let Some(result) = eval_array_fn(fn_name, args, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)? {
                        match result {
                            ArrayResult::Values(arr) => { self.arrays.insert(name.clone(), arr); }
                            ArrayResult::GpuFloats(arr) => { gpu_array_insert(name.clone(), arr); }
                            ArrayResult::Resident(buf) => { gpu_array_insert_resident(name.clone(), buf); }
                        }
                        if *mutable { self.mutable_scalars.insert(name.clone()); }
                        return Ok(StmtResult::Silent);
                    }
                    // Hashmap-returning functions: json_parse
                    if let Some(hm) = eval_hashmap_fn(fn_name, args, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)? {
                        self.hashmaps.insert(name.clone(), hm);
                        if *mutable { self.mutable_scalars.insert(name.clone()); }
                        return Ok(StmtResult::Silent);
                    }
                }
                let result = eval_scalar(value, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                // Check if a user function returned an array via the side-channel
                let returned_arr = RETURNED_ARRAY.with(|r| r.borrow_mut().take());
                if let Some(arr) = returned_arr {
                    self.arrays.insert(name.clone(), arr);
                } else {
                    let returned_hm = RETURNED_MAP.with(|r| r.borrow_mut().take());
                    if let Some(hm) = returned_hm {
                        self.hashmaps.insert(name.clone(), hm);
                    } else {
                        self.scalars.insert(name.clone(), result);
                    }
                }
                if *mutable { self.mutable_scalars.insert(name.clone()); }
                Ok(StmtResult::Silent)
            }
            Statement::Assign { name, value } => {
                if !self.mutable_scalars.contains(name) {
                    return Err(CliError::Compile(format!(
                        "Cannot assign to '{}': not declared as mutable (use 'let mut')", name
                    )));
                }
                let result = eval_scalar(value, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                self.scalars.insert(name.clone(), result);
                Ok(StmtResult::Silent)
            }
            Statement::ArrayAssign { array, index, value } => {
                if !self.mutable_scalars.contains(array) {
                    return Err(CliError::Compile(format!(
                        "Cannot assign to '{}': not declared as mutable (use 'let mut')", array
                    )));
                }
                gpu_array_materialize(array, &mut self.arrays);
                let idx = eval_scalar(index, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?.as_float()? as usize;
                let val = eval_scalar(value, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                let arr = self.arrays.get_mut(array)
                    .ok_or_else(|| CliError::Compile(format!("undefined array '{}'", array)))?;
                if idx >= arr.len() {
                    return Err(CliError::Compile(format!(
                        "Array index out of bounds: {}[{}] (length {})", array, idx, arr.len()
                    )));
                }
                arr[idx] = val;
                Ok(StmtResult::Silent)
            }
            Statement::ArrayPush { array, value } => {
                if !self.mutable_scalars.contains(array) {
                    self.mutable_scalars.insert(array.clone());  // Auto-promote to mutable on push
                }
                gpu_array_materialize(array, &mut self.arrays);
                let val = eval_scalar(value, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                let arr = self.arrays.get_mut(array)
                    .ok_or_else(|| CliError::Compile(format!("undefined array '{}'", array)))?;
                arr.push(val);
                Ok(StmtResult::Silent)
            }
            Statement::Emit { expr, path } => {
                let data = match expr {
                    Expr::Ref { name } => self.streams
                        .get(name)
                        .ok_or_else(|| CliError::UndefinedStream(name.clone()))?,
                    _ => return Err(CliError::Compile("emit() argument must be a stream name".into())),
                };
                let full_path = resolve_path(&self.base_dir, path)?;
                if crate::image_io::is_image_path(&full_path) {
                    let stream_name = match expr {
                        Expr::Ref { name } => name,
                        _ => unreachable!(),
                    };
                    let (w, h) = self.image_dims.get(stream_name)
                        .copied()
                        .ok_or_else(|| CliError::Compile(format!(
                            "cannot write image '{}': no image dimensions found for stream '{}'",
                            path, stream_name)))?;
                    crate::image_io::write_image(&full_path, data, w, h)?;
                } else if crate::octo_io::is_octo_path(&full_path) {
                    crate::octo_io::write_octo(&full_path, data)?;
                } else {
                    csv_write_floats(&full_path, data)?;
                }
                Ok(StmtResult::Silent)
            }
            Statement::Print { segments } => {
                let mut output = String::new();
                for seg in segments {
                    match seg {
                        PrintSegment::Literal(s) => output.push_str(s),
                        PrintSegment::Scalar { name, precision } => {
                            if let Some(value) = self.scalars.get(name) {
                                match (precision, value) {
                                    (Some(p), Value::Float(f)) => output.push_str(&format!("{:.prec$}", f, prec = *p)),
                                    (Some(_p), Value::Int(i)) => output.push_str(&format!("{}", i)),
                                    (Some(_), Value::Str(s)) => output.push_str(s),
                                    (Some(_), Value::Map(_)) => output.push_str(&format!("{}", value)),
                                    (Some(_), Value::None) => output.push_str("none"),
                                    (None, _) => output.push_str(&format!("{}", value)),
                                }
                            } else if let Some(s) = gpu_array_format(name) {
                                output.push_str(&s);
                            } else if let Some(arr) = self.arrays.get(name) {
                                let items: Vec<String> = arr.iter().map(|v| format!("{}", v)).collect();
                                output.push_str(&format!("[{}]", items.join(", ")));
                            } else if let Some(hm) = self.hashmaps.get(name) {
                                let mut keys: Vec<&String> = hm.keys().collect();
                                keys.sort();
                                let items: Vec<String> = keys.iter().map(|k| format!("{}={}", k, hm[*k])).collect();
                                output.push_str(&format!("{{{}}}", items.join(", ")));
                            } else {
                                return Err(CliError::UndefinedScalar(name.clone()));
                            }
                        }
                        PrintSegment::Expr(e) => {
                            let val = eval_scalar(e, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                            output.push_str(&format!("{}", val));
                        }
                    }
                }
                runtime_println(&output);
                Ok(StmtResult::Printed)
            }
            Statement::WhileLoop { condition, body } => {
                let max_while = self.max_iters;
                let mut iterations = 0;
                loop {
                    let cond = eval_scalar(condition, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?.as_float()?;
                    if cond == 0.0 { break; }
                    iterations += 1;
                    if iterations > max_while {
                        return Err(CliError::Compile(format!("while loop exceeded {} iterations (infinite loop?)", max_while)));
                    }
                    match execute_loop_body(body, &self.streams, &mut self.scalars, &mut self.mutable_scalars, &self.struct_defs, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.rng)? {
                        LoopControl::Break => break,
                        LoopControl::Continue => continue,
                        LoopControl::Return(_) => return Err(CliError::Compile("'return' can only be used inside a function".into())),
                        LoopControl::Normal => {}
                    }
                }
                Ok(StmtResult::Silent)
            }
            Statement::ForLoop { var, start, end, body } => {
                const MAX_FOR_ITERATIONS: usize = 10_000_000;
                let start_val = eval_scalar(start, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?.as_float()? as i64;
                let end_val = eval_scalar(end, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?.as_float()? as i64;
                let count = if end_val > start_val { (end_val - start_val) as usize } else { 0 };
                if count > MAX_FOR_ITERATIONS {
                    return Err(CliError::Compile(format!("for loop range {} to {} is {} iterations (max {})", start_val, end_val, count, MAX_FOR_ITERATIONS)));
                }
                for i in start_val..end_val {
                    self.scalars.insert(var.clone(), Value::Float(i as f32));
                    match execute_loop_body(body, &self.streams, &mut self.scalars, &mut self.mutable_scalars, &self.struct_defs, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.rng)? {
                        LoopControl::Break => break,
                        LoopControl::Continue => continue,
                        LoopControl::Return(_) => return Err(CliError::Compile("'return' can only be used inside a function".into())),
                        LoopControl::Normal => {}
                    }
                }
                self.scalars.remove(var);
                Ok(StmtResult::Silent)
            }
            Statement::ForEachLoop { var, iterable, body } => {
                // Materialize GPU array if needed (for-each requires Value iteration)
                gpu_array_materialize(iterable, &mut self.arrays);
                let arr = self.arrays.get(iterable)
                    .ok_or_else(|| CliError::Compile(format!("undefined array '{}' in for-each loop", iterable)))?
                    .clone();
                for val in &arr {
                    self.scalars.insert(var.clone(), val.clone());
                    match execute_loop_body(body, &self.streams, &mut self.scalars, &mut self.mutable_scalars, &self.struct_defs, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.rng)? {
                        LoopControl::Break => break,
                        LoopControl::Continue => continue,
                        LoopControl::Return(_) => return Err(CliError::Compile("'return' can only be used inside a function".into())),
                        LoopControl::Normal => {}
                    }
                }
                self.scalars.remove(var);
                Ok(StmtResult::Silent)
            }
            Statement::IfBlock { condition, body, elif_branches, else_body } => {
                let cond = eval_scalar(condition, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?.as_float()?;
                if cond != 0.0 {
                    for (s, _sp) in body {
                        execute_block_stmt(s, &self.streams, &mut self.scalars, &mut self.mutable_scalars, &self.struct_defs, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.rng)?;
                    }
                } else {
                    let mut matched = false;
                    for (elif_cond, elif_body) in elif_branches {
                        let ec = eval_scalar(elif_cond, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?.as_float()?;
                        if ec != 0.0 {
                            for (s, _sp) in elif_body {
                                execute_block_stmt(s, &self.streams, &mut self.scalars, &mut self.mutable_scalars, &self.struct_defs, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.rng)?;
                            }
                            matched = true;
                            break;
                        }
                    }
                    if !matched && !else_body.is_empty() {
                        for (s, _sp) in else_body {
                            execute_block_stmt(s, &self.streams, &mut self.scalars, &mut self.mutable_scalars, &self.struct_defs, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.rng)?;
                        }
                    }
                }
                Ok(StmtResult::Silent)
            }
            Statement::ScalarFnDecl { name, params, body } => {
                self.scalar_fns.insert(name.clone(), ScalarFnDef {
                    params: params.clone(),
                    body: body.clone(),
                    captured_scalars: HashMap::new(),
                    captured_arrays: HashMap::new(),
                });
                Ok(StmtResult::FnDefined(name.clone()))
            }
            Statement::MapDecl { name, mutable } => {
                self.hashmaps.insert(name.clone(), HashMap::new());
                if *mutable { self.mutable_scalars.insert(name.clone()); }
                Ok(StmtResult::Silent)
            }
            Statement::MapInsert { map, key, value } => {
                if !self.mutable_scalars.contains(map.as_str()) {
                    return Err(CliError::Compile(format!(
                        "Cannot insert into '{}': not declared as mutable (use 'let mut')", map
                    )));
                }
                let key_val = eval_scalar(key, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?.as_str()?.to_string();
                let val = eval_scalar(value, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                let hm = self.hashmaps.get_mut(map)
                    .ok_or_else(|| CliError::Compile(format!("undefined map '{}'", map)))?;
                hm.insert(key_val, val);
                Ok(StmtResult::Silent)
            }
            Statement::WriteFile { path, content } => {
                let path_val = eval_scalar(path, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("write_file() path must be a string".into()))?;
                check_write_permission_for(p)?;
                let content_val = eval_scalar(content, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                let c = content_val.to_string();
                std::fs::write(p, &c).map_err(|e| CliError::Io(format!("write_file(\"{}\"): {}", p, e)))?;
                Ok(StmtResult::Silent)
            }
            Statement::AppendFile { path, content } => {
                let path_val = eval_scalar(path, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("append_file() path must be a string".into()))?;
                check_write_permission_for(p)?;
                let content_val = eval_scalar(content, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                let c = content_val.to_string();
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new().create(true).append(true).open(p)
                    .map_err(|e| CliError::Io(format!("append_file(\"{}\"): {}", p, e)))?;
                file.write_all(c.as_bytes()).map_err(|e| CliError::Io(format!("append_file(\"{}\"): {}", p, e)))?;
                Ok(StmtResult::Silent)
            }
            Statement::SaveData { path, map_name } => {
                let path_val = eval_scalar(path, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("save_data() path must be a string".into()))?.to_string();
                check_write_permission_for(&p)?;
                let hm = self.hashmaps.get(map_name)
                    .ok_or_else(|| CliError::Compile(format!("save_data(): undefined map '{}'", map_name)))?;
                let content = serialize_od(hm);
                std::fs::write(&p, &content).map_err(|e| CliError::Io(format!("save_data(\"{}\"): {}", p, e)))?;
                Ok(StmtResult::Silent)
            }
            Statement::WriteCsv { path, array_name } => {
                let path_val = eval_scalar(path, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("write_csv() path must be a string".into()))?.to_string();
                check_write_permission_for(&p)?;
                gpu_array_materialize(array_name, &mut self.arrays);
                let arr = self.arrays.get(array_name)
                    .ok_or_else(|| CliError::Compile(format!("write_csv(): undefined array '{}'", array_name)))?;
                csv_write_structured(&p, arr)?;
                Ok(StmtResult::Silent)
            }
            Statement::WriteBytes { path, array_name } => {
                let path_val = eval_scalar(path, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("write_bytes() path must be a string".into()))?.to_string();
                check_write_permission_for(&p)?;
                gpu_array_materialize(array_name, &mut self.arrays);
                let arr = self.arrays.get(array_name)
                    .ok_or_else(|| CliError::Compile(format!("write_bytes(): undefined array '{}'", array_name)))?;
                do_write_bytes(&p, arr)?;
                Ok(StmtResult::Silent)
            }
            Statement::Return { .. } => {
                Err(CliError::Compile("'return' can only be used inside a function".into()))
            }
            Statement::Break => {
                Err(CliError::Compile("'break' can only be used inside a loop".into()))
            }
            Statement::Continue => {
                Err(CliError::Compile("'continue' can only be used inside a loop".into()))
            }
            Statement::ExternBlock { library, functions, .. } => {
                // Register extern fn declarations for later calls.
                // Actual library loading happens at call time (lazy).
                register_extern_block(library, functions);
                // Also keep in Context.extern_fns for REPL display
                for f in functions {
                    self.extern_fns.insert(f.name.clone(), ExternFnDecl {
                        library: library.clone(),
                        params: f.params.iter().map(|p| p.type_name.clone()).collect(),
                        return_type: f.return_type.clone(),
                    });
                }
                Ok(StmtResult::FnDefined(format!("extern \"{}\" ({} fn{})",
                    library, functions.len(), if functions.len() == 1 { "" } else { "s" })))
            }
            Statement::ExprStmt { expr } => {
                let _ = eval_scalar(expr, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)?;
                // R-05: Apply scalar writeback from user function calls
                SCALAR_WRITEBACK.with(|sw| {
                    if let Some(writes) = sw.borrow_mut().take() {
                        for (name, value) in writes {
                            self.scalars.insert(name, value);
                        }
                    }
                });
                Ok(StmtResult::Silent)
            }
        }).map_err(|e| e.with_line(span.line))
    }

    /// Evaluate a bare scalar expression.
    pub fn eval_expression(&mut self, expr: &ScalarExpr) -> Result<Value, CliError> {
        eval_scalar(expr, &self.streams, &self.scalars, &self.gpu, &mut self.arrays, &mut self.hashmaps, &self.scalar_fns, &self.struct_defs, &self.rng, &self.mutable_scalars)
    }

    /// List all variables with (name, type, display_value).
    pub fn list_variables(&self) -> Vec<(String, String, String)> {
        let mut vars: Vec<(String, String, String)> = self.scalars.iter()
            .map(|(name, val)| {
                let (ty, display) = match val {
                    Value::Float(_) => ("float".to_string(), format_repl_value(val)),
                    Value::Int(i) => ("int".to_string(), format!("{}", i)),
                    Value::Str(s) => ("string".to_string(), format!("\"{}\"", s)),
                    Value::Map(m) => ("map".to_string(), format!("{{{} entries}}", m.len())),
                    Value::None => ("none".to_string(), "none".to_string()),
                };
                (name.clone(), ty, display)
            })
            .collect();
        // Add arrays
        for (name, arr) in &self.arrays {
            vars.push((name.clone(), "array".to_string(), format!("[{} elements]", arr.len())));
        }
        // Add hashmaps
        for (name, hm) in &self.hashmaps {
            vars.push((name.clone(), "map".to_string(), format!("{{{} entries}}", hm.len())));
        }
        // Add streams
        for (name, data) in &self.streams {
            vars.push((name.clone(), "stream".to_string(), format!("[{} elements]", data.len())));
        }
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        vars
    }

    /// List all user-defined functions with (name, params_string).
    pub fn list_functions(&self) -> Vec<(String, String)> {
        let mut fns: Vec<(String, String)> = Vec::new();
        for (name, def) in &self.functions {
            fns.push((name.clone(), def.params.join(", ")));
        }
        for (name, def) in &self.scalar_fns {
            fns.push((name.clone(), def.params.join(", ")));
        }
        fns.sort_by(|a, b| a.0.cmp(&b.0));
        fns
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.streams.clear();
        self.scalars.clear();
        self.functions.clear();
        self.struct_defs.clear();
        self.arrays.clear();
        self.hashmaps.clear();
        self.scalar_fns.clear();
        self.mutable_scalars.clear();
        self.image_dims.clear();
        self.extern_fns.clear();
        self.total_elements = 0;
        let seed: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        self.rng.set(seed);
    }

    /// List GPU arrays with (name, len, location).
    pub fn list_gpu_arrays(&self) -> Vec<(String, usize, &'static str)> {
        GPU_ARRAYS.with(|ga| {
            let map = ga.borrow();
            let mut out: Vec<(String, usize, &'static str)> = map.iter().map(|(name, storage)| {
                match storage {
                    GpuArrayStorage::Cpu(v) => (name.clone(), v.len(), "cpu"),
                    GpuArrayStorage::Resident(buf) => (name.clone(), buf.len(), "gpu"),
                }
            }).collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out
        })
    }

    /// List streams with (name, len).
    pub fn list_streams(&self) -> Vec<(String, usize)> {
        let mut out: Vec<(String, usize)> = self.streams.iter()
            .map(|(name, data)| (name.clone(), data.len()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Count of scalars, arrays, hashmaps, streams, functions.
    pub fn state_counts(&self) -> (usize, usize, usize, usize, usize) {
        let fns = self.functions.len() + self.scalar_fns.len();
        (self.scalars.len(), self.arrays.len(), self.hashmaps.len(), self.streams.len(), fns)
    }

    /// Load and execute a .flow file in this context.
    pub fn load_file(&mut self, path: &str) -> Result<(), CliError> {
        check_read_permission_for(path)?;
        let source = std::fs::read_to_string(path)
            .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
        let program = octoflow_parser::parse(&source)
            .map_err(|e| CliError::Parse(format!("{}", e)))?;
        // Update base_dir to the file's directory
        let file_base = std::path::Path::new(path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());
        let old_base = self.base_dir.clone();
        self.base_dir = file_base;
        for (stmt, span) in &program.statements {
            self.eval_statement(stmt, span)?;
        }
        self.base_dir = old_base;
        Ok(())
    }
}

/// Format a Value for REPL display.
pub fn format_repl_value(val: &Value) -> String {
    match val {
        Value::Float(f) => {
            if *f == f.trunc() && f.is_finite() {
                // Integer-valued float: strip trailing .0
                format!("{}", *f as i64)
            } else {
                format!("{}", f)
            }
        }
        Value::Int(i) => format!("{}", i),
        Value::Str(s) => format!("\"{}\"", s),
        Value::Map(m) => format!("{}", Value::Map(m.clone())),
        Value::None => "none".to_string(),
    }
}

/// Execute a parsed OctoFlow program.
///
/// Returns `(elements_processed, used_gpu)`.
pub fn execute(program: &Program, base_dir: &str, overrides: &crate::Overrides) -> Result<(usize, bool), CliError> {
    // Set security flags for this execution run
    ALLOW_READ.with(|c| *c.borrow_mut() = overrides.allow_read.clone());
    ALLOW_WRITE.with(|c| *c.borrow_mut() = overrides.allow_write.clone());
    ALLOW_NET.with(|c| *c.borrow_mut() = overrides.allow_net.clone());
    ALLOW_EXEC.with(|c| *c.borrow_mut() = overrides.allow_exec.clone());
    ALLOW_FFI.with(|c| c.set(overrides.allow_ffi));
    GPU_MAX_BYTES.with(|c| c.set(overrides.gpu_max_bytes.unwrap_or(0)));
    GPU_ALLOCATED_BYTES.with(|c| c.set(0));
    RECURSION_DEPTH.with(|c| c.set(0));
    VERBOSE_INFER.with(|c| c.set(overrides.verbose));
    // Clear transitive import guard for fresh execution
    IMPORTED_PATHS.with(|s| s.borrow_mut().clear());

    // Apply path overrides (-i, -o) before execution
    let program = apply_path_overrides(program, overrides);

    let mut streams: HashMap<String, Vec<f32>> = HashMap::new();
    let mut scalars: HashMap<String, Value> = HashMap::new();
    let mut functions: HashMap<String, FnDef> = HashMap::new();
    let mut struct_defs: HashMap<String, Vec<String>> = HashMap::new();
    let mut arrays: HashMap<String, Vec<Value>> = HashMap::new();
    let mut hashmaps: HashMap<String, HashMap<String, Value>> = HashMap::new();
    let mut scalar_fns: HashMap<String, ScalarFnDef> = HashMap::new();
    let mut mutable_scalars: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut image_dims: HashMap<String, (u32, u32)> = HashMap::new();
    // Reset per-run thread_locals
    EXTERN_REGISTRY.with(|r| r.borrow_mut().clear());
    HTTP_REQUESTS.with(|r| r.borrow_mut().clear());
    gpu_array_clear();
    mem_table_clear();
    lib_cache_clear();
    let mut total_elements = 0;

    // RNG state: seeded from --set seed=N or system time
    let seed: u64 = if let Some(v) = overrides.scalars.get("seed") {
        v.as_float().map_err(|_| CliError::Compile("seed must be a number".into()))? as u64
    } else {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    };
    let rng = Cell::new(seed);

    // Try to initialize GPU; fall back to CPU if unavailable
    let gpu = match octoflow_vulkan::VulkanCompute::new() {
        Ok(g) => Some(g),
        Err(_) => None,
    };
    // Store raw pointer for fast GPU→CPU downloads in helpers (single-threaded, safe).
    if let Some(ref g) = gpu {
        GPU_DEVICE_PTR.with(|c| c.set(g as *const octoflow_vulkan::VulkanCompute as usize));
        GPU_SUPPORTS_F16.with(|c| c.set(g.supports_f16));
    }
    // Notify once when running without GPU
    if gpu.is_none() {
        thread_local! { static GPU_MSG_SHOWN: Cell<bool> = Cell::new(false); }
        GPU_MSG_SHOWN.with(|shown| {
            if !shown.get() {
                eprintln!("[info] No GPU detected — using CPU fallback (slower but functional)");
                shown.set(true);
            }
        });
    }

    for (stmt, span) in &program.statements {
        match stmt {
            Statement::FnDecl { name, params, body } => {
                functions.insert(name.clone(), FnDef {
                    params: params.clone(),
                    body: body.clone(),
                });
            }
            Statement::UseDecl { module } => {
                import_module(base_dir, module, &mut functions, &mut scalar_fns,
                              &mut struct_defs, &mut scalars, &mut arrays, &mut hashmaps, &mut mutable_scalars)?;
            }
            Statement::StructDef { name, fields } => {
                struct_defs.insert(name.clone(), fields.clone());
            }
            Statement::ExternBlock { library, functions: ext_fns, .. } => {
                register_extern_block(library, ext_fns);
            }
            Statement::ArrayDecl { name, elements, mutable } => {
                let mut values = Vec::new();
                for elem in elements {
                    let val = eval_scalar(elem, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)
                        .map_err(|e| e.with_line(span.line))?;
                    values.push(val);
                }
                arrays.insert(name.clone(), values);
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::StreamDecl { name, expr } => {
                let data = eval_expr(expr, &streams, &scalars, &gpu, base_dir, &functions,
                                     &mut image_dims)?;
                total_elements += data.len();
                if let Some(dims) = find_source_dims(expr, &image_dims) {
                    image_dims.insert(name.clone(), dims);
                }
                streams.insert(name.clone(), data);
            }
            Statement::LetDecl { name, value, mutable } => {
                // Check for vec or struct constructor: let v = vec3(...) or let p = Point(...)
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    // Built-in vec constructors
                    if let Some(dim) = match fn_name.as_str() {
                        "vec2" => Some(2usize),
                        "vec3" => Some(3usize),
                        "vec4" => Some(4usize),
                        _ => None,
                    } {
                        if args.len() != dim {
                            return Err(CliError::Compile(format!(
                                "{}() requires {} components, got {}",
                                fn_name, dim, args.len()
                            )));
                        }
                        let components = ["x", "y", "z", "w"];
                        for (i, arg) in args.iter().enumerate() {
                            let val = eval_scalar(arg, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                            scalars.insert(format!("{}.{}", name, components[i]), Value::Float(val.as_float()?));
                        }
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // User-defined struct constructors
                    if let Some(fields) = struct_defs.get(fn_name) {
                        if args.len() != fields.len() {
                            return Err(CliError::Compile(format!(
                                "{}() requires {} fields, got {}",
                                fn_name, fields.len(), args.len()
                            )));
                        }
                        for (i, arg) in args.iter().enumerate() {
                            let val = eval_scalar(arg, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                            scalars.insert(format!("{}.{}", name, &fields[i]), Value::Float(val.as_float()?));
                        }
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // try() error handling
                    if fn_name == "try" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("try() requires exactly 1 argument".into()));
                        }
                        match eval_scalar(&args[0], &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars) {
                            Ok(val) => {
                                scalars.insert(format!("{}.value", name), val);
                                scalars.insert(format!("{}.ok", name), Value::Float(1.0));
                                scalars.insert(format!("{}.error", name), Value::Str(String::new()));
                            }
                            Err(e) => {
                                scalars.insert(format!("{}.value", name), Value::Str(String::new()));
                                scalars.insert(format!("{}.ok", name), Value::Float(0.0));
                                scalars.insert(format!("{}.error", name), Value::Str(format!("{}", e)));
                            }
                        }
                        continue;
                    }
                    // read_image(path) → decompose into name.width, name.height + channel arrays
                    if fn_name == "read_image" && args.len() == 1 {
                        let path_val = eval_scalar(&args[0], &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                        let path = path_val.as_str().map_err(|_| CliError::Compile("read_image() path must be a string".into()))?.to_string();
                        check_read_permission_for(&path)?;
                        let (pixels, w, h) = crate::image_io::read_image(&path)?;
                        let n = (w as usize) * (h as usize);
                        let mut r_arr = Vec::with_capacity(n);
                        let mut g_arr = Vec::with_capacity(n);
                        let mut b_arr = Vec::with_capacity(n);
                        for i in 0..n {
                            r_arr.push(Value::Float(pixels[i * 3]));
                            g_arr.push(Value::Float(pixels[i * 3 + 1]));
                            b_arr.push(Value::Float(pixels[i * 3 + 2]));
                        }
                        let r_name = format!("{}.r", name);
                        let g_name = format!("{}.g", name);
                        let b_name = format!("{}.b", name);
                        arrays.insert(r_name.clone(), r_arr);
                        arrays.insert(g_name.clone(), g_arr);
                        arrays.insert(b_name.clone(), b_arr);
                        scalars.insert(format!("{}.width", name), Value::Float(w as f32));
                        scalars.insert(format!("{}.height", name), Value::Float(h as f32));
                        scalars.insert(format!("{}.r", name), Value::Str(r_name));
                        scalars.insert(format!("{}.g", name), Value::Str(g_name));
                        scalars.insert(format!("{}.b", name), Value::Str(b_name));
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // HTTP client functions
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        check_net_permission()?;
                        let (method, expected_args) = match fn_name.as_str() {
                            "http_get" => ("GET", 1), "http_post" => ("POST", 2),
                            "http_put" => ("PUT", 2), _ => ("DELETE", 1),
                        };
                        if args.len() != expected_args {
                            return Err(CliError::Compile(format!("{}() requires {} argument(s), got {}", fn_name, expected_args, args.len())));
                        }
                        let url_val = eval_scalar(&args[0], &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                        let url = url_val.as_str().map_err(|_| CliError::Compile(format!("{}() URL must be a string", fn_name)))?.to_string();
                        let body_str = if expected_args == 2 {
                            let b = eval_scalar(&args[1], &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                            Some(b.as_str().map_err(|_| CliError::Compile(format!("{}() body must be a string", fn_name)))?.to_string())
                        } else { None };
                        let (status, body, ok, error) = do_http_request(method, &url, body_str.as_deref());
                        scalars.insert(format!("{}.status", name), Value::Float(status));
                        scalars.insert(format!("{}.body", name), Value::Str(body));
                        scalars.insert(format!("{}.ok", name), Value::Float(ok));
                        scalars.insert(format!("{}.error", name), Value::Str(error));
                        continue;
                    }
                    // Command execution: exec(cmd, ...args)
                    if fn_name == "exec" {
                        check_exec_permission()?;
                        if args.is_empty() {
                            return Err(CliError::Compile("exec() requires at least 1 argument (command)".into()));
                        }
                        let mut cmd_args = Vec::new();
                        for arg in args {
                            let val = eval_scalar(arg, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                            cmd_args.push(val.to_string());
                        }
                        let command = &cmd_args[0];
                        let command_args = &cmd_args[1..];

                        use std::process::Command;
                        let output = Command::new(command)
                            .args(command_args)
                            .output()
                            .map_err(|e| CliError::Io(format!("exec(\"{}\"): {}", command, e)))?;

                        let status_code = output.status.code().unwrap_or(-1) as f32;
                        let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
                        let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();
                        let ok = if output.status.success() { 1.0 } else { 0.0 };

                        scalars.insert(format!("{}.status", name), Value::Float(status_code));
                        scalars.insert(format!("{}.output", name), Value::Str(stdout_str));
                        scalars.insert(format!("{}.ok", name), Value::Float(ok));
                        scalars.insert(format!("{}.error", name), Value::Str(stderr_str));
                        continue;
                    }
                    // video_open(byte_array) → handle + .width .height .frames .fps scalars
                    if fn_name == "video_open" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("video_open() requires 1 argument (byte array name)".into()));
                        }
                        let arr_name = match &args[0] {
                            ScalarExpr::Ref(n) => n.clone(),
                            _ => return Err(CliError::Compile("video_open() argument must be an array name".into())),
                        };
                        let bytes = video_open_extract_bytes(&arr_name, &arrays)?;
                        let (handle, w, h, fc, fps) = video_open_from_bytes(bytes)?;
                        video_open_register(handle, name, w, h, fc, fps, &mut scalars);
                        continue;
                    }
                    if fn_name == "video_open_file" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("video_open_file() requires 1 argument (file path)".into()));
                        }
                        let path_val = eval_scalar(&args[0], &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                        let path = path_val.as_str().map_err(|_| CliError::Compile("video_open_file() path must be a string".into()))?.to_string();
                        check_read_permission_for(&path)?;
                        let (handle, w, h, fc, fps) = video_open_file_bytes(&path)?;
                        video_open_register(handle, name, w, h, fc, fps, &mut scalars);
                        continue;
                    }
                    // video_frame(handle, index) → .r .g .b arrays
                    if fn_name == "video_frame" {
                        if args.len() != 2 {
                            return Err(CliError::Compile("video_frame() requires 2 arguments (handle, frame_index)".into()));
                        }
                        let handle_val = eval_scalar(&args[0], &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                        let idx_val = eval_scalar(&args[1], &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                        let handle_id = handle_val.as_float().map_err(|_| CliError::Compile("video_frame(): handle must be numeric".into()))? as u32;
                        let frame_idx = idx_val.as_float().map_err(|_| CliError::Compile("video_frame(): index must be numeric".into()))? as usize;
                        let (r, g, b) = video_frame_decode(handle_id, frame_idx)?;
                        gpu_array_insert(format!("{}.r", name), r);
                        gpu_array_insert(format!("{}.g", name), g);
                        gpu_array_insert(format!("{}.b", name), b);
                        continue;
                    }
                    // Extern FFI function call: let r = SomeCFn(args...)
                    if is_extern_fn(fn_name) {
                        let mut arg_vals = Vec::new();
                        for arg in args {
                            arg_vals.push(eval_scalar(arg, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?);
                        }
                        let result = call_extern_fn(fn_name, &arg_vals)?;
                        scalars.insert(name.clone(), result);
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // Array-returning functions: read_lines, list_dir, split
                    if let Some(result) = eval_array_fn(fn_name, args, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)
                        .map_err(|e| e.with_line(span.line))? {
                        match result {
                            ArrayResult::Values(arr) => { arrays.insert(name.clone(), arr); }
                            ArrayResult::GpuFloats(arr) => { gpu_array_insert(name.clone(), arr); }
                            ArrayResult::Resident(buf) => { gpu_array_insert_resident(name.clone(), buf); }
                        }
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                    // Hashmap-returning functions: json_parse
                    if let Some(hm) = eval_hashmap_fn(fn_name, args, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)? {
                        hashmaps.insert(name.clone(), hm);
                        if *mutable { mutable_scalars.insert(name.clone()); }
                        continue;
                    }
                }

                let result = if let Some(val) = overrides.scalars.get(name) {
                    val.clone()
                } else {
                    eval_scalar(value, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)
                        .map_err(|e| e.with_line(span.line))?
                };
                // Check if a user function returned an array/map via the side-channel
                let returned_arr = RETURNED_ARRAY.with(|r| r.borrow_mut().take());
                if let Some(arr) = returned_arr {
                    arrays.insert(name.clone(), arr);
                } else {
                    let returned_hm = RETURNED_MAP.with(|r| r.borrow_mut().take());
                    if let Some(hm) = returned_hm {
                        hashmaps.insert(name.clone(), hm);
                    } else {
                        scalars.insert(name.clone(), result);
                    }
                }
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::Assign { name, value } => {
                if !mutable_scalars.contains(name) {
                    return Err(CliError::Compile(format!(
                        "Cannot assign to '{}': not declared as mutable (use 'let mut')", name
                    )).with_line(span.line));
                }
                let result = eval_scalar(value, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)
                    .map_err(|e| e.with_line(span.line))?;
                scalars.insert(name.clone(), result);
            }
            Statement::ArrayAssign { array, index, value } => {
                if !mutable_scalars.contains(array.as_str()) {
                    return Err(CliError::Compile(format!(
                        "Cannot assign to '{}': not declared as mutable (use 'let mut')", array
                    )));
                }
                gpu_array_materialize(array, &mut arrays);
                let idx = eval_scalar(index, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?.as_float()? as usize;
                let val = eval_scalar(value, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                let arr = arrays.get_mut(array)
                    .ok_or_else(|| CliError::Compile(format!("undefined array '{}'", array)))?;
                if idx >= arr.len() {
                    return Err(CliError::Compile(format!(
                        "Array index out of bounds: {}[{}] (length {})", array, idx, arr.len()
                    )));
                }
                arr[idx] = val;
            }
            Statement::ArrayPush { array, value } => {
                if !mutable_scalars.contains(array.as_str()) {
                    mutable_scalars.insert(array.clone());  // Auto-promote to mutable on push
                }
                gpu_array_materialize(array, &mut arrays);
                let val = eval_scalar(value, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                let arr = arrays.get_mut(array)
                    .ok_or_else(|| CliError::Compile(format!("undefined array '{}'", array)))?;
                arr.push(val);
            }
            Statement::Emit { expr, path } => {
                let data = match expr {
                    Expr::Ref { name } => streams
                        .get(name)
                        .ok_or_else(|| CliError::UndefinedStream(name.clone()))?,
                    _ => return Err(CliError::Compile("emit() argument must be a stream name".into())),
                };
                let full_path = resolve_path(base_dir, path)?;
                if crate::image_io::is_image_path(&full_path) {
                    let stream_name = match expr {
                        Expr::Ref { name } => name,
                        _ => unreachable!(),
                    };
                    let (w, h) = image_dims.get(stream_name)
                        .copied()
                        .ok_or_else(|| CliError::Compile(format!(
                            "cannot write image '{}': no image dimensions found for stream '{}'",
                            path, stream_name)))?;
                    crate::image_io::write_image(&full_path, data, w, h)?;
                } else if crate::octo_io::is_octo_path(&full_path) {
                    crate::octo_io::write_octo(&full_path, data)?;
                } else {
                    csv_write_floats(&full_path, data)?;
                }
            }
            Statement::Print { segments } => {
                let mut output = String::new();
                for seg in segments {
                    match seg {
                        PrintSegment::Literal(s) => output.push_str(s),
                        PrintSegment::Scalar { name, precision } => {
                            if let Some(value) = scalars.get(name) {
                                match (precision, value) {
                                    (Some(p), Value::Float(f)) => output.push_str(&format!("{:.prec$}", f, prec = *p)),
                                    (Some(_p), Value::Int(i)) => output.push_str(&format!("{}", i)),
                                    (Some(_), Value::Str(s)) => output.push_str(s),
                                    (Some(_), Value::Map(_)) => output.push_str(&format!("{}", value)),
                                    (Some(_), Value::None) => output.push_str("none"),
                                    (None, _) => output.push_str(&format!("{}", value)),
                                }
                            } else if let Some(s) = gpu_array_format(name) {
                                output.push_str(&s);
                            } else if let Some(arr) = arrays.get(name) {
                                let items: Vec<String> = arr.iter().map(|v| format!("{}", v)).collect();
                                output.push_str(&format!("[{}]", items.join(", ")));
                            } else if let Some(hm) = hashmaps.get(name) {
                                let mut keys: Vec<&String> = hm.keys().collect();
                                keys.sort();
                                let items: Vec<String> = keys.iter().map(|k| format!("{}={}", k, hm[*k])).collect();
                                output.push_str(&format!("{{{}}}", items.join(", ")));
                            } else {
                                return Err(CliError::UndefinedScalar(name.clone()));
                            }
                        }
                        PrintSegment::Expr(e) => {
                            let val = eval_scalar(e, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                            output.push_str(&format!("{}", val));
                        }
                    }
                }
                runtime_println(&output);
            }
            Statement::WhileLoop { condition, body } => {
                let max_while2 = overrides.max_iters.unwrap_or(10_000_000);
                let mut iterations = 0;
                loop {
                    let cond = eval_scalar(condition, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?.as_float()?;
                    if cond == 0.0 {
                        break;
                    }
                    iterations += 1;
                    if iterations > max_while2 {
                        return Err(CliError::Compile(format!(
                            "while loop exceeded {} iterations (infinite loop?)", max_while2
                        )));
                    }
                    match execute_loop_body(body, &streams, &mut scalars, &mut mutable_scalars, &struct_defs, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &rng)? {
                        LoopControl::Break => break,
                        LoopControl::Continue => continue,
                        LoopControl::Return(_) => return Err(CliError::Compile("'return' can only be used inside a function".into())),
                        LoopControl::Normal => {}
                    }
                }
            }
            Statement::ForLoop { var, start, end, body } => {
                const MAX_FOR_ITERATIONS: usize = 10_000_000;
                let start_val = eval_scalar(start, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?.as_float()? as i64;
                let end_val = eval_scalar(end, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?.as_float()? as i64;
                let count = if end_val > start_val { (end_val - start_val) as usize } else { 0 };
                if count > MAX_FOR_ITERATIONS {
                    return Err(CliError::Compile(format!(
                        "for loop range {} to {} is {} iterations (max {})", start_val, end_val, count, MAX_FOR_ITERATIONS
                    )));
                }
                for i in start_val..end_val {
                    scalars.insert(var.clone(), Value::Float(i as f32));
                    match execute_loop_body(body, &streams, &mut scalars, &mut mutable_scalars, &struct_defs, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &rng)? {
                        LoopControl::Break => break,
                        LoopControl::Continue => continue,
                        LoopControl::Return(_) => return Err(CliError::Compile("'return' can only be used inside a function".into())),
                        LoopControl::Normal => {}
                    }
                }
                scalars.remove(var);
            }
            Statement::ForEachLoop { var, iterable, body } => {
                gpu_array_materialize(iterable, &mut arrays);
                let arr = arrays.get(iterable)
                    .ok_or_else(|| CliError::Compile(format!("undefined array '{}' in for-each loop", iterable)))?
                    .clone();
                for val in &arr {
                    scalars.insert(var.clone(), val.clone());
                    match execute_loop_body(body, &streams, &mut scalars, &mut mutable_scalars, &struct_defs, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &rng)? {
                        LoopControl::Break => break,
                        LoopControl::Continue => continue,
                        LoopControl::Return(_) => return Err(CliError::Compile("'return' can only be used inside a function".into())),
                        LoopControl::Normal => {}
                    }
                }
                scalars.remove(var);
            }
            Statement::IfBlock { condition, body, elif_branches, else_body } => {
                let cond = eval_scalar(condition, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?.as_float()?;
                if cond != 0.0 {
                    for (s, _sp) in body {
                        execute_block_stmt(s, &streams, &mut scalars, &mut mutable_scalars, &struct_defs, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &rng)?;
                    }
                } else {
                    let mut matched = false;
                    for (elif_cond, elif_body) in elif_branches {
                        let ec = eval_scalar(elif_cond, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?.as_float()?;
                        if ec != 0.0 {
                            for (s, _sp) in elif_body {
                                execute_block_stmt(s, &streams, &mut scalars, &mut mutable_scalars, &struct_defs, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &rng)?;
                            }
                            matched = true;
                            break;
                        }
                    }
                    if !matched && !else_body.is_empty() {
                        for (s, _sp) in else_body {
                            execute_block_stmt(s, &streams, &mut scalars, &mut mutable_scalars, &struct_defs, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &rng)?;
                        }
                    }
                }
            }
            Statement::ScalarFnDecl { name, params, body } => {
                scalar_fns.insert(name.clone(), ScalarFnDef {
                    params: params.clone(),
                    body: body.clone(),
                    captured_scalars: HashMap::new(),
                    captured_arrays: HashMap::new(),
                });
            }
            Statement::MapDecl { name, mutable } => {
                hashmaps.insert(name.clone(), HashMap::new());
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::MapInsert { map, key, value } => {
                if !mutable_scalars.contains(map.as_str()) {
                    return Err(CliError::Compile(format!(
                        "Cannot insert into '{}': not declared as mutable (use 'let mut')", map
                    )));
                }
                let key_val = eval_scalar(key, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?.as_str()?.to_string();
                let val = eval_scalar(value, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                let hm = hashmaps.get_mut(map)
                    .ok_or_else(|| CliError::Compile(format!("undefined map '{}'", map)))?;
                hm.insert(key_val, val);
            }
            Statement::WriteFile { path, content } => {
                let path_val = eval_scalar(path, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("write_file() path must be a string".into()))?;
                check_write_permission_for(p)?;
                let content_val = eval_scalar(content, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                let c = content_val.to_string();
                std::fs::write(p, &c).map_err(|e| CliError::Io(format!("write_file(\"{}\"): {}", p, e)))?;
            }
            Statement::AppendFile { path, content } => {
                let path_val = eval_scalar(path, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("append_file() path must be a string".into()))?;
                check_write_permission_for(p)?;
                let content_val = eval_scalar(content, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                let c = content_val.to_string();
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new().create(true).append(true).open(p)
                    .map_err(|e| CliError::Io(format!("append_file(\"{}\"): {}", p, e)))?;
                file.write_all(c.as_bytes()).map_err(|e| CliError::Io(format!("append_file(\"{}\"): {}", p, e)))?;
            }
            Statement::SaveData { path, map_name } => {
                let path_val = eval_scalar(path, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("save_data() path must be a string".into()))?.to_string();
                check_write_permission_for(&p)?;
                let hm = hashmaps.get(map_name)
                    .ok_or_else(|| CliError::Compile(format!("save_data(): undefined map '{}'", map_name)))?;
                let content = serialize_od(hm);
                std::fs::write(&p, &content).map_err(|e| CliError::Io(format!("save_data(\"{}\"): {}", p, e)))?;
            }
            Statement::WriteCsv { path, array_name } => {
                let path_val = eval_scalar(path, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("write_csv() path must be a string".into()))?.to_string();
                check_write_permission_for(&p)?;
                gpu_array_materialize(array_name, &mut arrays);
                let arr = arrays.get(array_name)
                    .ok_or_else(|| CliError::Compile(format!("write_csv(): undefined array '{}'", array_name)))?;
                csv_write_structured(&p, arr)?;
            }
            Statement::WriteBytes { path, array_name } => {
                let path_val = eval_scalar(path, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("write_bytes() path must be a string".into()))?.to_string();
                check_write_permission_for(&p)?;
                gpu_array_materialize(array_name, &mut arrays);
                let arr = arrays.get(array_name)
                    .ok_or_else(|| CliError::Compile(format!("write_bytes(): undefined array '{}'", array_name)))?;
                do_write_bytes(&p, arr)?;
            }
            Statement::Return { .. } => {
                return Err(CliError::Compile("'return' can only be used inside a function".into()));
            }
            Statement::Break => {
                return Err(CliError::Compile("'break' can only be used inside a loop".into()));
            }
            Statement::Continue => {
                return Err(CliError::Compile("'continue' can only be used inside a loop".into()));
            }
            Statement::ExprStmt { expr } => {
                let _ = eval_scalar(expr, &streams, &scalars, &gpu, &mut arrays, &mut hashmaps, &scalar_fns, &struct_defs, &rng, &mutable_scalars)
                    .map_err(|e| e.with_line(span.line))?;
                // R-05: Apply scalar writeback from user function calls
                SCALAR_WRITEBACK.with(|sw| {
                    if let Some(writes) = sw.borrow_mut().take() {
                        for (name, value) in writes {
                            scalars.insert(name, value);
                        }
                    }
                });
            }
        }
    }

    // Clear GPU caches before VulkanCompute is dropped — prevents stale Vulkan handles.
    GPU_BUFFER_CACHE.with(|gc| gc.borrow_mut().clear());
    GPU_DEVICE_PTR.with(|c| c.set(0));
    GPU_SUPPORTS_F16.with(|c| c.set(false));
    Ok((total_elements, gpu.is_some()))
}

/// Control flow signal from loop body execution.
enum LoopControl {
    /// Continue normal execution.
    Normal,
    /// `break` — exit the current loop.
    Break,
    /// `continue` — skip to the next iteration.
    Continue,
    /// `return <value>` — return from a scalar function.
    Return(Value),
}

/// Execute a loop body — handles let, assign, print, array, nested loops, break, continue.
fn execute_loop_body(
    body: &[(Statement, octoflow_parser::ast::Span)],
    streams: &HashMap<String, Vec<f32>>,
    scalars: &mut HashMap<String, Value>,
    mutable_scalars: &mut std::collections::HashSet<String>,
    struct_defs: &HashMap<String, Vec<String>>,
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    arrays: &mut HashMap<String, Vec<Value>>,
    hashmaps: &mut HashMap<String, HashMap<String, Value>>,
    scalar_fns: &HashMap<String, ScalarFnDef>,
    rng: &Cell<u64>,
) -> Result<LoopControl, CliError> {
    for (body_stmt, _body_span) in body {
        match body_stmt {
            Statement::LetDecl { name: bname, value, mutable } => {
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    if let Some(dim) = match fn_name.as_str() {
                        "vec2" => Some(2usize), "vec3" => Some(3usize), "vec4" => Some(4usize), _ => None,
                    } {
                        if args.len() != dim {
                            return Err(CliError::Compile(format!("{}() requires {} components, got {}", fn_name, dim, args.len())));
                        }
                        let components = ["x", "y", "z", "w"];
                        for (i, arg) in args.iter().enumerate() {
                            let val = eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
                            scalars.insert(format!("{}.{}", bname, components[i]), Value::Float(val));
                        }
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    if let Some(fields) = struct_defs.get(fn_name) {
                        if args.len() != fields.len() {
                            return Err(CliError::Compile(format!("{}() requires {} fields, got {}", fn_name, fields.len(), args.len())));
                        }
                        let fields = fields.clone();
                        for (i, arg) in args.iter().enumerate() {
                            let val = eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
                            scalars.insert(format!("{}.{}", bname, &fields[i]), Value::Float(val));
                        }
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    // try() error handling
                    if fn_name == "try" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("try() requires exactly 1 argument".into()));
                        }
                        match eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars) {
                            Ok(val) => {
                                scalars.insert(format!("{}.value", bname), val);
                                scalars.insert(format!("{}.ok", bname), Value::Float(1.0));
                                scalars.insert(format!("{}.error", bname), Value::Str(String::new()));
                            }
                            Err(e) => {
                                scalars.insert(format!("{}.value", bname), Value::Str(String::new()));
                                scalars.insert(format!("{}.ok", bname), Value::Float(0.0));
                                scalars.insert(format!("{}.error", bname), Value::Str(format!("{}", e)));
                            }
                        }
                        continue;
                    }
                    // HTTP client functions
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        check_net_permission()?;
                        let (method, expected_args) = match fn_name.as_str() {
                            "http_get" => ("GET", 1), "http_post" => ("POST", 2),
                            "http_put" => ("PUT", 2), _ => ("DELETE", 1),
                        };
                        if args.len() != expected_args {
                            return Err(CliError::Compile(format!("{}() requires {} argument(s), got {}", fn_name, expected_args, args.len())));
                        }
                        let url_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                        let url = url_val.as_str().map_err(|_| CliError::Compile(format!("{}() URL must be a string", fn_name)))?.to_string();
                        let body_str = if expected_args == 2 {
                            let b = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                            Some(b.as_str().map_err(|_| CliError::Compile(format!("{}() body must be a string", fn_name)))?.to_string())
                        } else { None };
                        let (status, body, ok, error) = do_http_request(method, &url, body_str.as_deref());
                        scalars.insert(format!("{}.status", bname), Value::Float(status));
                        scalars.insert(format!("{}.body", bname), Value::Str(body));
                        scalars.insert(format!("{}.ok", bname), Value::Float(ok));
                        scalars.insert(format!("{}.error", bname), Value::Str(error));
                        continue;
                    }
                    // Command execution: exec(cmd, ...args)
                    if fn_name == "exec" {
                        check_exec_permission()?;
                        if args.is_empty() {
                            return Err(CliError::Compile("exec() requires at least 1 argument (command)".into()));
                        }
                        let mut cmd_args = Vec::new();
                        for arg in args {
                            let val = eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                            cmd_args.push(val.to_string());
                        }
                        let command = &cmd_args[0];
                        let command_args = &cmd_args[1..];

                        use std::process::Command;
                        let output = Command::new(command)
                            .args(command_args)
                            .output()
                            .map_err(|e| CliError::Io(format!("exec(\"{}\"): {}", command, e)))?;

                        let status_code = output.status.code().unwrap_or(-1) as f32;
                        let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
                        let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();
                        let ok = if output.status.success() { 1.0 } else { 0.0 };

                        scalars.insert(format!("{}.status", bname), Value::Float(status_code));
                        scalars.insert(format!("{}.output", bname), Value::Str(stdout_str));
                        scalars.insert(format!("{}.ok", bname), Value::Float(ok));
                        scalars.insert(format!("{}.error", bname), Value::Str(stderr_str));
                        continue;
                    }
                    // video_open(byte_array) → handle + .width .height .frames .fps scalars
                    if fn_name == "video_open" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("video_open() requires 1 argument (byte array name)".into()));
                        }
                        let arr_name = match &args[0] {
                            ScalarExpr::Ref(n) => n.clone(),
                            _ => return Err(CliError::Compile("video_open() argument must be an array name".into())),
                        };
                        let bytes = video_open_extract_bytes(&arr_name, arrays)?;
                        let (handle, w, h, fc, fps) = video_open_from_bytes(bytes)?;
                        video_open_register(handle, bname, w, h, fc, fps, scalars);
                        continue;
                    }
                    if fn_name == "video_open_file" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("video_open_file() requires 1 argument (file path)".into()));
                        }
                        let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                        let path = path_val.as_str().map_err(|_| CliError::Compile("video_open_file() path must be a string".into()))?.to_string();
                        check_read_permission_for(&path)?;
                        let (handle, w, h, fc, fps) = video_open_file_bytes(&path)?;
                        video_open_register(handle, bname, w, h, fc, fps, scalars);
                        continue;
                    }
                    // video_frame(handle, index) → .r .g .b arrays
                    if fn_name == "video_frame" {
                        if args.len() != 2 {
                            return Err(CliError::Compile("video_frame() requires 2 arguments (handle, frame_index)".into()));
                        }
                        let handle_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                        let idx_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                        let handle_id = handle_val.as_float().map_err(|_| CliError::Compile("video_frame(): handle must be numeric".into()))? as u32;
                        let frame_idx = idx_val.as_float().map_err(|_| CliError::Compile("video_frame(): index must be numeric".into()))? as usize;
                        let (r, g, b) = video_frame_decode(handle_id, frame_idx)?;
                        gpu_array_insert(format!("{}.r", bname), r);
                        gpu_array_insert(format!("{}.g", bname), g);
                        gpu_array_insert(format!("{}.b", bname), b);
                        continue;
                    }
                    // Extern FFI function call: let r = SomeCFn(args...)
                    if is_extern_fn(fn_name) {
                        let mut arg_vals = Vec::new();
                        for arg in args {
                            arg_vals.push(eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?);
                        }
                        let result = call_extern_fn(fn_name, &arg_vals)?;
                        scalars.insert(bname.clone(), result);
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    // Array-returning functions: read_lines, list_dir, split
                    if let Some(result) = eval_array_fn(fn_name, args, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)? {
                        match result {
                            ArrayResult::Values(arr) => { arrays.insert(bname.clone(), arr); }
                            ArrayResult::GpuFloats(arr) => { gpu_array_insert(bname.clone(), arr); }
                            ArrayResult::Resident(buf) => { gpu_array_insert_resident(bname.clone(), buf); }
                        }
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                    // Hashmap-returning functions: json_parse
                    if let Some(hm) = eval_hashmap_fn(fn_name, args, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)? {
                        hashmaps.insert(bname.clone(), hm);
                        if *mutable { mutable_scalars.insert(bname.clone()); }
                        continue;
                    }
                }
                let result = eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                // Check if a user function returned an array/map via the side-channel
                let returned_arr = RETURNED_ARRAY.with(|r| r.borrow_mut().take());
                if let Some(arr) = returned_arr {
                    arrays.insert(bname.clone(), arr);
                } else {
                    let returned_hm = RETURNED_MAP.with(|r| r.borrow_mut().take());
                    if let Some(hm) = returned_hm {
                        hashmaps.insert(bname.clone(), hm);
                    } else {
                        scalars.insert(bname.clone(), result);
                    }
                }
                // R-05: Apply scalar writeback from user function calls
                SCALAR_WRITEBACK.with(|sw| {
                    if let Some(writes) = sw.borrow_mut().take() {
                        for (name, value) in writes {
                            scalars.insert(name, value);
                        }
                    }
                });
                if *mutable { mutable_scalars.insert(bname.clone()); }
            }
            Statement::Assign { name: aname, value } => {
                if !mutable_scalars.contains(aname) {
                    return Err(CliError::Compile(format!(
                        "Cannot assign to '{}': not declared as mutable (use 'let mut')", aname
                    )));
                }
                let result = eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                scalars.insert(aname.clone(), result);
            }
            Statement::ArrayAssign { array, index, value } => {
                if !mutable_scalars.contains(array.as_str()) {
                    return Err(CliError::Compile(format!(
                        "Cannot assign to '{}': not declared as mutable (use 'let mut')", array
                    )));
                }
                gpu_array_materialize(array, arrays);
                let idx = eval_scalar(index, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let val = eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let arr = arrays.get_mut(array)
                    .ok_or_else(|| CliError::Compile(format!("undefined array '{}'", array)))?;
                if idx >= arr.len() {
                    return Err(CliError::Compile(format!(
                        "Array index out of bounds: {}[{}] (length {})", array, idx, arr.len()
                    )));
                }
                arr[idx] = val;
            }
            Statement::ArrayPush { array, value } => {
                if !mutable_scalars.contains(array.as_str()) {
                    mutable_scalars.insert(array.clone());  // Auto-promote to mutable on push
                }
                gpu_array_materialize(array, arrays);
                let val = eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let arr = arrays.get_mut(array)
                    .ok_or_else(|| CliError::Compile(format!("undefined array '{}'", array)))?;
                arr.push(val);
            }
            Statement::Print { segments: segs } => {
                let mut out = String::new();
                for seg in segs {
                    match seg {
                        PrintSegment::Literal(s) => out.push_str(s),
                        PrintSegment::Scalar { name: sname, precision } => {
                            if let Some(value) = scalars.get(sname) {
                                match (precision, value) {
                                    (Some(p), Value::Float(f)) => out.push_str(&format!("{:.prec$}", f, prec = *p)),
                                    (Some(_p), Value::Int(i)) => out.push_str(&format!("{}", i)),
                                    (Some(_), Value::Str(s)) => out.push_str(s),
                                    (Some(_), Value::Map(_)) => out.push_str(&format!("{}", value)),
                                    (Some(_), Value::None) => out.push_str("none"),
                                    (None, _) => out.push_str(&format!("{}", value)),
                                }
                            } else if let Some(s) = gpu_array_format(sname.as_str()) {
                                out.push_str(&s);
                            } else if let Some(arr) = arrays.get(sname.as_str()) {
                                let items: Vec<String> = arr.iter().map(|v| format!("{}", v)).collect();
                                out.push_str(&format!("[{}]", items.join(", ")));
                            } else if let Some(hm) = hashmaps.get(sname.as_str()) {
                                let mut keys: Vec<&String> = hm.keys().collect();
                                keys.sort();
                                let items: Vec<String> = keys.iter().map(|k| format!("{}={}", k, hm[*k])).collect();
                                out.push_str(&format!("{{{}}}", items.join(", ")));
                            } else {
                                return Err(CliError::UndefinedScalar(sname.clone()));
                            }
                        }
                        PrintSegment::Expr(e) => {
                            let val = eval_scalar(e, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                            out.push_str(&format!("{}", val));
                        }
                    }
                }
                runtime_println(&out);
            }
            Statement::ArrayDecl { name: aname, elements, mutable } => {
                let mut values = Vec::new();
                for elem in elements {
                    let val = eval_scalar(elem, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    values.push(val);
                }
                arrays.insert(aname.clone(), values);
                if *mutable { mutable_scalars.insert(aname.clone()); }
            }
            Statement::WhileLoop { condition, body: inner_body } => {
                const MAX_WHILE_ITERATIONS: usize = 10_000_000;
                let mut iterations = 0;
                loop {
                    let cond = eval_scalar(condition, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
                    if cond == 0.0 {
                        break;
                    }
                    iterations += 1;
                    if iterations > MAX_WHILE_ITERATIONS {
                        return Err(CliError::Compile(format!(
                            "while loop exceeded {} iterations (infinite loop?)", MAX_WHILE_ITERATIONS
                        )));
                    }
                    match execute_loop_body(inner_body, streams, scalars, mutable_scalars, struct_defs, gpu, arrays, hashmaps, scalar_fns, rng)? {
                        LoopControl::Break => break,
                        LoopControl::Continue => continue,
                        LoopControl::Return(v) => return Ok(LoopControl::Return(v)),
                        LoopControl::Normal => {}
                    }
                }
            }
            Statement::ForLoop { var, start, end, body: inner_body } => {
                const MAX_FOR_ITERATIONS: usize = 10_000_000;
                let start_val = eval_scalar(start, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as i64;
                let end_val = eval_scalar(end, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as i64;
                let count = if end_val > start_val { (end_val - start_val) as usize } else { 0 };
                if count > MAX_FOR_ITERATIONS {
                    return Err(CliError::Compile(format!(
                        "for loop range {} to {} is {} iterations (max {})", start_val, end_val, count, MAX_FOR_ITERATIONS
                    )));
                }
                for i in start_val..end_val {
                    scalars.insert(var.clone(), Value::Float(i as f32));
                    match execute_loop_body(inner_body, streams, scalars, mutable_scalars, struct_defs, gpu, arrays, hashmaps, scalar_fns, rng)? {
                        LoopControl::Break => break,
                        LoopControl::Continue => continue,
                        LoopControl::Return(v) => return Ok(LoopControl::Return(v)),
                        LoopControl::Normal => {}
                    }
                }
                scalars.remove(var);
            }
            Statement::ForEachLoop { var, iterable, body: inner_body } => {
                gpu_array_materialize(iterable, arrays);
                let arr = arrays.get(iterable)
                    .ok_or_else(|| CliError::Compile(format!("undefined array '{}' in for-each loop", iterable)))?
                    .clone();
                for val in &arr {
                    scalars.insert(var.clone(), val.clone());
                    match execute_loop_body(inner_body, streams, scalars, mutable_scalars, struct_defs, gpu, arrays, hashmaps, scalar_fns, rng)? {
                        LoopControl::Break => break,
                        LoopControl::Continue => continue,
                        LoopControl::Return(v) => return Ok(LoopControl::Return(v)),
                        LoopControl::Normal => {}
                    }
                }
                scalars.remove(var);
            }
            Statement::IfBlock { condition, body: if_body, elif_branches, else_body } => {
                let cond = eval_scalar(condition, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
                let chosen_body = if cond != 0.0 {
                    Some(if_body.as_slice())
                } else {
                    let mut found = None;
                    for (elif_cond, elif_body) in elif_branches {
                        let ec = eval_scalar(elif_cond, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
                        if ec != 0.0 {
                            found = Some(elif_body.as_slice());
                            break;
                        }
                    }
                    if found.is_none() && !else_body.is_empty() {
                        found = Some(else_body.as_slice());
                    }
                    found
                };
                if let Some(body_stmts) = chosen_body {
                    let ctrl = execute_loop_body(body_stmts, streams, scalars, mutable_scalars, struct_defs, gpu, arrays, hashmaps, scalar_fns, rng)?;
                    match ctrl {
                        LoopControl::Break => return Ok(LoopControl::Break),
                        LoopControl::Continue => return Ok(LoopControl::Continue),
                        LoopControl::Return(v) => return Ok(LoopControl::Return(v)),
                        LoopControl::Normal => {}
                    }
                }
            }
            Statement::MapDecl { name, mutable } => {
                hashmaps.insert(name.clone(), HashMap::new());
                if *mutable { mutable_scalars.insert(name.clone()); }
            }
            Statement::MapInsert { map, key, value } => {
                if !mutable_scalars.contains(map.as_str()) {
                    return Err(CliError::Compile(format!(
                        "Cannot insert into '{}': not declared as mutable (use 'let mut')", map
                    )));
                }
                let key_val = eval_scalar(key, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_str()?.to_string();
                let val = eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let hm = hashmaps.get_mut(map)
                    .ok_or_else(|| CliError::Compile(format!("undefined map '{}'", map)))?;
                hm.insert(key_val, val);
            }
            Statement::WriteFile { path, content } => {
                let path_val = eval_scalar(path, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("write_file() path must be a string".into()))?;
                check_write_permission_for(p)?;
                let content_val = eval_scalar(content, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let c = content_val.to_string();
                std::fs::write(p, &c).map_err(|e| CliError::Io(format!("write_file(\"{}\"): {}", p, e)))?;
            }
            Statement::AppendFile { path, content } => {
                let path_val = eval_scalar(path, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("append_file() path must be a string".into()))?;
                check_write_permission_for(p)?;
                let content_val = eval_scalar(content, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let c = content_val.to_string();
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new().create(true).append(true).open(p)
                    .map_err(|e| CliError::Io(format!("append_file(\"{}\"): {}", p, e)))?;
                file.write_all(c.as_bytes()).map_err(|e| CliError::Io(format!("append_file(\"{}\"): {}", p, e)))?;
            }
            Statement::SaveData { path, map_name } => {
                let path_val = eval_scalar(path, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("save_data() path must be a string".into()))?.to_string();
                check_write_permission_for(&p)?;
                let hm = hashmaps.get(map_name)
                    .ok_or_else(|| CliError::Compile(format!("save_data(): undefined map '{}'", map_name)))?;
                let content = serialize_od(hm);
                std::fs::write(&p, &content).map_err(|e| CliError::Io(format!("save_data(\"{}\"): {}", p, e)))?;
            }
            Statement::WriteCsv { path, array_name } => {
                let path_val = eval_scalar(path, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("write_csv() path must be a string".into()))?.to_string();
                check_write_permission_for(&p)?;
                gpu_array_materialize(array_name, arrays);
                let arr = arrays.get(array_name)
                    .ok_or_else(|| CliError::Compile(format!("write_csv(): undefined array '{}'", array_name)))?;
                csv_write_structured(&p, arr)?;
            }
            Statement::WriteBytes { path, array_name } => {
                let path_val = eval_scalar(path, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let p = path_val.as_str().map_err(|_| CliError::Compile("write_bytes() path must be a string".into()))?.to_string();
                check_write_permission_for(&p)?;
                gpu_array_materialize(array_name, arrays);
                let arr = arrays.get(array_name)
                    .ok_or_else(|| CliError::Compile(format!("write_bytes(): undefined array '{}'", array_name)))?;
                do_write_bytes(&p, arr)?;
            }
            Statement::Return { value } => {
                // Handle array literal return: return [1, 2, 3]
                if let ScalarExpr::ArrayLiteral(elements) = value {
                    let mut vals = Vec::new();
                    for elem in elements {
                        vals.push(eval_scalar(elem, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?);
                    }
                    RETURNED_ARRAY.with(|r| *r.borrow_mut() = Some(vals));
                    return Ok(LoopControl::Return(Value::Float(0.0)));
                }
                // Try normal scalar eval first; fall back to array return
                match eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars) {
                    Ok(val) => return Ok(LoopControl::Return(val)),
                    Err(CliError::UndefinedScalar(ref name)) => {
                        // Check if the return value is an array name
                        if let Some(arr) = arrays.get(name) {
                            RETURNED_ARRAY.with(|r| *r.borrow_mut() = Some(arr.clone()));
                            return Ok(LoopControl::Return(Value::Float(0.0)));
                        }
                        // Check if the return value is a hashmap name
                        if let Some(hm) = hashmaps.get(name) {
                            RETURNED_MAP.with(|r| *r.borrow_mut() = Some(hm.clone()));
                            return Ok(LoopControl::Return(Value::Float(0.0)));
                        }
                        return Err(CliError::UndefinedScalar(name.clone()));
                    }
                    Err(e) => return Err(e),
                }
            }
            Statement::Break => {
                return Ok(LoopControl::Break);
            }
            Statement::Continue => {
                return Ok(LoopControl::Continue);
            }
            Statement::ExternBlock { library, functions, .. } => {
                // Extern blocks inside loops/functions: re-register (valid use case)
                register_extern_block(library, functions);
            }
            Statement::ExprStmt { expr } => {
                let _ = eval_scalar(expr, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                // R-05: Apply scalar writeback from user function calls
                SCALAR_WRITEBACK.with(|sw| {
                    if let Some(writes) = sw.borrow_mut().take() {
                        for (name, value) in writes {
                            scalars.insert(name, value);
                        }
                    }
                });
            }
            _ => {
                return Err(CliError::Compile(
                    "only let, assignment, print, array, map, while, for, if, break, continue, and return are allowed inside loops/functions".into()
                ));
            }
        }
    }
    Ok(LoopControl::Normal)
}

/// Execute a single statement inside an if-block (top-level, non-loop context).
fn execute_block_stmt(
    stmt: &Statement,
    streams: &HashMap<String, Vec<f32>>,
    scalars: &mut HashMap<String, Value>,
    mutable_scalars: &mut std::collections::HashSet<String>,
    struct_defs: &HashMap<String, Vec<String>>,
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    arrays: &mut HashMap<String, Vec<Value>>,
    hashmaps: &mut HashMap<String, HashMap<String, Value>>,
    scalar_fns: &HashMap<String, ScalarFnDef>,
    rng: &Cell<u64>,
) -> Result<(), CliError> {
    match stmt {
        Statement::LetDecl { name, value, mutable } => {
            if let ScalarExpr::FnCall { name: fn_name, args } = value {
                if let Some(dim) = match fn_name.as_str() {
                    "vec2" => Some(2usize), "vec3" => Some(3usize), "vec4" => Some(4usize), _ => None,
                } {
                    if args.len() != dim {
                        return Err(CliError::Compile(format!("{}() requires {} components, got {}", fn_name, dim, args.len())));
                    }
                    let components = ["x", "y", "z", "w"];
                    for (i, arg) in args.iter().enumerate() {
                        let val = eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
                        scalars.insert(format!("{}.{}", name, components[i]), Value::Float(val));
                    }
                    if *mutable { mutable_scalars.insert(name.clone()); }
                    return Ok(());
                }
                if let Some(fields) = struct_defs.get(fn_name) {
                    if args.len() != fields.len() {
                        return Err(CliError::Compile(format!("{}() requires {} fields, got {}", fn_name, fields.len(), args.len())));
                    }
                    let fields = fields.clone();
                    for (i, arg) in args.iter().enumerate() {
                        let val = eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
                        scalars.insert(format!("{}.{}", name, &fields[i]), Value::Float(val));
                    }
                    if *mutable { mutable_scalars.insert(name.clone()); }
                    return Ok(());
                }
                // try() error handling
                if fn_name == "try" {
                    if args.len() != 1 {
                        return Err(CliError::Compile("try() requires exactly 1 argument".into()));
                    }
                    match eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars) {
                        Ok(val) => {
                            scalars.insert(format!("{}.value", name), val);
                            scalars.insert(format!("{}.ok", name), Value::Float(1.0));
                            scalars.insert(format!("{}.error", name), Value::Str(String::new()));
                        }
                        Err(e) => {
                            scalars.insert(format!("{}.value", name), Value::Str(String::new()));
                            scalars.insert(format!("{}.ok", name), Value::Float(0.0));
                            scalars.insert(format!("{}.error", name), Value::Str(format!("{}", e)));
                        }
                    }
                    return Ok(());
                }
                // HTTP client functions
                if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                    check_net_permission()?;
                    let (method, expected_args) = match fn_name.as_str() {
                        "http_get" => ("GET", 1), "http_post" => ("POST", 2),
                        "http_put" => ("PUT", 2), _ => ("DELETE", 1),
                    };
                    if args.len() != expected_args {
                        return Err(CliError::Compile(format!("{}() requires {} argument(s), got {}", fn_name, expected_args, args.len())));
                    }
                    let url_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let url = url_val.as_str().map_err(|_| CliError::Compile(format!("{}() URL must be a string", fn_name)))?.to_string();
                    let body_str = if expected_args == 2 {
                        let b = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                        Some(b.as_str().map_err(|_| CliError::Compile(format!("{}() body must be a string", fn_name)))?.to_string())
                    } else { None };
                    let (status, body, ok, error) = do_http_request(method, &url, body_str.as_deref());
                    scalars.insert(format!("{}.status", name), Value::Float(status));
                    scalars.insert(format!("{}.body", name), Value::Str(body));
                    scalars.insert(format!("{}.ok", name), Value::Float(ok));
                    scalars.insert(format!("{}.error", name), Value::Str(error));
                    return Ok(());
                }
                // Command execution: exec(cmd, ...args)
                if fn_name == "exec" {
                    check_exec_permission()?;
                    if args.is_empty() {
                        return Err(CliError::Compile("exec() requires at least 1 argument (command)".into()));
                    }
                    let mut cmd_args = Vec::new();
                    for arg in args {
                        let val = eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                        cmd_args.push(val.to_string());
                    }
                    let command = &cmd_args[0];
                    let command_args = &cmd_args[1..];

                    use std::process::Command;
                    let output = Command::new(command)
                        .args(command_args)
                        .output()
                        .map_err(|e| CliError::Io(format!("exec(\"{}\"): {}", command, e)))?;

                    let status_code = output.status.code().unwrap_or(-1) as f32;
                    let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();
                    let ok = if output.status.success() { 1.0 } else { 0.0 };

                    scalars.insert(format!("{}.status", name), Value::Float(status_code));
                    scalars.insert(format!("{}.output", name), Value::Str(stdout_str));
                    scalars.insert(format!("{}.ok", name), Value::Float(ok));
                    scalars.insert(format!("{}.error", name), Value::Str(stderr_str));
                    return Ok(());
                }
                // video_open(byte_array) → handle + .width .height .frames .fps scalars
                if fn_name == "video_open" {
                    if args.len() != 1 {
                        return Err(CliError::Compile("video_open() requires 1 argument (byte array name)".into()));
                    }
                    let arr_name = match &args[0] {
                        ScalarExpr::Ref(n) => n.clone(),
                        _ => return Err(CliError::Compile("video_open() argument must be an array name".into())),
                    };
                    let bytes = video_open_extract_bytes(&arr_name, arrays)?;
                    let (handle, w, h, fc, fps) = video_open_from_bytes(bytes)?;
                    video_open_register(handle, name, w, h, fc, fps, scalars);
                    return Ok(());
                }
                if fn_name == "video_open_file" {
                    if args.len() != 1 {
                        return Err(CliError::Compile("video_open_file() requires 1 argument (file path)".into()));
                    }
                    let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let path = path_val.as_str().map_err(|_| CliError::Compile("video_open_file() path must be a string".into()))?.to_string();
                    check_read_permission_for(&path)?;
                    let (handle, w, h, fc, fps) = video_open_file_bytes(&path)?;
                    video_open_register(handle, name, w, h, fc, fps, scalars);
                    return Ok(());
                }
                // video_frame(handle, index) → .r .g .b arrays
                if fn_name == "video_frame" {
                    if args.len() != 2 {
                        return Err(CliError::Compile("video_frame() requires 2 arguments (handle, frame_index)".into()));
                    }
                    let handle_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let idx_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let handle_id = handle_val.as_float().map_err(|_| CliError::Compile("video_frame(): handle must be numeric".into()))? as u32;
                    let frame_idx = idx_val.as_float().map_err(|_| CliError::Compile("video_frame(): index must be numeric".into()))? as usize;
                    let (r, g, b) = video_frame_decode(handle_id, frame_idx)?;
                    gpu_array_insert(format!("{}.r", name), r);
                    gpu_array_insert(format!("{}.g", name), g);
                    gpu_array_insert(format!("{}.b", name), b);
                    return Ok(());
                }
                // Extern FFI function call: let r = SomeCFn(args...)
                if is_extern_fn(fn_name) {
                    let mut arg_vals = Vec::new();
                    for arg in args {
                        arg_vals.push(eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?);
                    }
                    let result = call_extern_fn(fn_name, &arg_vals)?;
                    scalars.insert(name.clone(), result);
                    if *mutable { mutable_scalars.insert(name.clone()); }
                    return Ok(());
                }
                // Array-returning functions: read_lines, list_dir, split
                if let Some(result) = eval_array_fn(fn_name, args, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)? {
                    match result {
                        ArrayResult::Values(arr) => { arrays.insert(name.clone(), arr); }
                        ArrayResult::GpuFloats(arr) => { gpu_array_insert(name.clone(), arr); }
                            ArrayResult::Resident(buf) => { gpu_array_insert_resident(name.clone(), buf); }
                    }
                    if *mutable { mutable_scalars.insert(name.clone()); }
                    return Ok(());
                }
                // Hashmap-returning functions: json_parse
                if let Some(hm) = eval_hashmap_fn(fn_name, args, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)? {
                    hashmaps.insert(name.clone(), hm);
                    if *mutable { mutable_scalars.insert(name.clone()); }
                    return Ok(());
                }
            }
            let result = eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            // Check if a user function returned an array/map via the side-channel
            let returned_arr = RETURNED_ARRAY.with(|r| r.borrow_mut().take());
            if let Some(arr) = returned_arr {
                arrays.insert(name.clone(), arr);
            } else {
                let returned_hm = RETURNED_MAP.with(|r| r.borrow_mut().take());
                if let Some(hm) = returned_hm {
                    hashmaps.insert(name.clone(), hm);
                } else {
                    scalars.insert(name.clone(), result);
                }
            }
            if *mutable { mutable_scalars.insert(name.clone()); }
        }
        Statement::Assign { name, value } => {
            if !mutable_scalars.contains(name) {
                return Err(CliError::Compile(format!(
                    "Cannot assign to '{}': not declared as mutable (use 'let mut')", name
                )));
            }
            let result = eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            scalars.insert(name.clone(), result);
        }
        Statement::ArrayAssign { array, index, value } => {
            if !mutable_scalars.contains(array.as_str()) {
                return Err(CliError::Compile(format!(
                    "Cannot assign to '{}': not declared as mutable (use 'let mut')", array
                )));
            }
            gpu_array_materialize(array, arrays);
            let idx = eval_scalar(index, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
            let val = eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let arr = arrays.get_mut(array)
                .ok_or_else(|| CliError::Compile(format!("undefined array '{}'", array)))?;
            if idx >= arr.len() {
                return Err(CliError::Compile(format!(
                    "Array index out of bounds: {}[{}] (length {})", array, idx, arr.len()
                )));
            }
            arr[idx] = val;
        }
        Statement::ArrayPush { array, value } => {
            if !mutable_scalars.contains(array.as_str()) {
                mutable_scalars.insert(array.clone());  // Auto-promote to mutable on push
            }
            gpu_array_materialize(array, arrays);
            let val = eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let arr = arrays.get_mut(array)
                .ok_or_else(|| CliError::Compile(format!("undefined array '{}'", array)))?;
            arr.push(val);
        }
        Statement::Print { segments } => {
            let mut output = String::new();
            for seg in segments {
                match seg {
                    PrintSegment::Literal(s) => output.push_str(s),
                    PrintSegment::Scalar { name, precision } => {
                        if let Some(value) = scalars.get(name) {
                            match (precision, value) {
                                (Some(p), Value::Float(f)) => output.push_str(&format!("{:.prec$}", f, prec = *p)),
                                (Some(_p), Value::Int(i)) => output.push_str(&format!("{}", i)),
                                (Some(_), Value::Str(s)) => output.push_str(s),
                                (Some(_), Value::Map(_)) => output.push_str(&format!("{}", value)),
                                (Some(_), Value::None) => output.push_str("none"),
                                (None, _) => output.push_str(&format!("{}", value)),
                            }
                        } else if let Some(s) = gpu_array_format(name.as_str()) {
                            output.push_str(&s);
                        } else if let Some(arr) = arrays.get(name.as_str()) {
                            let items: Vec<String> = arr.iter().map(|v| format!("{}", v)).collect();
                            output.push_str(&format!("[{}]", items.join(", ")));
                        } else if let Some(hm) = hashmaps.get(name.as_str()) {
                            let mut keys: Vec<&String> = hm.keys().collect();
                            keys.sort();
                            let items: Vec<String> = keys.iter().map(|k| format!("{}={}", k, hm[*k])).collect();
                            output.push_str(&format!("{{{}}}", items.join(", ")));
                        } else {
                            return Err(CliError::UndefinedScalar(name.clone()));
                        }
                    }
                    PrintSegment::Expr(e) => {
                        let val = eval_scalar(e, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                        output.push_str(&format!("{}", val));
                    }
                }
            }
            runtime_println(&output);
        }
        Statement::ArrayDecl { name, elements, mutable } => {
            let mut values = Vec::new();
            for elem in elements {
                let val = eval_scalar(elem, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                values.push(val);
            }
            arrays.insert(name.clone(), values);
            if *mutable { mutable_scalars.insert(name.clone()); }
        }
        Statement::IfBlock { condition, body, elif_branches, else_body } => {
            let cond = eval_scalar(condition, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            if cond != 0.0 {
                for (s, _sp) in body {
                    execute_block_stmt(s, streams, scalars, mutable_scalars, struct_defs, gpu, arrays, hashmaps, scalar_fns, rng)?;
                }
            } else {
                let mut matched = false;
                for (elif_cond, elif_body) in elif_branches {
                    let ec = eval_scalar(elif_cond, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
                    if ec != 0.0 {
                        for (s, _sp) in elif_body {
                            execute_block_stmt(s, streams, scalars, mutable_scalars, struct_defs, gpu, arrays, hashmaps, scalar_fns, rng)?;
                        }
                        matched = true;
                        break;
                    }
                }
                if !matched && !else_body.is_empty() {
                    for (s, _sp) in else_body {
                        execute_block_stmt(s, streams, scalars, mutable_scalars, struct_defs, gpu, arrays, hashmaps, scalar_fns, rng)?;
                    }
                }
            }
        }
        Statement::MapDecl { name, mutable } => {
            hashmaps.insert(name.clone(), HashMap::new());
            if *mutable { mutable_scalars.insert(name.clone()); }
        }
        Statement::MapInsert { map, key, value } => {
            if !mutable_scalars.contains(map.as_str()) {
                return Err(CliError::Compile(format!(
                    "Cannot insert into '{}': not declared as mutable (use 'let mut')", map
                )));
            }
            let key_val = eval_scalar(key, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_str()?.to_string();
            let val = eval_scalar(value, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let hm = hashmaps.get_mut(map)
                .ok_or_else(|| CliError::Compile(format!("undefined map '{}'", map)))?;
            hm.insert(key_val, val);
        }
        Statement::WriteFile { path, content } => {
            let path_val = eval_scalar(path, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let p = path_val.as_str().map_err(|_| CliError::Compile("write_file() path must be a string".into()))?;
            check_write_permission_for(p)?;
            let content_val = eval_scalar(content, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let c = content_val.to_string();
            std::fs::write(p, &c).map_err(|e| CliError::Io(format!("write_file(\"{}\"): {}", p, e)))?;
        }
        Statement::AppendFile { path, content } => {
            let path_val = eval_scalar(path, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let p = path_val.as_str().map_err(|_| CliError::Compile("append_file() path must be a string".into()))?;
            check_write_permission_for(p)?;
            let content_val = eval_scalar(content, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let c = content_val.to_string();
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().create(true).append(true).open(p)
                .map_err(|e| CliError::Io(format!("append_file(\"{}\"): {}", p, e)))?;
            file.write_all(c.as_bytes()).map_err(|e| CliError::Io(format!("append_file(\"{}\"): {}", p, e)))?;
        }
        Statement::SaveData { path, map_name } => {
            let path_val = eval_scalar(path, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let p = path_val.as_str().map_err(|_| CliError::Compile("save_data() path must be a string".into()))?.to_string();
            check_write_permission_for(&p)?;
            let hm = hashmaps.get(map_name)
                .ok_or_else(|| CliError::Compile(format!("save_data(): undefined map '{}'", map_name)))?;
            let content = serialize_od(hm);
            std::fs::write(&p, &content).map_err(|e| CliError::Io(format!("save_data(\"{}\"): {}", p, e)))?;
        }
        Statement::WriteCsv { path, array_name } => {
            let path_val = eval_scalar(path, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let p = path_val.as_str().map_err(|_| CliError::Compile("write_csv() path must be a string".into()))?.to_string();
            check_write_permission_for(&p)?;
            gpu_array_materialize(array_name.as_str(), arrays);
            let arr = arrays.get(array_name.as_str())
                .ok_or_else(|| CliError::Compile(format!("write_csv(): undefined array '{}'", array_name)))?;
            csv_write_structured(&p, arr)?;
        }
        Statement::WriteBytes { path, array_name } => {
            let path_val = eval_scalar(path, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let p = path_val.as_str().map_err(|_| CliError::Compile("write_bytes() path must be a string".into()))?.to_string();
            check_write_permission_for(&p)?;
            gpu_array_materialize(array_name.as_str(), arrays);
            let arr = arrays.get(array_name.as_str())
                .ok_or_else(|| CliError::Compile(format!("write_bytes(): undefined array '{}'", array_name)))?;
            do_write_bytes(&p, arr)?;
        }
        Statement::ExternBlock { library, functions, .. } => {
            register_extern_block(library, functions);
        }
        Statement::ExprStmt { expr } => {
            let _ = eval_scalar(expr, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            // R-05: Apply scalar writeback from user function calls
            SCALAR_WRITEBACK.with(|sw| {
                if let Some(writes) = sw.borrow_mut().take() {
                    for (name, value) in writes {
                        scalars.insert(name, value);
                    }
                }
            });
        }
        _ => {
            return Err(CliError::Compile(
                "only let, assignment, print, array, map, push, file I/O, and if blocks are allowed inside if blocks".into()
            ));
        }
    }
    Ok(())
}

/// Import definitions from another .flow file.
///
/// Imports pipeline fns, scalar fns, structs, constants, and arrays.
/// Module `let` declarations are imported as immutable regardless of `let mut` in source.
/// Nested `use` declarations are NOT imported (no transitive imports).
fn import_module(
    base_dir: &str,
    module: &str,
    functions: &mut HashMap<String, FnDef>,
    scalar_fns: &mut HashMap<String, ScalarFnDef>,
    struct_defs: &mut HashMap<String, Vec<String>>,
    scalars: &mut HashMap<String, Value>,
    arrays: &mut HashMap<String, Vec<Value>>,
    hashmaps: &mut HashMap<String, HashMap<String, Value>>,
    mutable_scalars: &mut std::collections::HashSet<String>,
) -> Result<(), CliError> {
    // Support path-style module names (e.g., use "../stdlib/compiler/ir")
    // Path modules use direct join (allowing ".."), bare names use resolve_path.
    // Falls back to stdlib directory (exe_dir/../stdlib/) when not found locally.
    let module_path = if module.contains('/') || module.contains('\\') {
        let p = std::path::Path::new(base_dir).join(format!("{}.flow", module));
        match p.canonicalize() {
            Ok(cp) => cp.to_string_lossy().into_owned(),
            Err(_) => {
                // Fallback: try stdlib directory relative to executable
                resolve_stdlib_module(module)?
            }
        }
    } else {
        let local = resolve_path(base_dir, &format!("{}.flow", module))?;
        if std::path::Path::new(&local).exists() {
            local
        } else {
            // Fallback: try stdlib directory relative to executable
            resolve_stdlib_module(module)?
        }
    };
    // Circular import guard (transitive imports)
    let canonical = std::path::Path::new(&module_path)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(&module_path));
    let canon_str = canonical.to_string_lossy().into_owned();
    let already = IMPORTED_PATHS.with(|s| s.borrow().contains(&canon_str));
    if already { return Ok(()); }
    IMPORTED_PATHS.with(|s| { s.borrow_mut().insert(canon_str); });
    let source = std::fs::read_to_string(&module_path)
        .map_err(|e| CliError::Compile(format!("cannot read module '{}': {}", module_path, e)))?;
    let program = octoflow_parser::parse(&source)
        .map_err(|e| CliError::Compile(format!("error in module '{}': {}", module, e)))?;

    // Temporary context for evaluating module-level constants
    let empty_streams: HashMap<String, Vec<f32>> = HashMap::new();
    let mut mod_scalars: HashMap<String, Value> = HashMap::new();
    let mut mod_arrays: HashMap<String, Vec<Value>> = HashMap::new();
    let mut mod_hashmaps: HashMap<String, HashMap<String, Value>> = HashMap::new();
    let mut mod_struct_defs: HashMap<String, Vec<String>> = HashMap::new();
    let mod_mutable: std::collections::HashSet<String> = std::collections::HashSet::new();
    let gpu: Option<octoflow_vulkan::VulkanCompute> = None;
    let rng = Cell::new(42u64);

    for (stmt, _span) in &program.statements {
        match stmt {
            Statement::FnDecl { name, params, body } => {
                let def = FnDef { params: params.clone(), body: body.clone() };
                functions.insert(format!("{}.{}", module, name), def.clone());
                functions.insert(name.clone(), def);
            }
            Statement::ScalarFnDecl { name, params, body } => {
                let def = ScalarFnDef {
                    params: params.clone(), body: body.clone(),
                    captured_scalars: HashMap::new(), captured_arrays: HashMap::new(),
                };
                scalar_fns.insert(format!("{}.{}", module, name), def.clone());
                scalar_fns.insert(name.clone(), def);
            }
            Statement::StructDef { name, fields } => {
                mod_struct_defs.insert(name.clone(), fields.clone());
                struct_defs.insert(format!("{}.{}", module, name), fields.clone());
                struct_defs.insert(name.clone(), fields.clone());
            }
            Statement::LetDecl { name, value, mutable } => {
                // try() error handling in module context
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    if fn_name == "try" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("try() requires exactly 1 argument".into()));
                        }
                        match eval_scalar(&args[0], &empty_streams, &mod_scalars, &gpu, &mut mod_arrays, &mut mod_hashmaps, scalar_fns, &mod_struct_defs, &rng, &mod_mutable) {
                            Ok(val) => {
                                let fields = [("value", val), ("ok", Value::Float(1.0)), ("error", Value::Str(String::new()))];
                                for (field, fval) in &fields {
                                    mod_scalars.insert(format!("{}.{}", name, field), fval.clone());
                                    scalars.insert(format!("{}.{}.{}", module, name, field), fval.clone());
                                    scalars.insert(format!("{}.{}", name, field), fval.clone());
                                }
                            }
                            Err(e) => {
                                let fields: [(&str, Value); 3] = [("value", Value::Str(String::new())), ("ok", Value::Float(0.0)), ("error", Value::Str(format!("{}", e)))];
                                for (field, fval) in &fields {
                                    mod_scalars.insert(format!("{}.{}", name, field), fval.clone());
                                    scalars.insert(format!("{}.{}.{}", module, name, field), fval.clone());
                                    scalars.insert(format!("{}.{}", name, field), fval.clone());
                                }
                            }
                        }
                        continue;
                    }
                    // HTTP client functions in module context
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        check_net_permission()?;
                        let (method, expected_args) = match fn_name.as_str() {
                            "http_get" => ("GET", 1), "http_post" => ("POST", 2),
                            "http_put" => ("PUT", 2), _ => ("DELETE", 1),
                        };
                        if args.len() != expected_args {
                            return Err(CliError::Compile(format!("{}() requires {} argument(s), got {}", fn_name, expected_args, args.len())));
                        }
                        let url_val = eval_scalar(&args[0], &empty_streams, &mod_scalars, &gpu, &mut mod_arrays, &mut mod_hashmaps, scalar_fns, &mod_struct_defs, &rng, &mod_mutable)?;
                        let url = url_val.as_str().map_err(|_| CliError::Compile(format!("{}() URL must be a string", fn_name)))?.to_string();
                        let body_str = if expected_args == 2 {
                            let b = eval_scalar(&args[1], &empty_streams, &mod_scalars, &gpu, &mut mod_arrays, &mut mod_hashmaps, scalar_fns, &mod_struct_defs, &rng, &mod_mutable)?;
                            Some(b.as_str().map_err(|_| CliError::Compile(format!("{}() body must be a string", fn_name)))?.to_string())
                        } else { None };
                        let (status, body, ok, error) = do_http_request(method, &url, body_str.as_deref());
                        let http_fields: [(&str, Value); 4] = [("status", Value::Float(status)), ("body", Value::Str(body)), ("ok", Value::Float(ok)), ("error", Value::Str(error))];
                        for (field, fval) in &http_fields {
                            mod_scalars.insert(format!("{}.{}", name, field), fval.clone());
                            scalars.insert(format!("{}.{}.{}", module, name, field), fval.clone());
                            scalars.insert(format!("{}.{}", name, field), fval.clone());
                        }
                        continue;
                    }
                    // Command execution: exec(cmd, ...args) in module context
                    if fn_name == "exec" {
                        check_exec_permission()?;
                        if args.is_empty() {
                            return Err(CliError::Compile("exec() requires at least 1 argument (command)".into()));
                        }
                        let mut cmd_args = Vec::new();
                        for arg in args {
                            let val = eval_scalar(arg, &empty_streams, &mod_scalars, &gpu, &mut mod_arrays, &mut mod_hashmaps, scalar_fns, &mod_struct_defs, &rng, &mod_mutable)?;
                            cmd_args.push(val.to_string());
                        }
                        let command = &cmd_args[0];
                        let command_args = &cmd_args[1..];

                        use std::process::Command;
                        let output = Command::new(command)
                            .args(command_args)
                            .output()
                            .map_err(|e| CliError::Io(format!("exec(\"{}\"): {}", command, e)))?;

                        let status_code = output.status.code().unwrap_or(-1) as f32;
                        let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
                        let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();
                        let ok = if output.status.success() { 1.0 } else { 0.0 };

                        let exec_fields: [(&str, Value); 4] = [("status", Value::Float(status_code)), ("output", Value::Str(stdout_str)), ("ok", Value::Float(ok)), ("error", Value::Str(stderr_str))];
                        for (field, fval) in &exec_fields {
                            mod_scalars.insert(format!("{}.{}", name, field), fval.clone());
                            scalars.insert(format!("{}.{}.{}", module, name, field), fval.clone());
                            scalars.insert(format!("{}.{}", name, field), fval.clone());
                        }
                        continue;
                    }
                    // video_open(byte_array) → handle + .width .height .frames .fps scalars
                    if fn_name == "video_open" {
                        if args.len() != 1 {
                            return Err(CliError::Compile("video_open() requires 1 argument (byte array name)".into()));
                        }
                        let arr_name = match &args[0] {
                            ScalarExpr::Ref(n) => n.clone(),
                            _ => return Err(CliError::Compile("video_open() argument must be an array name".into())),
                        };
                        let bytes = video_open_extract_bytes(&arr_name, &mod_arrays)?;
                        let (handle, w, h, fc, fps) = video_open_from_bytes(bytes)?;
                        let id = video_open_register(handle, name, w, h, fc, fps, &mut mod_scalars);
                        // Module-prefixed scalars
                        for (field, fval) in &[("width", Value::Float(w as f32)), ("height", Value::Float(h as f32)), ("frames", Value::Float(fc as f32)), ("fps", Value::Float(fps))] {
                            scalars.insert(format!("{}.{}.{}", module, name, field), fval.clone());
                            scalars.insert(format!("{}.{}", name, field), fval.clone());
                        }
                        let id_val = Value::Float(id as f32);
                        scalars.insert(format!("{}.{}", module, name), id_val.clone());
                        scalars.insert(name.clone(), id_val);
                        continue;
                    }
                    // video_frame(handle, index) → .r .g .b arrays
                    if fn_name == "video_frame" {
                        if args.len() != 2 {
                            return Err(CliError::Compile("video_frame() requires 2 arguments (handle, frame_index)".into()));
                        }
                        let handle_val = eval_scalar(&args[0], &empty_streams, &mod_scalars, &gpu, &mut mod_arrays, &mut mod_hashmaps, scalar_fns, &mod_struct_defs, &rng, &mod_mutable)?;
                        let idx_val = eval_scalar(&args[1], &empty_streams, &mod_scalars, &gpu, &mut mod_arrays, &mut mod_hashmaps, scalar_fns, &mod_struct_defs, &rng, &mod_mutable)?;
                        let handle_id = handle_val.as_float().map_err(|_| CliError::Compile("video_frame(): handle must be numeric".into()))? as u32;
                        let frame_idx = idx_val.as_float().map_err(|_| CliError::Compile("video_frame(): index must be numeric".into()))? as usize;
                        let (r, g, b) = video_frame_decode(handle_id, frame_idx)?;
                        for (ch, data) in &[("r", &r), ("g", &g), ("b", &b)] {
                            gpu_array_insert(format!("{}.{}", name, ch), data.to_vec());
                            gpu_array_insert(format!("{}.{}.{}", module, name, ch), data.to_vec());
                        }
                        continue;
                    }
                    // Hashmap-returning functions in module context: json_parse
                    if let Some(hm) = eval_hashmap_fn(fn_name, args, &empty_streams, &mod_scalars, &gpu, &mut mod_arrays, &mut mod_hashmaps, scalar_fns, &mod_struct_defs, &rng, &mod_mutable)? {
                        mod_hashmaps.insert(name.clone(), hm.clone());
                        hashmaps.insert(format!("{}.{}", module, name), hm.clone());
                        hashmaps.insert(name.clone(), hm);
                        continue;
                    }
                }
                // Evaluate in module-local context (only prior module constants visible)
                let result = eval_scalar(value, &empty_streams, &mod_scalars, &gpu, &mut mod_arrays, &mut mod_hashmaps, scalar_fns, &mod_struct_defs, &rng, &mod_mutable)?;
                mod_scalars.insert(name.clone(), result.clone());
                scalars.insert(format!("{}.{}", module, name), result.clone());
                scalars.insert(name.clone(), result);
                if *mutable {
                    mutable_scalars.insert(format!("{}.{}", module, name));
                    mutable_scalars.insert(name.clone());
                }
            }
            Statement::ArrayDecl { name, elements, mutable } => {
                let mut values = Vec::new();
                for elem in elements {
                    let val = eval_scalar(elem, &empty_streams, &mod_scalars, &gpu, &mut mod_arrays, &mut mod_hashmaps, scalar_fns, &mod_struct_defs, &rng, &mod_mutable)?;
                    values.push(val);
                }
                mod_arrays.insert(name.clone(), values.clone());
                arrays.insert(format!("{}.{}", module, name), values.clone());
                arrays.insert(name.clone(), values);
                if *mutable {
                    mutable_scalars.insert(format!("{}.{}", module, name));
                    mutable_scalars.insert(name.clone());
                }
            }
            Statement::ExternBlock { library, functions: ext_fns, .. } => {
                register_extern_block(library, ext_fns);
            }
            Statement::UseDecl { module: nested_module } => {
                // Transitive import: resolve relative to this module's directory
                let mod_base = std::path::Path::new(&module_path)
                    .parent()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|| base_dir.to_string());
                let _ = import_module(&mod_base, nested_module, functions, scalar_fns,
                                      struct_defs, scalars, arrays, hashmaps, mutable_scalars);
            }
            // Skip executable statements (ForLoop, WhileLoop, Print, etc.)
            _ => {}
        }
    }

    // R-04: Capture module-level constants into all functions defined in this module.
    // This enables nested import scope: if module A imports B, A's functions carry
    // B's constants so they resolve correctly when called from a third module.
    if !mod_scalars.is_empty() || !mod_arrays.is_empty() {
        for (stmt, _) in &program.statements {
            if let Statement::ScalarFnDecl { name, .. } = stmt {
                // Update both namespaced and bare versions
                for key in &[format!("{}.{}", module, name), name.clone()] {
                    if let Some(fn_def) = scalar_fns.get_mut(key) {
                        fn_def.captured_scalars = mod_scalars.clone();
                        fn_def.captured_arrays = mod_arrays.clone();
                    }
                }
            }
        }
    }

    Ok(())
}

/// Inline a function call: substitute parameters with arguments.
fn inline_fn_call(fn_def: &FnDef, call_args: &[Arg]) -> Vec<StageCall> {
    fn_def.body.iter().map(|stage| {
        StageCall {
            operation: stage.operation.clone(),
            args: stage.args.iter().map(|arg| {
                if let Arg::Ref(name) = arg {
                    if let Some(idx) = fn_def.params.iter().position(|p| p == name) {
                        return call_args[idx].clone();
                    }
                }
                arg.clone()
            }).collect(),
        }
    }).collect()
}

/// Apply `-i` and `-o` path overrides by rewriting the AST.
fn apply_path_overrides(program: &Program, overrides: &crate::Overrides) -> Program {
    let statements = program.statements.iter().map(|(stmt, span)| {
        let new_stmt = match stmt {
            Statement::StreamDecl { name, expr } if overrides.input_path.is_some() => {
                Statement::StreamDecl {
                    name: name.clone(),
                    expr: rewrite_taps(expr, overrides.input_path.as_ref().unwrap()),
                }
            }
            Statement::Emit { expr, path: _ } if overrides.output_path.is_some() => {
                Statement::Emit {
                    expr: expr.clone(),
                    path: overrides.output_path.as_ref().unwrap().clone(),
                }
            }
            Statement::WhileLoop { condition, body } => {
                let new_body = body.iter().map(|(s, sp)| {
                    let ns = match s {
                        Statement::StreamDecl { name: n, expr: e } if overrides.input_path.is_some() => {
                            Statement::StreamDecl { name: n.clone(), expr: rewrite_taps(e, overrides.input_path.as_ref().unwrap()) }
                        }
                        Statement::Emit { expr: e, path: _ } if overrides.output_path.is_some() => {
                            Statement::Emit { expr: e.clone(), path: overrides.output_path.as_ref().unwrap().clone() }
                        }
                        o => o.clone(),
                    };
                    (ns, *sp)
                }).collect();
                Statement::WhileLoop { condition: condition.clone(), body: new_body }
            }
            Statement::ForLoop { var, start, end, body } => {
                let new_body = body.iter().map(|(s, sp)| {
                    let ns = match s {
                        Statement::StreamDecl { name: n, expr: e } if overrides.input_path.is_some() => {
                            Statement::StreamDecl { name: n.clone(), expr: rewrite_taps(e, overrides.input_path.as_ref().unwrap()) }
                        }
                        Statement::Emit { expr: e, path: _ } if overrides.output_path.is_some() => {
                            Statement::Emit { expr: e.clone(), path: overrides.output_path.as_ref().unwrap().clone() }
                        }
                        o => o.clone(),
                    };
                    (ns, *sp)
                }).collect();
                Statement::ForLoop { var: var.clone(), start: start.clone(), end: end.clone(), body: new_body }
            }
            Statement::ForEachLoop { var, iterable, body } => {
                let new_body = body.iter().map(|(s, sp)| {
                    let ns = match s {
                        Statement::StreamDecl { name: n, expr: e } if overrides.input_path.is_some() => {
                            Statement::StreamDecl { name: n.clone(), expr: rewrite_taps(e, overrides.input_path.as_ref().unwrap()) }
                        }
                        Statement::Emit { expr: e, path: _ } if overrides.output_path.is_some() => {
                            Statement::Emit { expr: e.clone(), path: overrides.output_path.as_ref().unwrap().clone() }
                        }
                        o => o.clone(),
                    };
                    (ns, *sp)
                }).collect();
                Statement::ForEachLoop { var: var.clone(), iterable: iterable.clone(), body: new_body }
            }
            other => other.clone(),
        };
        (new_stmt, *span)
    }).collect();
    Program { statements }
}

/// Recursively replace all Tap paths in an expression tree.
fn rewrite_taps(expr: &Expr, new_path: &str) -> Expr {
    match expr {
        Expr::Tap { .. } => Expr::Tap { path: new_path.to_string() },
        Expr::Pipe { input, stages } => Expr::Pipe {
            input: Box::new(rewrite_taps(input, new_path)),
            stages: stages.clone(),
        },
        other => other.clone(),
    }
}

/// Generate a dispatch plan for a parsed OctoFlow program (no execution).
pub fn plan(program: &Program) -> Vec<String> {
    let mut steps = Vec::new();
    let mut step_num = 1;
    let mut known_scalars: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut known_structs: HashMap<String, Vec<String>> = HashMap::new();

    for (stmt, _span) in &program.statements {
        match stmt {
            Statement::StreamDecl { name, expr } => {
                let desc = plan_expr(expr, name, &known_scalars);
                steps.push(format!("  {}. {}", step_num, desc));
                step_num += 1;
            }
            Statement::LetDecl { name, value, mutable } => {
                // Check for vec or struct constructor in plan mode
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    if matches!(fn_name.as_str(), "vec2" | "vec3" | "vec4") {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        let mut_str = if *mutable { "mut " } else { "" };
                        steps.push(format!("  {}. [VEC]     let {}{} = {}({})",
                            step_num, mut_str, name, fn_name, args_str.join(", ")));
                        let components = ["x", "y", "z", "w"];
                        for (i, _) in args.iter().enumerate() {
                            known_scalars.insert(format!("{}.{}", name, components[i]));
                        }
                        step_num += 1;
                        continue;
                    }
                    if let Some(fields) = known_structs.get(fn_name) {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        let mut_str = if *mutable { "mut " } else { "" };
                        steps.push(format!("  {}. [STRUCT]  let {}{} = {}({})",
                            step_num, mut_str, name, fn_name, args_str.join(", ")));
                        for field in fields {
                            known_scalars.insert(format!("{}.{}", name, field));
                        }
                        step_num += 1;
                        continue;
                    }
                    if fn_name == "try" {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}. [TRY]     let {} = try({})   → .value .ok .error",
                            step_num, name, args_str.join(", ")));
                        for field in &["value", "ok", "error"] {
                            known_scalars.insert(format!("{}.{}", name, field));
                        }
                        step_num += 1;
                        continue;
                    }
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}. [HTTP]    let {} = {}({})   → .status .body .ok .error",
                            step_num, name, fn_name, args_str.join(", ")));
                        for field in &["status", "body", "ok", "error"] {
                            known_scalars.insert(format!("{}.{}", name, field));
                        }
                        step_num += 1;
                        continue;
                    }
                    if fn_name == "exec" {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}. [EXEC]    let {} = exec({})   → .status .output .ok .error",
                            step_num, name, args_str.join(", ")));
                        for field in &["status", "output", "ok", "error"] {
                            known_scalars.insert(format!("{}.{}", name, field));
                        }
                        step_num += 1;
                        continue;
                    }
                    if fn_name == "video_open" {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}. [VIDEO]   let {} = video_open({})   -> .width .height .frames .fps",
                            step_num, name, args_str.join(", ")));
                        for field in &["width", "height", "frames", "fps"] {
                            known_scalars.insert(format!("{}.{}", name, field));
                        }
                        known_scalars.insert(name.clone());
                        step_num += 1;
                        continue;
                    }
                    if fn_name == "video_frame" {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}. [VIDEO]   let {} = video_frame({})   -> .r .g .b",
                            step_num, name, args_str.join(", ")));
                        step_num += 1;
                        continue;
                    }
                    if fn_name == "json_parse" || fn_name == "json_parse_array" {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}. [JSON]    let {} = {}({})",
                            step_num, name, fn_name, args_str.join(", ")));
                        known_scalars.insert(name.clone());
                        step_num += 1;
                        continue;
                    }
                }
                let mut_str = if *mutable { "mut " } else { "" };
                let desc = plan_scalar(value, &format!("{}{}", mut_str, name));
                steps.push(format!("  {}. {}", step_num, desc));
                known_scalars.insert(name.clone());
                step_num += 1;
            }
            Statement::Assign { name, value } => {
                let desc = format!("[CPU]     {} = {}", name, scalar_to_string(value));
                steps.push(format!("  {}. {}", step_num, desc));
                step_num += 1;
            }
            Statement::ArrayAssign { array, index, value } => {
                steps.push(format!("  {}. [CPU]     {}[{}] = {}", step_num, array, scalar_to_string(index), scalar_to_string(value)));
                step_num += 1;
            }
            Statement::ArrayPush { array, value } => {
                steps.push(format!("  {}. [CPU]     push({}, {})", step_num, array, scalar_to_string(value)));
                step_num += 1;
            }
            Statement::Emit { expr, path } => {
                let stream_name = match expr {
                    Expr::Ref { name } => name.as_str(),
                    _ => "?",
                };
                steps.push(format!("  {}. [EMIT]    emit({}, \"{}\")", step_num, stream_name, path));
                step_num += 1;
            }
            Statement::Print { segments } => {
                let display: String = segments.iter().map(|seg| match seg {
                    PrintSegment::Literal(s) => s.clone(),
                    PrintSegment::Scalar { name, precision: None } => format!("{{{}}}", name),
                    PrintSegment::Scalar { name, precision: Some(p) } => format!("{{{}:.{}}}", name, p),
                    PrintSegment::Expr(_) => "<expr>".to_string(),
                }).collect();
                steps.push(format!("  {}. [PRINT]   print(\"{}\")", step_num, display));
                step_num += 1;
            }
            Statement::FnDecl { name, params, body } => {
                let params_str = params.join(", ");
                let body_str: Vec<String> = body.iter().map(|s| {
                    if s.args.is_empty() { format!("{}()", s.operation) }
                    else {
                        let args: Vec<String> = s.args.iter().map(|a| match a {
                            Arg::Literal(n) => format!("{}", n),
                            Arg::IntLiteral(n) => format!("{}", n),
                            Arg::Ref(r) => r.clone(),
                        }).collect();
                        format!("{}({})", s.operation, args.join(", "))
                    }
                }).collect();
                steps.push(format!("  {}. [FN]      fn {}({}): {}", step_num, name, params_str, body_str.join(" |> ")));
                step_num += 1;
            }
            Statement::UseDecl { module } => {
                // Try to summarize what the module provides
                let module_path = format!("{}.flow", module);
                let summary = if let Ok(path) = resolve_path(".", &module_path) {
                    if let Ok(source) = std::fs::read_to_string(&path) {
                        if let Ok(prog) = octoflow_parser::parse(&source) {
                            let mut pipeline_fns = Vec::new();
                            let mut scalar_fn_count = 0;
                            let mut struct_count = 0;
                            let mut const_count = 0;
                            let mut array_count = 0;
                            for (s, _) in &prog.statements {
                                match s {
                                    Statement::FnDecl { name, .. } => pipeline_fns.push(name.clone()),
                                    Statement::ScalarFnDecl { .. } => scalar_fn_count += 1,
                                    Statement::StructDef { .. } => struct_count += 1,
                                    Statement::LetDecl { .. } => const_count += 1,
                                    Statement::ArrayDecl { .. } => array_count += 1,
                                    _ => {}
                                }
                            }
                            let mut parts = Vec::new();
                            if !pipeline_fns.is_empty() {
                                parts.push(pipeline_fns.join(", "));
                            }
                            if scalar_fn_count > 0 { parts.push(format!("{} scalar fn{}", scalar_fn_count, if scalar_fn_count == 1 { "" } else { "s" })); }
                            if struct_count > 0 { parts.push(format!("{} struct{}", struct_count, if struct_count == 1 { "" } else { "s" })); }
                            if const_count > 0 { parts.push(format!("{} const{}", const_count, if const_count == 1 { "" } else { "s" })); }
                            if array_count > 0 { parts.push(format!("{} array{}", array_count, if array_count == 1 { "" } else { "s" })); }
                            if parts.is_empty() { String::new() } else { format!(" ({})", parts.join(" + ")) }
                        } else { String::new() }
                    } else { String::new() }
                } else { String::new() };
                steps.push(format!("  {}. [USE]     use {}{}", step_num, module, summary));
                step_num += 1;
            }
            Statement::StructDef { name, fields } => {
                known_structs.insert(name.clone(), fields.clone());
                steps.push(format!("  {}. [STRUCT]  struct {}({})", step_num, name, fields.join(", ")));
                step_num += 1;
            }
            Statement::ArrayDecl { name, elements, mutable } => {
                let mut_str = if *mutable { "mut " } else { "" };
                let elems_str: Vec<String> = elements.iter().map(scalar_to_string).collect();
                steps.push(format!("  {}. [ARRAY]   let {}{} = [{}]", step_num, mut_str, name, elems_str.join(", ")));
                step_num += 1;
            }
            Statement::WhileLoop { condition, body } => {
                steps.push(format!("  {}. [WHILE]   while {}", step_num, scalar_to_string(condition)));
                step_num += 1;
                plan_loop_body(body, &mut steps, &mut step_num, &mut known_scalars, 1);
                steps.push(format!("  {}. [END]", step_num));
                step_num += 1;
            }
            Statement::ForLoop { var, start, end, body } => {
                steps.push(format!("  {}. [FOR]     for {} in range({}, {})", step_num, var, scalar_to_string(start), scalar_to_string(end)));
                step_num += 1;
                plan_loop_body(body, &mut steps, &mut step_num, &mut known_scalars, 1);
                steps.push(format!("  {}. [END]", step_num));
                step_num += 1;
            }
            Statement::ForEachLoop { var, iterable, body } => {
                steps.push(format!("  {}. [FOREACH] for {} in {}", step_num, var, iterable));
                step_num += 1;
                plan_loop_body(body, &mut steps, &mut step_num, &mut known_scalars, 1);
                steps.push(format!("  {}. [END]", step_num));
                step_num += 1;
            }
            Statement::IfBlock { condition, body, elif_branches, else_body } => {
                steps.push(format!("  {}. [IF]      if {}", step_num, scalar_to_string(condition)));
                step_num += 1;
                plan_block_body(body, &mut steps, &mut step_num, &mut known_scalars, 1);
                for (elif_cond, elif_body) in elif_branches {
                    steps.push(format!("  {}. [ELIF]    elif {}", step_num, scalar_to_string(elif_cond)));
                    step_num += 1;
                    plan_block_body(elif_body, &mut steps, &mut step_num, &mut known_scalars, 1);
                }
                if !else_body.is_empty() {
                    steps.push(format!("  {}. [ELSE]", step_num));
                    step_num += 1;
                    plan_block_body(else_body, &mut steps, &mut step_num, &mut known_scalars, 1);
                }
                steps.push(format!("  {}. [END]", step_num));
                step_num += 1;
            }
            Statement::ScalarFnDecl { name, params, body } => {
                let params_str = params.join(", ");
                steps.push(format!("  {}. [FN]      fn {}({})", step_num, name, params_str));
                step_num += 1;
                plan_loop_body(body, &mut steps, &mut step_num, &mut known_scalars, 1);
                steps.push(format!("  {}. [END]", step_num));
                step_num += 1;
            }
            Statement::MapDecl { name, mutable } => {
                let mut_str = if *mutable { "mut " } else { "" };
                steps.push(format!("  {}. [MAP]     let {}{} = map()", step_num, mut_str, name));
                step_num += 1;
            }
            Statement::MapInsert { map, key, value } => {
                steps.push(format!("  {}. [CPU]     map_set({}, {}, {})", step_num, map, scalar_to_string(key), scalar_to_string(value)));
                step_num += 1;
            }
            Statement::WriteFile { path, content } => {
                steps.push(format!("  {}. [IO]      write_file({}, {})", step_num, scalar_to_string(path), scalar_to_string(content)));
                step_num += 1;
            }
            Statement::AppendFile { path, content } => {
                steps.push(format!("  {}. [IO]      append_file({}, {})", step_num, scalar_to_string(path), scalar_to_string(content)));
                step_num += 1;
            }
            Statement::SaveData { path, map_name } => {
                steps.push(format!("  {}. [IO]      save_data({}, {})", step_num, scalar_to_string(path), map_name));
                step_num += 1;
            }
            Statement::WriteCsv { path, array_name } => {
                steps.push(format!("  {}. [IO]      write_csv({}, {})", step_num, scalar_to_string(path), array_name));
                step_num += 1;
            }
            Statement::WriteBytes { path, array_name } => {
                steps.push(format!("  {}. [IO]      write_bytes({}, {})", step_num, scalar_to_string(path), array_name));
                step_num += 1;
            }
            Statement::Break | Statement::Continue | Statement::Return { .. } => {
                // break/continue/return at top level would be caught by preflight
            }
            Statement::ExternBlock { library, functions, .. } => {
                let fn_names: Vec<String> = functions.iter().map(|f| f.name.clone()).collect();
                steps.push(format!("  {}. [FFI]     extern \"{}\" {{ {} }}", step_num, library, fn_names.join(", ")));
                step_num += 1;
            }
            Statement::ExprStmt { expr } => {
                steps.push(format!("  {}. [CALL]    {}", step_num, scalar_to_string(expr)));
                step_num += 1;
            }
        }
    }

    steps
}

/// Plan loop body statements recursively for graph output.
fn plan_loop_body(
    body: &[(Statement, octoflow_parser::ast::Span)],
    steps: &mut Vec<String>,
    step_num: &mut usize,
    known_scalars: &mut std::collections::HashSet<String>,
    depth: usize,
) {
    let indent = "  ".repeat(depth);
    for (body_stmt, _) in body {
        match body_stmt {
            Statement::LetDecl { name: bname, value, mutable } => {
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    if fn_name == "try" {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}.{}  [TRY]     let {} = try({})   → .value .ok .error",
                            step_num, indent, bname, args_str.join(", ")));
                        for field in &["value", "ok", "error"] {
                            known_scalars.insert(format!("{}.{}", bname, field));
                        }
                        *step_num += 1;
                        continue;
                    }
                    if matches!(fn_name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}.{}  [HTTP]    let {} = {}({})   → .status .body .ok .error",
                            step_num, indent, bname, fn_name, args_str.join(", ")));
                        for field in &["status", "body", "ok", "error"] {
                            known_scalars.insert(format!("{}.{}", bname, field));
                        }
                        *step_num += 1;
                        continue;
                    }
                    if fn_name == "exec" {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}.{}  [EXEC]    let {} = exec({})   → .status .output .ok .error",
                            step_num, indent, bname, args_str.join(", ")));
                        for field in &["status", "output", "ok", "error"] {
                            known_scalars.insert(format!("{}.{}", bname, field));
                        }
                        *step_num += 1;
                        continue;
                    }
                    if fn_name == "video_open" {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}.{}  [VIDEO]   let {} = video_open({})   -> .width .height .frames .fps",
                            step_num, indent, bname, args_str.join(", ")));
                        for field in &["width", "height", "frames", "fps"] {
                            known_scalars.insert(format!("{}.{}", bname, field));
                        }
                        known_scalars.insert(bname.clone());
                        *step_num += 1;
                        continue;
                    }
                    if fn_name == "video_frame" {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}.{}  [VIDEO]   let {} = video_frame({})   -> .r .g .b",
                            step_num, indent, bname, args_str.join(", ")));
                        *step_num += 1;
                        continue;
                    }
                    if fn_name == "json_parse" || fn_name == "json_parse_array" {
                        let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
                        steps.push(format!("  {}.{}  [JSON]    let {} = {}({})",
                            step_num, indent, bname, fn_name, args_str.join(", ")));
                        known_scalars.insert(bname.clone());
                        *step_num += 1;
                        continue;
                    }
                }
                let mut_str = if *mutable { "mut " } else { "" };
                let desc = plan_scalar(value, &format!("{}{}", mut_str, bname));
                steps.push(format!("  {}.{}  {}", step_num, indent, desc));
                known_scalars.insert(bname.clone());
                *step_num += 1;
            }
            Statement::Assign { name: aname, value } => {
                let desc = format!("[CPU]     {} = {}", aname, scalar_to_string(value));
                steps.push(format!("  {}.{}  {}", step_num, indent, desc));
                *step_num += 1;
            }
            Statement::ArrayAssign { array, index, value } => {
                steps.push(format!("  {}.{}  [CPU]     {}[{}] = {}", step_num, indent, array, scalar_to_string(index), scalar_to_string(value)));
                *step_num += 1;
            }
            Statement::ArrayPush { array, value } => {
                steps.push(format!("  {}.{}  [CPU]     push({}, {})", step_num, indent, array, scalar_to_string(value)));
                *step_num += 1;
            }
            Statement::Print { segments: segs } => {
                let display: String = segs.iter().map(|seg| match seg {
                    PrintSegment::Literal(s) => s.clone(),
                    PrintSegment::Scalar { name: sname, precision: None } => format!("{{{}}}", sname),
                    PrintSegment::Scalar { name: sname, precision: Some(p) } => format!("{{{}:.{}}}", sname, p),
                    PrintSegment::Expr(_) => "<expr>".to_string(),
                }).collect();
                steps.push(format!("  {}.{}  [PRINT]   print(\"{}\")", step_num, indent, display));
                *step_num += 1;
            }
            Statement::ArrayDecl { name: aname, elements, mutable } => {
                let mut_str = if *mutable { "mut " } else { "" };
                let elems_str: Vec<String> = elements.iter().map(scalar_to_string).collect();
                steps.push(format!("  {}.{}  [ARRAY]   let {}{} = [{}]", step_num, indent, mut_str, aname, elems_str.join(", ")));
                *step_num += 1;
            }
            Statement::WhileLoop { condition, body: inner } => {
                steps.push(format!("  {}.{}  [WHILE]   while {}", step_num, indent, scalar_to_string(condition)));
                *step_num += 1;
                plan_loop_body(inner, steps, step_num, known_scalars, depth + 1);
                steps.push(format!("  {}.{}  [END]", step_num, indent));
                *step_num += 1;
            }
            Statement::ForLoop { var, start, end, body: inner } => {
                steps.push(format!("  {}.{}  [FOR]     for {} in range({}, {})", step_num, indent, var, scalar_to_string(start), scalar_to_string(end)));
                *step_num += 1;
                plan_loop_body(inner, steps, step_num, known_scalars, depth + 1);
                steps.push(format!("  {}.{}  [END]", step_num, indent));
                *step_num += 1;
            }
            Statement::ForEachLoop { var, iterable, body: inner } => {
                steps.push(format!("  {}.{}  [FOREACH] for {} in {}", step_num, indent, var, iterable));
                *step_num += 1;
                plan_loop_body(inner, steps, step_num, known_scalars, depth + 1);
                steps.push(format!("  {}.{}  [END]", step_num, indent));
                *step_num += 1;
            }
            Statement::IfBlock { condition, body: if_body, elif_branches, else_body } => {
                steps.push(format!("  {}.{}  [IF]      if {}", step_num, indent, scalar_to_string(condition)));
                *step_num += 1;
                plan_block_body(if_body, steps, step_num, known_scalars, depth + 1);
                for (elif_cond, elif_body) in elif_branches {
                    steps.push(format!("  {}.{}  [ELIF]    elif {}", step_num, indent, scalar_to_string(elif_cond)));
                    *step_num += 1;
                    plan_block_body(elif_body, steps, step_num, known_scalars, depth + 1);
                }
                if !else_body.is_empty() {
                    steps.push(format!("  {}.{}  [ELSE]", step_num, indent));
                    *step_num += 1;
                    plan_block_body(else_body, steps, step_num, known_scalars, depth + 1);
                }
                steps.push(format!("  {}.{}  [END]", step_num, indent));
                *step_num += 1;
            }
            Statement::Break => {
                steps.push(format!("  {}.{}  [BREAK]", step_num, indent));
                *step_num += 1;
            }
            Statement::Continue => {
                steps.push(format!("  {}.{}  [CONTINUE]", step_num, indent));
                *step_num += 1;
            }
            Statement::Return { value } => {
                steps.push(format!("  {}.{}  [RETURN]  return {}", step_num, indent, scalar_to_string(value)));
                *step_num += 1;
            }
            Statement::WriteFile { path, content } => {
                steps.push(format!("  {}.{}  [IO]      write_file({}, {})", step_num, indent, scalar_to_string(path), scalar_to_string(content)));
                *step_num += 1;
            }
            Statement::AppendFile { path, content } => {
                steps.push(format!("  {}.{}  [IO]      append_file({}, {})", step_num, indent, scalar_to_string(path), scalar_to_string(content)));
                *step_num += 1;
            }
            Statement::SaveData { path, map_name } => {
                steps.push(format!("  {}.{}  [IO]      save_data({}, {})", step_num, indent, scalar_to_string(path), map_name));
                *step_num += 1;
            }
            Statement::WriteCsv { path, array_name } => {
                steps.push(format!("  {}.{}  [IO]      write_csv({}, {})", step_num, indent, scalar_to_string(path), array_name));
                *step_num += 1;
            }
            Statement::WriteBytes { path, array_name } => {
                steps.push(format!("  {}.{}  [IO]      write_bytes({}, {})", step_num, indent, scalar_to_string(path), array_name));
                *step_num += 1;
            }
            Statement::ExprStmt { expr } => {
                steps.push(format!("  {}.{}  [CALL]    {}", step_num, indent, scalar_to_string(expr)));
                *step_num += 1;
            }
            _ => {
                steps.push(format!("  {}.{}  [...]", step_num, indent));
                *step_num += 1;
            }
        }
    }
}

/// Plan output for if-block body (reuses the loop body pattern at given depth).
fn plan_block_body(
    body: &[(Statement, octoflow_parser::ast::Span)],
    steps: &mut Vec<String>,
    step_num: &mut usize,
    known_scalars: &mut std::collections::HashSet<String>,
    depth: usize,
) {
    // Reuse plan_loop_body — it handles all the same statement types
    plan_loop_body(body, steps, step_num, known_scalars, depth);
}

fn plan_expr(expr: &Expr, name: &str, known_scalars: &std::collections::HashSet<String>) -> String {
    match expr {
        Expr::Tap { path } => {
            format!("[TAP]     {} = tap(\"{}\")", name, path)
        }
        Expr::RandomStream { lo, hi, .. } => {
            format!("[RNG]     {} = random_stream(N, {}, {})  [CPU xorshift64*→GPU pipeline]", name, lo, hi)
        }
        Expr::Cache { key, inner } => {
            format!("[CACHE]   {} = cache(\"{}\") {}", name, key,
                plan_expr(inner, name, known_scalars))
        }
        Expr::Ref { name: ref_name } => {
            format!("[REF]     {} = {}", name, ref_name)
        }
        Expr::Pipe { input, stages } => {
            let input_name = match input.as_ref() {
                Expr::Tap { path } => format!("tap(\"{}\")", path),
                Expr::RandomStream { lo, hi, .. } => format!("random_stream(N, {}, {})", lo, hi),
                Expr::Cache { key, .. } => format!("cache(\"{}\")", key),
                Expr::Ref { name } => name.clone(),
                Expr::Pipe { .. } => "...".to_string(),
            };

            // Check for fusion opportunities
            let stages_str: Vec<String> = stages.iter().map(|s| {
                if s.args.is_empty() {
                    format!("{}()", s.operation)
                } else {
                    let args: Vec<String> = s.args.iter().map(|a| match a {
                        Arg::Literal(n) => format!("{}", n),
                        Arg::IntLiteral(n) => format!("{}", n),
                        Arg::Ref(r) => r.clone(),
                    }).collect();
                    format!("{}({})", s.operation, args.join(", "))
                }
            }).collect();
            let pipe_str = stages_str.join(" |> ");

            // Detect operation type
            let has_temporal = stages.iter().any(|s| matches!(s.operation.as_str(), "ema" | "decay"));
            let has_scan = stages.iter().any(|s| s.operation == "prefix_sum");
            let fused = detect_fusion_label(stages, known_scalars);

            let label = if has_temporal {
                "[GPU:TMP]"
            } else if has_scan {
                "[GPU:SCN]"
            } else if fused.is_some() {
                "[GPU:MAP]"
            } else {
                "[GPU:MAP]"
            };

            let suffix = if let Some(fused_name) = fused {
                format!("  [FUSED: {}]", fused_name)
            } else {
                String::new()
            };

            format!("{} {} = {} |> {}{}", label, name, input_name, pipe_str, suffix)
        }
    }
}

fn detect_fusion_label(stages: &[StageCall], _known_scalars: &std::collections::HashSet<String>) -> Option<&'static str> {
    if stages.len() >= 2 {
        if stages[0].operation == "subtract" && stages[1].operation == "divide" {
            return Some("normalize");
        }
        if stages[0].operation == "multiply" && stages[1].operation == "add" {
            return Some("scale_shift");
        }
    }
    None
}

fn plan_scalar(value: &ScalarExpr, name: &str) -> String {
    match value {
        ScalarExpr::Reduce { op, stream } => {
            format!("[GPU:RED] {} = {}({})", name, op, stream)
        }
        ScalarExpr::BinOp { .. } | ScalarExpr::Compare { .. }
        | ScalarExpr::And { .. } | ScalarExpr::Or { .. }
        | ScalarExpr::If { .. } | ScalarExpr::FnCall { .. } => {
            format!("[CPU]     {} = {}", name, scalar_to_string(value))
        }
        ScalarExpr::Ref(r) => format!("[REF]     {} = {}", name, r),
        ScalarExpr::Literal(n) => format!("[CONST]   {} = {}", name, n),
        ScalarExpr::IntLiteral(n) => format!("[CONST]   {} = {}", name, n),
        ScalarExpr::Bool(b) => format!("[CONST]   {} = {}", name, b),
        ScalarExpr::NoneLiteral => format!("[CONST]   {} = none", name),
        ScalarExpr::Index { array, index } => {
            format!("[CPU]     {} = {}[{}]", name, array, scalar_to_string(index))
        }
        ScalarExpr::StringLiteral(s) => format!("[CONST]   {} = \"{}\"", name, s),
        ScalarExpr::Lambda { .. } => format!("[LAMBDA]  {} = fn(...) ... end", name),
        ScalarExpr::ArrayLiteral(elements) => format!("[CONST]   {} = [{}]", name, elements.len()),
    }
}

fn scalar_to_string(expr: &ScalarExpr) -> String {
    match expr {
        ScalarExpr::Reduce { op, stream } => format!("{}({})", op, stream),
        ScalarExpr::Ref(r) => r.clone(),
        ScalarExpr::Literal(n) => format!("{}", n),
        ScalarExpr::IntLiteral(n) => format!("{}", n),
        ScalarExpr::Bool(b) => format!("{}", b),
        ScalarExpr::NoneLiteral => "none".to_string(),
        ScalarExpr::BinOp { left, op, right } => {
            let op_str = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Mod => "%",
                BinOp::Shl => "<<",
                BinOp::Shr => ">>",
                BinOp::BitAnd => "&",
                BinOp::BitOr => "|",
                BinOp::BitXor => "^",
            };
            format!("{} {} {}", scalar_to_string(left), op_str, scalar_to_string(right))
        }
        ScalarExpr::Compare { left, op, right } => {
            let op_str = match op {
                CompareOp::Less => "<",
                CompareOp::Greater => ">",
                CompareOp::LessEqual => "<=",
                CompareOp::GreaterEqual => ">=",
                CompareOp::Equal => "==",
                CompareOp::NotEqual => "!=",
            };
            format!("{} {} {}", scalar_to_string(left), op_str, scalar_to_string(right))
        }
        ScalarExpr::And { left, right } => {
            format!("{} && {}", scalar_to_string(left), scalar_to_string(right))
        }
        ScalarExpr::Or { left, right } => {
            format!("{} || {}", scalar_to_string(left), scalar_to_string(right))
        }
        ScalarExpr::If { condition, then_expr, else_expr } => {
            format!("if {} then {} else {}",
                scalar_to_string(condition),
                scalar_to_string(then_expr),
                scalar_to_string(else_expr))
        }
        ScalarExpr::FnCall { name, args } => {
            let args_str: Vec<String> = args.iter().map(scalar_to_string).collect();
            format!("{}({})", name, args_str.join(", "))
        }
        ScalarExpr::Index { array, index } => {
            format!("{}[{}]", array, scalar_to_string(index))
        }
        ScalarExpr::StringLiteral(s) => format!("\"{}\"", s),
        ScalarExpr::Lambda { params, body } => {
            format!("fn({}) {} end", params.join(", "), scalar_to_string(body))
        }
        ScalarExpr::ArrayLiteral(elements) => {
            let elems: Vec<String> = elements.iter().map(scalar_to_string).collect();
            format!("[{}]", elems.join(", "))
        }
    }
}

/// Evaluate a scalar expression.
fn eval_scalar(
    expr: &ScalarExpr,
    streams: &HashMap<String, Vec<f32>>,
    scalars: &HashMap<String, Value>,
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    arrays: &mut HashMap<String, Vec<Value>>,
    hashmaps: &mut HashMap<String, HashMap<String, Value>>,
    scalar_fns: &HashMap<String, ScalarFnDef>,
    struct_defs: &HashMap<String, Vec<String>>,
    rng: &Cell<u64>,
    mutable_scalars: &std::collections::HashSet<String>,
) -> Result<Value, CliError> {
    match expr {
        ScalarExpr::Reduce { op, stream } => {
            // Check GPU arrays first — native f32 fast path avoids Value overhead
            if let Some(ga) = gpu_array_get(stream) {
                match op.as_str() {
                    "sum" => { return Ok(Value::Float(ga.iter().sum::<f32>())); }
                    "min" => {
                        if ga.is_empty() { return Err(CliError::Compile(format!("min() called on empty array '{}'", stream))); }
                        return Ok(Value::Float(ga.iter().cloned().fold(f32::INFINITY, f32::min)));
                    }
                    "max" => {
                        if ga.is_empty() { return Err(CliError::Compile(format!("max() called on empty array '{}'", stream))); }
                        return Ok(Value::Float(ga.iter().cloned().fold(f32::NEG_INFINITY, f32::max)));
                    }
                    "count" => { return Ok(Value::Float(ga.len() as f32)); }
                    _ => {} // fall through to stream reduce
                }
            }
            // Check if the name refers to an array first (Phase 33: array sum/min/max/count)
            if let Some(arr) = arrays.get(stream) {
                match op.as_str() {
                    "sum" => {
                        let mut total: f32 = 0.0;
                        for val in arr {
                            total += val.as_float().map_err(|_| CliError::Compile("sum() on array requires all elements to be floats".into()))?;
                        }
                        return Ok(Value::Float(total));
                    }
                    "min" => {
                        if arr.is_empty() { return Err(CliError::Compile(format!("min() called on empty array '{}'", stream))); }
                        let mut m = f32::INFINITY;
                        for val in arr { let f = val.as_float().map_err(|_| CliError::Compile("min() on array requires floats".into()))?; if f < m { m = f; } }
                        return Ok(Value::Float(m));
                    }
                    "max" => {
                        if arr.is_empty() { return Err(CliError::Compile(format!("max() called on empty array '{}'", stream))); }
                        let mut m = f32::NEG_INFINITY;
                        for val in arr { let f = val.as_float().map_err(|_| CliError::Compile("max() on array requires floats".into()))?; if f > m { m = f; } }
                        return Ok(Value::Float(m));
                    }
                    "count" => {
                        return Ok(Value::Float(arr.len() as f32));
                    }
                    _ => {} // fall through to stream reduce
                }
            }
            let data = streams
                .get(stream)
                .ok_or_else(|| CliError::UndefinedStream(stream.clone()))?;
            Ok(Value::Float(dispatch_reduce_op(gpu, op, data)?))
        }
        ScalarExpr::BinOp { left, op, right } => {
            let l = eval_scalar(left, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let r = eval_scalar(right, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            match (&l, &r, op) {
                // Int + Int = Int
                (Value::Int(li), Value::Int(ri), _) => Ok(Value::Int(match op {
                    BinOp::Add => li.wrapping_add(*ri),
                    BinOp::Sub => li.wrapping_sub(*ri),
                    BinOp::Mul => li.wrapping_mul(*ri),
                    BinOp::Div => if *ri == 0 { return Err(CliError::Runtime("division by zero".into())); } else { li / ri },
                    BinOp::Mod => if *ri == 0 { return Err(CliError::Runtime("modulo by zero".into())); } else { li % ri },
                    BinOp::Shl => li << (*ri as u32),
                    BinOp::Shr => li >> (*ri as u32),
                    BinOp::BitAnd => li & ri,
                    BinOp::BitOr => li | ri,
                    BinOp::BitXor => li ^ ri,
                })),
                // Float + Float = Float
                (Value::Float(lf), Value::Float(rf), _) => Ok(Value::Float(match op {
                    BinOp::Add => lf + rf,
                    BinOp::Sub => lf - rf,
                    BinOp::Mul => lf * rf,
                    BinOp::Div => lf / rf,
                    BinOp::Mod => lf % rf,
                    BinOp::Shl => ((*lf as u32) << (*rf as u32)) as f32,
                    BinOp::Shr => ((*lf as u32) >> (*rf as u32)) as f32,
                    BinOp::BitAnd => ((*lf as u32) & (*rf as u32)) as f32,
                    BinOp::BitOr => ((*lf as u32) | (*rf as u32)) as f32,
                    BinOp::BitXor => ((*lf as u32) ^ (*rf as u32)) as f32,
                })),
                // Int + Float or Float + Int = Float (auto-promote)
                (Value::Int(li), Value::Float(rf), _) => {
                    let lf = *li as f32;
                    Ok(Value::Float(match op {
                        BinOp::Add => lf + rf,
                        BinOp::Sub => lf - rf,
                        BinOp::Mul => lf * rf,
                        BinOp::Div => lf / rf,
                        BinOp::Mod => lf % rf,
                        BinOp::Shl => ((lf as u32) << (*rf as u32)) as f32,
                        BinOp::Shr => ((lf as u32) >> (*rf as u32)) as f32,
                        BinOp::BitAnd => ((lf as u32) & (*rf as u32)) as f32,
                        BinOp::BitOr => ((lf as u32) | (*rf as u32)) as f32,
                        BinOp::BitXor => ((lf as u32) ^ (*rf as u32)) as f32,
                    }))
                }
                (Value::Float(lf), Value::Int(ri), _) => {
                    let rf = *ri as f32;
                    Ok(Value::Float(match op {
                        BinOp::Add => lf + rf,
                        BinOp::Sub => lf - rf,
                        BinOp::Mul => lf * rf,
                        BinOp::Div => lf / rf,
                        BinOp::Mod => lf % rf,
                        BinOp::Shl => ((*lf as u32) << (rf as u32)) as f32,
                        BinOp::Shr => ((*lf as u32) >> (rf as u32)) as f32,
                        BinOp::BitAnd => ((*lf as u32) & (rf as u32)) as f32,
                        BinOp::BitOr => ((*lf as u32) | (rf as u32)) as f32,
                        BinOp::BitXor => ((*lf as u32) ^ (rf as u32)) as f32,
                    }))
                }
                (Value::Str(ls), Value::Str(rs), BinOp::Add) => {
                    Ok(Value::Str(format!("{}{}", ls, rs)))
                }
                (_, _, BinOp::Add) if l.is_str() || r.is_str() => {
                    Err(CliError::Compile("cannot concatenate string with non-string; both operands must be strings".into()))
                }
                _ if l.is_str() || r.is_str() => {
                    Err(CliError::Compile(format!("cannot apply arithmetic to strings")))
                }
                _ => unreachable!(),
            }
        }
        ScalarExpr::Ref(name) => scalars
            .get(name)
            .cloned()
            .ok_or_else(|| {
                // Check if this looks like a builtin name used without parentheses
                if KNOWN_BUILTINS.contains(&name.as_str()) {
                    return CliError::Compile(format!("'{}' is a function — call it with parentheses: {}()", name, name));
                }
                // Don't suggest if the name might be an array/map (handled elsewhere)
                if arrays.contains_key(name) || hashmaps.contains_key(name) || gpu_array_has(name) {
                    return CliError::UndefinedScalar(name.clone());
                }
                let scalar_names: Vec<&str> = scalars.keys().map(|s| s.as_str()).collect();
                if let Some(suggestion) = crate::suggest_closest(name, &scalar_names) {
                    CliError::Compile(format!("undefined scalar '{}'. Did you mean '{}'?", name, suggestion))
                } else {
                    CliError::UndefinedScalar(name.clone())
                }
            }),
        ScalarExpr::Literal(n) => Ok(Value::Float(*n as f32)),
        ScalarExpr::IntLiteral(n) => Ok(Value::Int(*n)),
        ScalarExpr::Bool(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
        ScalarExpr::NoneLiteral => Ok(Value::None),
        ScalarExpr::StringLiteral(s) => Ok(Value::Str(s.clone())),
        ScalarExpr::If { condition, then_expr, else_expr } => {
            let cond = eval_scalar(condition, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            if cond != 0.0 {
                eval_scalar(then_expr, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)
            } else {
                eval_scalar(else_expr, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)
            }
        }
        ScalarExpr::Compare { left, op, right } => {
            let l = eval_scalar(left, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let r = eval_scalar(right, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let result = match (&l, &r) {
                // Int == Int: exact comparison
                (Value::Int(li), Value::Int(ri)) => match op {
                    CompareOp::Less => li < ri,
                    CompareOp::Greater => li > ri,
                    CompareOp::LessEqual => li <= ri,
                    CompareOp::GreaterEqual => li >= ri,
                    CompareOp::Equal => li == ri,
                    CompareOp::NotEqual => li != ri,
                },
                // Float == Float: tolerance comparison
                (Value::Float(lf), Value::Float(rf)) => match op {
                    CompareOp::Less => lf < rf,
                    CompareOp::Greater => lf > rf,
                    CompareOp::LessEqual => lf <= rf,
                    CompareOp::GreaterEqual => lf >= rf,
                    CompareOp::Equal => (lf - rf).abs() < 1e-6,
                    CompareOp::NotEqual => (lf - rf).abs() >= 1e-6,
                },
                // Int vs Float / Float vs Int: promote to float
                (Value::Int(li), Value::Float(rf)) => {
                    let lf = *li as f32;
                    match op {
                        CompareOp::Less => lf < *rf,
                        CompareOp::Greater => lf > *rf,
                        CompareOp::LessEqual => lf <= *rf,
                        CompareOp::GreaterEqual => lf >= *rf,
                        CompareOp::Equal => (lf - rf).abs() < 1e-6,
                        CompareOp::NotEqual => (lf - rf).abs() >= 1e-6,
                    }
                }
                (Value::Float(lf), Value::Int(ri)) => {
                    let rf = *ri as f32;
                    match op {
                        CompareOp::Less => *lf < rf,
                        CompareOp::Greater => *lf > rf,
                        CompareOp::LessEqual => *lf <= rf,
                        CompareOp::GreaterEqual => *lf >= rf,
                        CompareOp::Equal => (lf - rf).abs() < 1e-6,
                        CompareOp::NotEqual => (lf - rf).abs() >= 1e-6,
                    }
                }
                (Value::Str(ls), Value::Str(rs)) => match op {
                    CompareOp::Equal => ls == rs,
                    CompareOp::NotEqual => ls != rs,
                    _ => return Err(CliError::Compile(format!(
                        "cannot use ordered comparison on strings (only == and != are supported)"
                    ))),
                },
                // none == none → true, none == anything → false
                (Value::None, Value::None) => match op {
                    CompareOp::Equal => true,
                    CompareOp::NotEqual => false,
                    _ => return Err(CliError::Compile("cannot use ordered comparison on none".into())),
                },
                (Value::None, _) | (_, Value::None) => match op {
                    CompareOp::Equal => false,
                    CompareOp::NotEqual => true,
                    _ => return Err(CliError::Compile("cannot use ordered comparison on none".into())),
                },
                _ => {
                    let ltype = match &l {
                        Value::Float(_) => "float",
                        Value::Int(_) => "int",
                        Value::Str(_) => "string",
                        Value::Map(_) => "map",
                        Value::None => "none",
                    };
                    let rtype = match &r {
                        Value::Float(_) => "float",
                        Value::Int(_) => "int",
                        Value::Str(_) => "string",
                        Value::Map(_) => "map",
                        Value::None => "none",
                    };
                    return Err(CliError::Compile(format!(
                        "cannot compare {} with {} (mismatched types)", ltype, rtype
                    )));
                }
            };
            Ok(Value::Float(if result { 1.0 } else { 0.0 }))
        }
        ScalarExpr::And { left, right } => {
            let l = eval_scalar(left, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            if l == 0.0 {
                Ok(Value::Float(0.0))
            } else {
                let r = eval_scalar(right, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
                Ok(Value::Float(if r != 0.0 { 1.0 } else { 0.0 }))
            }
        }
        ScalarExpr::Or { left, right } => {
            let l = eval_scalar(left, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            if l != 0.0 {
                Ok(Value::Float(1.0))
            } else {
                let r = eval_scalar(right, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
                Ok(Value::Float(if r != 0.0 { 1.0 } else { 0.0 }))
            }
        }
        ScalarExpr::FnCall { name, args } => {
            // vm_* deprecation warnings — suggest loom_* equivalents (once per name)
            if name.starts_with("vm_") {
                thread_local! {
                    static DEPRECATED_WARNED: RefCell<std::collections::HashSet<String>> =
                        RefCell::new(std::collections::HashSet::new());
                }
                DEPRECATED_WARNED.with(|warned| {
                    let mut w = warned.borrow_mut();
                    if w.insert(name.to_string()) {
                        let loom_name = format!("loom_{}", &name[3..]);
                        eprintln!("warning: {}() is deprecated, use {}() instead (this warning shown once)", name, loom_name);
                    }
                });
            }
            // Guard: try() must be used with let, not as bare expression
            if name == "try" {
                return Err(CliError::Compile("try() must be used with let: `let r = try(expr)`".into()));
            }
            // Guard: http_*() must be used with let
            if matches!(name.as_str(), "http_get" | "http_post" | "http_put" | "http_delete") {
                return Err(CliError::Compile(format!("{}() must be used with let: `let r = {}(url)`", name, name)));
            }
            // Guard: json_parse() / json_parse_array() must be used with let
            if matches!(name.as_str(), "json_parse" | "json_parse_array") {
                return Err(CliError::Compile(format!("{}() must be used with let: `let data = {}(str)`", name, name)));
            }
            // Guard: load_data() must be used with let
            if name == "load_data" {
                return Err(CliError::Compile("load_data() must be used with let: `let mut config = load_data(path)`".into()));
            }
            // sum/min/max/count as FnCall — works with array literals and array refs
            if matches!(name.as_str(), "sum" | "min" | "max" | "count") && args.len() == 1 {
                // Collect array values — either from ArrayLiteral or named array ref
                let values: Vec<f32> = if let ScalarExpr::ArrayLiteral(elements) = &args[0] {
                    let mut vals = Vec::new();
                    for elem in elements {
                        let v = eval_scalar(elem, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                        vals.push(v.as_float().map_err(|_| CliError::Compile(format!("{}() requires numeric elements", name)))?);
                    }
                    vals
                } else if let ScalarExpr::Ref(arr_name) = &args[0] {
                    gpu_array_materialize(arr_name, arrays);
                    if let Some(ga) = gpu_array_get(arr_name) {
                        ga.to_vec()
                    } else if let Some(arr) = arrays.get(arr_name) {
                        let mut vals = Vec::new();
                        for v in arr {
                            vals.push(v.as_float().map_err(|_| CliError::Compile(format!("{}() requires numeric elements", name)))?);
                        }
                        vals
                    } else {
                        return Err(CliError::Compile(format!("{}(): undefined array '{}'", name, arr_name)));
                    }
                } else {
                    return Err(CliError::Compile(format!("{}() requires an array argument", name)));
                };
                return match name.as_str() {
                    "sum" => Ok(Value::Float(values.iter().sum())),
                    "min" => {
                        if values.is_empty() { return Err(CliError::Compile(format!("min() called on empty array"))); }
                        Ok(Value::Float(values.iter().cloned().fold(f32::INFINITY, f32::min)))
                    }
                    "max" => {
                        if values.is_empty() { return Err(CliError::Compile(format!("max() called on empty array"))); }
                        Ok(Value::Float(values.iter().cloned().fold(f32::NEG_INFINITY, f32::max)))
                    }
                    "count" => Ok(Value::Float(values.len() as f32)),
                    _ => unreachable!(),
                };
            }
            // Handle reduce(arr, init, fn(acc, x) expr end) → fold array to scalar
            if name == "reduce" && args.len() == 3 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    gpu_array_materialize(arr_name, arrays);
                    let init = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let (params, body) = extract_lambda(&args[2])?;
                    if params.len() != 2 {
                        return Err(CliError::Compile("reduce() lambda must take exactly 2 parameters (accumulator, element)".into()));
                    }
                    let source = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile(format!("reduce() requires array, '{}' not found", arr_name)))?
                        .clone();
                    let mut acc = init;
                    for elem in &source {
                        acc = invoke_lambda(params, body, &[acc, elem.clone()], scalars, streams, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng)?;
                    }
                    return Ok(acc);
                }
                return Err(CliError::Compile("reduce() first argument must be an array name".into()));
            }
            // ── GPU scalar reductions (Phase 75a) ────────────────────
            if name == "gpu_sum" && args.len() == 1 {
                let arr = extract_array_arg("gpu_sum", &args[0], arrays)?;
                return Ok(Value::Float(dispatch_reduce_op(gpu, "sum", &arr)?));
            }
            if name == "gpu_min" && args.len() == 1 {
                let arr = extract_array_arg("gpu_min", &args[0], arrays)?;
                return Ok(Value::Float(dispatch_reduce_op(gpu, "min", &arr)?));
            }
            if name == "gpu_max" && args.len() == 1 {
                let arr = extract_array_arg("gpu_max", &args[0], arrays)?;
                return Ok(Value::Float(dispatch_reduce_op(gpu, "max", &arr)?));
            }
            if name == "gpu_mean" && args.len() == 1 {
                let arr = extract_array_arg("gpu_mean", &args[0], arrays)?;
                if arr.is_empty() { return Err(CliError::Compile("gpu_mean(): empty array".into())); }
                let sum = dispatch_reduce_op(gpu, "sum", &arr)?;
                return Ok(Value::Float(sum / arr.len() as f32));
            }
            // ── GPU Tier 2 reductions (Phase 75b) ─────────────────────────
            if name == "gpu_product" && args.len() == 1 {
                let arr = extract_array_arg("gpu_product", &args[0], arrays)?;
                if arr.is_empty() { return Err(CliError::Compile("gpu_product(): empty array".into())); }
                return Ok(Value::Float(dispatch_reduce_op(gpu, "mul", &arr)?));
            }
            if name == "gpu_variance" && args.len() == 1 {
                let arr = extract_array_arg("gpu_variance", &args[0], arrays)?;
                let n = arr.len();
                if n < 2 { return Err(CliError::Compile("gpu_variance(): need at least 2 elements".into())); }
                let sum = dispatch_reduce_op(gpu, "sum", &arr)?;
                let mean = sum / n as f32;
                // Compute sum of squared deviations on GPU
                let mean_arr = vec![mean; n];
                if let Some(ref gpu_dev) = gpu {
                    let mean_buf = octoflow_vulkan::upload_buffer(gpu_dev, &mean_arr)
                        .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                    let arr_buf = octoflow_vulkan::upload_buffer(gpu_dev, &arr)
                        .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                    let diff = octoflow_vulkan::dispatch_binop_resident(gpu_dev, BinaryOp::Sub, arr_buf.as_ref(), mean_buf.as_ref())
                        .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                    let sq = octoflow_vulkan::dispatch_binop_resident(gpu_dev, BinaryOp::Mul, diff.as_ref(), diff.as_ref())
                        .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                    let sq_data = sq.download().map_err(|e| CliError::Gpu(e))?;
                    let sum_sq = dispatch_reduce_op(gpu, "sum", &sq_data)?;
                    // FIX: Use N-1 for sample variance (not N for population variance)
                    return Ok(Value::Float(sum_sq / (n - 1) as f32));
                } else {
                    // CPU fallback - also use N-1 for sample variance
                    let sum_sq: f32 = arr.iter().map(|x| (x - mean) * (x - mean)).sum();
                    return Ok(Value::Float(sum_sq / (n - 1) as f32));
                }
            }
            if name == "gpu_stddev" && args.len() == 1 {
                let arr = extract_array_arg("gpu_stddev", &args[0], arrays)?;
                let n = arr.len();
                if n < 2 { return Err(CliError::Compile("gpu_stddev(): need at least 2 elements".into())); }
                let sum = dispatch_reduce_op(gpu, "sum", &arr)?;
                let mean = sum / n as f32;
                if let Some(ref gpu_dev) = gpu {
                    let mean_buf = octoflow_vulkan::upload_buffer(gpu_dev, &vec![mean; n])
                        .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                    let arr_buf = octoflow_vulkan::upload_buffer(gpu_dev, &arr)
                        .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                    let diff = octoflow_vulkan::dispatch_binop_resident(gpu_dev, BinaryOp::Sub, arr_buf.as_ref(), mean_buf.as_ref())
                        .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                    let sq = octoflow_vulkan::dispatch_binop_resident(gpu_dev, BinaryOp::Mul, diff.as_ref(), diff.as_ref())
                        .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                    let sq_data = sq.download().map_err(|e| CliError::Gpu(e))?;
                    let sum_sq = dispatch_reduce_op(gpu, "sum", &sq_data)?;
                    // FIX: Use N-1 for sample standard deviation
                    return Ok(Value::Float((sum_sq / (n - 1) as f32).sqrt()));
                } else {
                    let sum_sq: f32 = arr.iter().map(|x| (x - mean) * (x - mean)).sum();
                    // FIX: Use N-1 for sample standard deviation
                    return Ok(Value::Float((sum_sq / (n - 1) as f32).sqrt()));
                }
            }
            // ── GPU filesystem: save GPU array to disk (Phase 80) ────────
            if name == "gpu_save_csv" && args.len() == 2 {
                let arr = extract_array_arg("gpu_save_csv", &args[0], arrays)?;
                let path_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("gpu_save_csv() path must be a string".into()))?.to_string();
                check_write_permission_for(&path)?;
                csv_write_floats(&path, &arr)?;
                return Ok(Value::Float(arr.len() as f32));
            }
            if name == "gpu_save_binary" && args.len() == 2 {
                let arr = extract_array_arg("gpu_save_binary", &args[0], arrays)?;
                let path_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("gpu_save_binary() path must be a string".into()))?.to_string();
                check_write_permission_for(&path)?;
                let bytes: Vec<u8> = arr.iter().flat_map(|f| f.to_le_bytes()).collect();
                if let Some(parent) = std::path::Path::new(&path).parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| CliError::Io(format!("gpu_save_binary(\"{}\"): {}", path, e)))?;
                    }
                }
                std::fs::write(&path, &bytes)
                    .map_err(|e| CliError::Io(format!("gpu_save_binary(\"{}\"): {}", path, e)))?;
                return Ok(Value::Float(arr.len() as f32));
            }
            // ── dot product: sum(a[i] * b[i]) ──────────────────────────
            if name == "dot" && args.len() == 2 {
                let a = extract_array_arg("dot", &args[0], arrays)?;
                let b = extract_array_arg("dot", &args[1], arrays)?;
                if a.len() != b.len() {
                    return Err(CliError::Compile(format!("dot(): arrays must have same length ({} vs {})", a.len(), b.len())));
                }
                let product = dispatch_gpu_binop(gpu, BinaryOp::Mul, &a, &b)?;
                let sum = dispatch_reduce_op(gpu, "sum", &product)?;
                return Ok(Value::Float(sum));
            }
            // ── L2 norm: sqrt(sum(x^2)) ─────────────────────────────────
            if name == "norm" && args.len() == 1 {
                let arr = extract_array_arg("norm", &args[0], arrays)?;
                let squared = dispatch_gpu_map(gpu, MapOp::Pow(2.0), &arr)?;
                let sum = dispatch_reduce_op(gpu, "sum", &squared)?;
                return Ok(Value::Float(sum.sqrt()));
            }
            // ── cosine_similarity(a, b): dot(a,b) / (norm(a) * norm(b)) ──
            if name == "cosine_similarity" && args.len() == 2 {
                let a = extract_array_arg("cosine_similarity", &args[0], arrays)?;
                let b = extract_array_arg("cosine_similarity", &args[1], arrays)?;
                if a.len() != b.len() {
                    return Err(CliError::Compile(format!(
                        "cosine_similarity(): arrays must have same length ({} vs {})", a.len(), b.len()
                    )));
                }
                if a.is_empty() {
                    return Ok(Value::Float(0.0));
                }
                // dot product
                let product = dispatch_gpu_binop(gpu, BinaryOp::Mul, &a, &b)?;
                let dot = dispatch_reduce_op(gpu, "sum", &product)?;
                // norms
                let a_sq = dispatch_gpu_map(gpu, MapOp::Pow(2.0), &a)?;
                let b_sq = dispatch_gpu_map(gpu, MapOp::Pow(2.0), &b)?;
                let norm_a = dispatch_reduce_op(gpu, "sum", &a_sq)?.sqrt();
                let norm_b = dispatch_reduce_op(gpu, "sum", &b_sq)?.sqrt();
                let denom = norm_a * norm_b;
                if denom == 0.0 {
                    return Ok(Value::Float(0.0));
                }
                return Ok(Value::Float(dot / denom));
            }
            // Handle time() — Unix epoch seconds as f32
            if name == "time" && args.is_empty() {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64() as f32;
                return Ok(Value::Float(secs));
            }
            // Handle env(name) — environment variable lookup
            if name == "env" && args.len() == 1 {
                let key = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let key_str = key.as_str().map_err(|_| CliError::Compile("env() argument must be a string".into()))?.to_string();
                return Ok(Value::Str(std::env::var(&key_str).unwrap_or_default()));
            }
            // Handle os_name() — returns "windows", "linux", or "macos"
            if name == "os_name" && args.is_empty() {
                let os = if cfg!(target_os = "windows") { "windows" }
                    else if cfg!(target_os = "macos") { "macos" }
                    else { "linux" };
                return Ok(Value::Str(os.to_string()));
            }
            // sleep(ms) — pause execution for N milliseconds (OS boundary)
            if name == "sleep" && args.len() == 1 {
                let ms_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ms = ms_val.as_float()? as u64;
                std::thread::sleep(std::time::Duration::from_millis(ms));
                return Ok(Value::Float(0.0));
            }
            // print_raw(str) — write string to stdout without trailing newline (OS boundary)
            if name == "print_raw" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = val.as_str()?;
                use std::io::Write;
                let stdout = std::io::stdout();
                let mut out = stdout.lock();
                let _ = out.write_all(s.as_bytes());
                let _ = out.flush();
                return Ok(Value::Float(0.0));
            }
            // chat_emit_token(text) — stream a token to the chat callback (used by LLM generation)
            if name == "chat_emit_token" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let text = match &val {
                    Value::Str(s) => s.clone(),
                    Value::Float(f) => format!("{}", f),
                    Value::Int(i) => format!("{}", i),
                    Value::Map(_) => String::new(),
                    Value::None => "none".to_string(),
                };
                TOKEN_CALLBACK.with(|tc| {
                    if let Some(ref mut cb) = *tc.borrow_mut() {
                        cb(&text);
                    }
                });
                return Ok(Value::Float(0.0));
            }

            // gguf_tokens_per_sec() — return the last measured inference throughput (tokens/sec).
            if name == "gguf_tokens_per_sec" && args.is_empty() {
                let bits = crate::chat::INFERENCE_LAST_TOK_PER_SEC.load(std::sync::atomic::Ordering::Relaxed);
                let tok_per_sec = f64::from_bits(bits) as f32;
                return Ok(Value::Float(tok_per_sec));
            }

            // ── Grammar-Constrained Decoding builtins ──

            // grammar_load(path) — load a GBNF grammar file, store in thread-local state.
            // Returns 1.0 on success, 0.0 on failure.
            if name == "grammar_load" && args.len() == 1 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("grammar_load: path must be string".into()))?.to_string();
                check_read_permission_for(&path)?;
                match std::fs::read_to_string(&path) {
                    Ok(source) => match grammar::grammar_load(&source) {
                        Ok(state) => {
                            grammar::set_grammar_state(state);
                            return Ok(Value::Float(1.0));
                        }
                        Err(e) => {
                            eprintln!("grammar_load: {}", e);
                            return Ok(Value::Float(0.0));
                        }
                    },
                    Err(e) => {
                        eprintln!("grammar_load: cannot read '{}': {}", path, e);
                        return Ok(Value::Float(0.0));
                    }
                }
            }

            // grammar_load_str(source) — load a GBNF grammar from a string literal.
            if name == "grammar_load_str" && args.len() == 1 {
                let source = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("grammar_load_str: source must be string".into()))?.to_string();
                match grammar::grammar_load(&source) {
                    Ok(state) => {
                        grammar::set_grammar_state(state);
                        return Ok(Value::Float(1.0));
                    }
                    Err(e) => {
                        eprintln!("grammar_load_str: {}", e);
                        return Ok(Value::Float(0.0));
                    }
                }
            }

            // grammar_advance(token_text) — advance grammar state after sampling a token.
            if name == "grammar_advance" && args.len() == 1 {
                let text = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("grammar_advance: token_text must be string".into()))?.to_string();
                grammar::advance_grammar(&text);
                return Ok(Value::Float(0.0));
            }

            // grammar_reset() — reset grammar state to initial position.
            if name == "grammar_reset" && args.len() == 0 {
                grammar::GRAMMAR_STATE.with(|gs| {
                    if let Some(ref mut state) = *gs.borrow_mut() {
                        grammar::grammar_reset(state);
                    }
                });
                return Ok(Value::Float(0.0));
            }

            // grammar_active() — returns 1.0 if a grammar is loaded, 0.0 otherwise.
            if name == "grammar_active" && args.len() == 0 {
                let active = grammar::GRAMMAR_STATE.with(|gs| gs.borrow().is_some());
                return Ok(Value::Float(if active { 1.0 } else { 0.0 }));
            }

            // print_bytes(arr) — write array of byte values (0-255) to stdout (OS boundary)
            if name == "print_bytes" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    use std::io::Write;
                    // Materialize GPU arrays to CPU if needed
                    gpu_array_materialize(arr_name, arrays);
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile(format!("print_bytes(): undefined array '{}'", arr_name)))?;
                    let bytes: Vec<u8> = arr.iter().map(|v| {
                        let f = v.as_float().unwrap_or(0.0);
                        (f.clamp(0.0, 255.0)) as u8
                    }).collect();
                    let stdout = std::io::stdout();
                    let mut out = stdout.lock();
                    let _ = out.write_all(&bytes);
                    let _ = out.flush();
                    return Ok(Value::Float(bytes.len() as f32));
                }
                return Err(CliError::Compile("print_bytes() requires an array argument".into()));
            }
            // to_str(n) — convert number to string ("42.0" → "42", "3.14" → "3.14")
            // ── assert / panic / format / clone (v1.2) ────────────────
            // assert(cond) — halt if condition is 0.0 (false)
            if name == "assert" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                if val.as_float().unwrap_or(0.0) == 0.0 {
                    return Err(CliError::Runtime("assertion failed".into()));
                }
                return Ok(Value::Float(1.0));
            }
            // assert(cond, msg) — halt with message if condition is 0.0
            if name == "assert" && args.len() == 2 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                if val.as_float().unwrap_or(0.0) == 0.0 {
                    let msg = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let msg_str = match &msg {
                        Value::Str(s) => s.clone(),
                        Value::Float(f) => format!("{}", f),
                        Value::Int(i) => format!("{}", i),
                        Value::Map(_) => "assertion failed".to_string(),
                        Value::None => "assertion failed (none)".to_string(),
                    };
                    return Err(CliError::Runtime(format!("assertion failed: {}", msg_str)));
                }
                return Ok(Value::Float(1.0));
            }
            // panic(msg) — always halt with message
            if name == "panic" && args.len() == 1 {
                let msg = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let msg_str = match &msg {
                    Value::Str(s) => s.clone(),
                    Value::Float(f) => format!("{}", f),
                    Value::Int(i) => format!("{}", i),
                    Value::Map(_) => "panic".to_string(),
                    Value::None => "panic (none)".to_string(),
                };
                return Err(CliError::Runtime(msg_str));
            }
            // format(template, ...) — string interpolation with {}
            if name == "format" && args.len() >= 1 {
                let template = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let template_str = match &template {
                    Value::Str(s) => s.clone(),
                    other => format!("{}", other),
                };
                let mut result = template_str.clone();
                for arg in &args[1..] {
                    let val = eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let val_str = match &val {
                        Value::Str(s) => s.clone(),
                        Value::Float(f) => {
                            if *f == f.floor() && f.abs() < 1e15 && f.is_finite() {
                                format!("{}", *f as i64)
                            } else {
                                format!("{}", f)
                            }
                        }
                        Value::Int(i) => format!("{}", i),
                        Value::Map(_) => format!("{}", val),
                        Value::None => "none".to_string(),
                    };
                    if let Some(pos) = result.find("{}") {
                        result.replace_range(pos..pos + 2, &val_str);
                    }
                }
                return Ok(Value::Str(result));
            }
            // clone(arr) — deep copy an array
            if name == "clone" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    gpu_array_materialize(arr_name, arrays);
                    if let Some(arr) = arrays.get(arr_name) {
                        let cloned = arr.clone();
                        // Return the length — caller should use `let new_arr = clone(old_arr)` which
                        // triggers the RETURNED_ARRAY side-channel in the let handler.
                        RETURNED_ARRAY.with(|ra| {
                            *ra.borrow_mut() = Some(cloned);
                        });
                        return Ok(Value::Float(arr.len() as f32));
                    }
                    return Err(CliError::Compile(format!("clone(): undefined array '{}'", arr_name)));
                }
                return Err(CliError::Compile("clone() requires an array name argument".into()));
            }
            if name == "to_str" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let n = val.as_float()?;
                let s = if n == n.floor() && n.abs() < 1e15 {
                    format!("{}", n as i64)
                } else {
                    format!("{}", n)
                };
                return Ok(Value::Str(s));
            }
            // term_move_up(n) — move cursor up N lines (for animation frame overwrite)
            if name == "term_move_up" && args.len() == 1 {
                let n_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let n = n_val.as_float()? as usize;
                if n > 0 {
                    use std::io::Write;
                    let stdout = std::io::stdout();
                    let mut out = stdout.lock();
                    let _ = write!(out, "\x1b[{}A", n);
                    let _ = out.flush();
                }
                return Ok(Value::Float(0.0));
            }
            // term_clear() — clear screen and move cursor to home
            if name == "term_clear" && args.is_empty() {
                use std::io::Write;
                let stdout = std::io::stdout();
                let mut out = stdout.lock();
                let _ = write!(out, "\x1b[2J\x1b[H");
                let _ = out.flush();
                return Ok(Value::Float(0.0));
            }
            // ── Phase 83: Terminal Pixel Graphics ────────────────────────────
            // term_supports_graphics() → "kitty" | "sixel" | "halfblock"
            if name == "term_supports_graphics" && args.is_empty() {
                return Ok(Value::Str(term_supports_graphics_impl().to_string()));
            }
            // term_image(width, height, r_arr, g_arr, b_arr [, mode]) — display pixels in terminal
            if name == "term_image" && (args.len() == 5 || args.len() == 6) {
                let w_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let w = w_val.as_float()? as usize;
                let h = h_val.as_float()? as usize;
                let r_arr = extract_array_arg("term_image", &args[2], arrays)?;
                let g_arr = extract_array_arg("term_image", &args[3], arrays)?;
                let b_arr = extract_array_arg("term_image", &args[4], arrays)?;
                let mode_str = if args.len() == 6 {
                    let m = eval_scalar(&args[5], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    Some(m.as_str()?.to_string())
                } else { None };
                let total = w * h;
                if r_arr.len() < total || g_arr.len() < total || b_arr.len() < total {
                    return Err(CliError::Compile(format!(
                        "term_image(): arrays must have at least {}*{}={} elements (got r={}, g={}, b={})",
                        w, h, total, r_arr.len(), g_arr.len(), b_arr.len())));
                }
                let mut rgb = Vec::with_capacity(total * 3);
                for i in 0..total {
                    rgb.push(r_arr[i].clamp(0.0, 255.0) as u8);
                    rgb.push(g_arr[i].clamp(0.0, 255.0) as u8);
                    rgb.push(b_arr[i].clamp(0.0, 255.0) as u8);
                }
                term_image_dispatch(w, h, &rgb, None, mode_str.as_deref());
                return Ok(Value::Float(0.0));
            }
            // term_image_at(width, height, r_arr, g_arr, b_arr, id) — with image ID for animation
            if name == "term_image_at" && args.len() == 6 {
                let w_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let w = w_val.as_float()? as usize;
                let h = h_val.as_float()? as usize;
                let r_arr = extract_array_arg("term_image_at", &args[2], arrays)?;
                let g_arr = extract_array_arg("term_image_at", &args[3], arrays)?;
                let b_arr = extract_array_arg("term_image_at", &args[4], arrays)?;
                let id_val = eval_scalar(&args[5], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let id = id_val.as_float()? as u8;
                let total = w * h;
                if r_arr.len() < total || g_arr.len() < total || b_arr.len() < total {
                    return Err(CliError::Compile(format!(
                        "term_image_at(): arrays must have at least {}*{}={} elements",
                        w, h, total)));
                }
                let mut rgb = Vec::with_capacity(total * 3);
                for i in 0..total {
                    rgb.push(r_arr[i].clamp(0.0, 255.0) as u8);
                    rgb.push(g_arr[i].clamp(0.0, 255.0) as u8);
                    rgb.push(b_arr[i].clamp(0.0, 255.0) as u8);
                }
                term_image_dispatch(w, h, &rgb, Some(id), None);
                return Ok(Value::Float(0.0));
            }

            // ── Phase 85: Window/GUI builtins ────────────────────────────
            if name == "window_open" && args.len() == 3 {
                let w_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let t_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let w = w_val.as_float()? as u32;
                let h = h_val.as_float()? as u32;
                let title = t_val.as_str()?.to_string();
                match crate::window_io::window_open_impl(w, h, &title) {
                    Ok(true) => return Ok(Value::Float(1.0)),
                    Ok(false) => return Ok(Value::Float(0.0)),
                    Err(e) => return Err(CliError::Runtime(e)),
                }
            }
            if name == "window_close" && args.is_empty() {
                crate::window_io::window_close_impl();
                return Ok(Value::Float(0.0));
            }
            if name == "window_alive" && args.is_empty() {
                return Ok(Value::Float(if crate::window_io::window_alive_impl() { 1.0 } else { 0.0 }));
            }
            if name == "window_draw" && args.len() == 3 {
                let r_arr = extract_array_arg("window_draw", &args[0], arrays)?;
                let g_arr = extract_array_arg("window_draw", &args[1], arrays)?;
                let b_arr = extract_array_arg("window_draw", &args[2], arrays)?;
                crate::window_io::window_draw_impl(&r_arr, &g_arr, &b_arr)
                    .map_err(|e| CliError::Runtime(e))?;
                return Ok(Value::Float(0.0));
            }
            // vm_present(vm_id, total) — batch download R/G/B in ONE staging submission + window_draw
            // Replaces 3× vm_read_globals + window_draw: 3 fence waits → 1 fence wait.
            if (name == "vm_present" || name == "loom_present") && args.len() == 2 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let total = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("vm_present: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                // Settle homeostasis pacing debt (batch: one sleep per frame)
                let debt = LOOM_PACE_DEBT_US.with(|d| { let v = d.get(); d.set(0); v });
                if debt > 0 {
                    std::thread::sleep(std::time::Duration::from_micros(debt));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };

                // Async present: double-buffer pattern
                // 1. Complete previous frame's GPU→CPU download (if any) and blit it
                let prev_pending = PENDING_FB_READ.with(|p| p.borrow_mut().take());
                let prev_rgb = PENDING_FB_RGB.with(|p| p.borrow_mut().take());
                let blit_data = if let Some(pending) = prev_pending {
                    // Wait for previous frame's fence + download
                    Some(octoflow_vulkan::vm::vm_finish_fb_read(gpu_dev, pending)
                        .map_err(|e| CliError::Runtime(format!("vm_present: {}", e)))?)
                } else {
                    prev_rgb
                };
                if let Some((r, g, b)) = blit_data {
                    crate::window_io::window_draw_impl(&r, &g, &b)
                        .map_err(|e| CliError::Runtime(e))?;
                }

                // 2. Submit THIS frame's GPU→staging copy (non-blocking)
                let (pending, immediate) = GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_present: unknown VM {}", vm_id)))?;
                    octoflow_vulkan::vm::vm_submit_fb_copy(gpu_dev, vm, total)
                        .map_err(|e| CliError::Runtime(format!("vm_present: {}", e)))
                })?;

                if let Some(pend) = pending {
                    // DEVICE_LOCAL: store pending, GPU copy runs while CPU does next frame
                    PENDING_FB_READ.with(|p| *p.borrow_mut() = Some(pend));
                } else if let Some(rgb) = immediate {
                    // HOST_VISIBLE: already have data, store for next frame's blit
                    PENDING_FB_RGB.with(|p| *p.borrow_mut() = Some(rgb));
                }

                return Ok(Value::Float(0.0));
            }
            if name == "window_poll" && args.is_empty() {
                return Ok(Value::Str(crate::window_io::window_poll_impl()));
            }
            if name == "window_event_key" && args.is_empty() {
                return Ok(Value::Str(crate::window_io::window_event_key_impl()));
            }
            if name == "window_event_x" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::window_event_x_impl()));
            }
            if name == "window_event_y" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::window_event_y_impl()));
            }
            if name == "window_width" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::window_width_impl() as f32));
            }
            if name == "window_height" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::window_height_impl() as f32));
            }
            if name == "window_title" && args.len() == 1 {
                let t_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let title = t_val.as_str()?.to_string();
                crate::window_io::window_title_impl(&title);
                return Ok(Value::Float(0.0));
            }

            // ── Phase 3H: New window/GUI builtins ────────────────────────
            // H-01: WM_CHAR text input
            if name == "window_event_char" && args.is_empty() {
                return Ok(Value::Str(crate::window_io::window_event_char_impl()));
            }
            // H-03: WM_MOUSEWHEEL scroll
            if name == "window_event_scroll" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::window_event_scroll_impl()));
            }
            // H-04: Mouse capture
            if name == "window_capture_mouse" && args.is_empty() {
                crate::window_io::window_capture_mouse_impl();
                return Ok(Value::Float(0.0));
            }
            if name == "window_release_mouse" && args.is_empty() {
                crate::window_io::window_release_mouse_impl();
                return Ok(Value::Float(0.0));
            }
            // H-05: Clipboard
            if name == "clipboard_get" && args.is_empty() {
                return Ok(Value::Str(crate::window_io::clipboard_get_impl()));
            }
            if name == "clipboard_set" && args.len() == 1 {
                let text = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                crate::window_io::clipboard_set_impl(&text);
                return Ok(Value::Float(0.0));
            }
            // H-06: File dialogs
            if name == "dialog_open_file" && args.len() == 2 {
                let title = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                let filter = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                return Ok(Value::Str(crate::window_io::dialog_open_file_impl(&title, &filter)));
            }
            if name == "dialog_save_file" && args.len() == 2 {
                let title = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                let filter = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                return Ok(Value::Str(crate::window_io::dialog_save_file_impl(&title, &filter)));
            }
            if name == "dialog_message" && args.len() == 3 {
                let title = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                let message = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                let msg_type = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                return Ok(Value::Str(crate::window_io::dialog_message_impl(&title, &message, &msg_type)));
            }
            // H-07: Cursor shapes
            if name == "window_set_cursor" && args.len() == 1 {
                let cursor = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                crate::window_io::window_set_cursor_impl(&cursor);
                return Ok(Value::Float(0.0));
            }
            // H-08: Timers
            if name == "window_set_timer" && args.len() == 2 {
                let id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let ms = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                crate::window_io::window_set_timer_impl(id, ms);
                return Ok(Value::Float(0.0));
            }
            if name == "window_kill_timer" && args.len() == 1 {
                let id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                crate::window_io::window_kill_timer_impl(id);
                return Ok(Value::Float(0.0));
            }
            if name == "window_event_timer_id" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::window_event_timer_id_impl()));
            }
            // H-09: Menu bars
            if name == "window_create_menu" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::window_create_menu_impl() as f32));
            }
            if name == "menu_add_item" && args.len() == 3 {
                let menu = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as f64;
                let id = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let label = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                crate::window_io::menu_add_item_impl(menu, id, &label);
                return Ok(Value::Float(0.0));
            }
            if name == "menu_add_submenu" && args.len() == 3 {
                let menu = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as f64;
                let label = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str()?.to_string();
                let submenu = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as f64;
                crate::window_io::menu_add_submenu_impl(menu, &label, submenu);
                return Ok(Value::Float(0.0));
            }
            if name == "window_set_menu" && args.len() == 1 {
                let menu = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as f64;
                crate::window_io::window_set_menu_impl(menu);
                return Ok(Value::Float(0.0));
            }
            // H-11: DPI awareness
            if name == "window_dpi" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::window_dpi_impl()));
            }
            // R-06: Mouse button state
            if name == "gui_mouse_down" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::gui_mouse_down_impl()));
            }
            if name == "gui_mouse_buttons" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::gui_mouse_buttons_impl()));
            }
            // gui_scroll_y() — accumulated scroll delta (positive=up, negative=down), resets on read
            if name == "gui_scroll_y" && args.is_empty() {
                return Ok(Value::Float(crate::window_io::gui_scroll_y_impl()));
            }
            // R-08: Key held state (continuous)
            if name == "window_key_held" && args.len() == 1 {
                let key = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let key_str = key.as_str().map_err(|_| CliError::Compile("window_key_held() requires a string key name".into()))?;
                return Ok(Value::Float(crate::window_io::window_key_held_impl(key_str)));
            }

            // ── Audio builtins (R-25, R-26) ──────────────────────────
            if name == "audio_play" && args.len() == 2 {
                let arr = extract_array_arg("audio_play", &args[0], arrays)?;
                let rate = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                return Ok(Value::Float(crate::audio_io::audio_play_impl(&arr, rate)));
            }
            if name == "audio_play_file" && args.len() == 1 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path_str = path.as_str().map_err(|_| CliError::Compile("audio_play_file() requires a string path".into()))?;
                return Ok(Value::Float(crate::audio_io::audio_play_file_impl(path_str)));
            }
            if name == "audio_stop" && args.is_empty() {
                return Ok(Value::Float(crate::audio_io::audio_stop_impl()));
            }

            // GDI text rendering builtins
            #[cfg(target_os = "windows")]
            {
                if name == "gdi_text_begin" && args.is_empty() {
                    crate::text_render::text_begin();
                    return Ok(Value::Float(0.0));
                }
                if name == "gdi_text_add" && args.len() == 2 {
                    let text_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let text = text_val.as_str()?.to_string();
                    let size = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                        .as_float()? as u32;
                    let idx = crate::text_render::text_add(&text, size);
                    return Ok(Value::Float(idx as f32));
                }
                if name == "gdi_text_w" && args.len() == 1 {
                    let idx = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                        .as_float()? as usize;
                    return Ok(Value::Float(crate::text_render::text_w(idx) as f32));
                }
                if name == "gdi_text_h" && args.len() == 1 {
                    let idx = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                        .as_float()? as usize;
                    return Ok(Value::Float(crate::text_render::text_h(idx) as f32));
                }
                if name == "gdi_text_off" && args.len() == 1 {
                    let idx = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                        .as_float()? as usize;
                    return Ok(Value::Float(crate::text_render::text_off(idx) as f32));
                }
                if name == "gdi_text_width" && args.len() == 2 {
                    let text_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let text = text_val.as_str()?.to_string();
                    let size = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                        .as_float()? as u32;
                    return Ok(Value::Float(crate::text_render::text_width(&text, size) as f32));
                }
                if name == "gdi_text_height" && args.len() == 1 {
                    let size = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                        .as_float()? as u32;
                    return Ok(Value::Float(crate::text_render::text_height(size) as f32));
                }

                // ── CPU metrics ──
                if name == "cpu_util" && args.is_empty() {
                    return Ok(Value::Float(crate::cpu_io::cpu_util()));
                }
                if name == "cpu_count" && args.is_empty() {
                    return Ok(Value::Float(crate::cpu_io::cpu_count()));
                }

                // ── NVML GPU metrics ──
                if name == "nvml_init" && args.is_empty() {
                    return Ok(Value::Float(crate::nvml_io::nvml_init()));
                }
                if name == "nvml_gpu_util" && args.is_empty() {
                    return Ok(Value::Float(crate::nvml_io::nvml_gpu_util()));
                }
                if name == "nvml_mem_util" && args.is_empty() {
                    return Ok(Value::Float(crate::nvml_io::nvml_mem_util()));
                }
                if name == "nvml_temperature" && args.is_empty() {
                    return Ok(Value::Float(crate::nvml_io::nvml_temperature()));
                }
                if name == "nvml_vram_used" && args.is_empty() {
                    return Ok(Value::Float(crate::nvml_io::nvml_vram_used()));
                }
                if name == "nvml_vram_total" && args.is_empty() {
                    return Ok(Value::Float(crate::nvml_io::nvml_vram_total()));
                }
                if name == "nvml_power" && args.is_empty() {
                    return Ok(Value::Float(crate::nvml_io::nvml_power()));
                }
                if name == "nvml_gpu_name" && args.is_empty() {
                    return Ok(Value::Str(crate::nvml_io::nvml_gpu_name()));
                }
                if name == "nvml_clock_gpu" && args.is_empty() {
                    return Ok(Value::Float(crate::nvml_io::nvml_clock_gpu()));
                }
            }

            // read_line() — read one line from stdin (for REPL, interactive programs)
            if name == "read_line" && args.is_empty() {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line)
                    .map_err(|e| CliError::Compile(format!("read_line(): {}", e)))?;
                // Trim trailing newline
                if line.ends_with('\n') { line.pop(); }
                if line.ends_with('\r') { line.pop(); }
                return Ok(Value::Str(line));
            }
            // read_line(prompt) — print prompt then read line
            if name == "read_line" && args.len() == 1 {
                let prompt = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let prompt_str = prompt.as_str().unwrap_or_default();
                use std::io::Write;
                print!("{}", prompt_str);
                std::io::stdout().flush().ok();
                let mut line = String::new();
                std::io::stdin().read_line(&mut line)
                    .map_err(|e| CliError::Compile(format!("read_line(): {}", e)))?;
                if line.ends_with('\n') { line.pop(); }
                if line.ends_with('\r') { line.pop(); }
                return Ok(Value::Str(line));
            }
            // Handle random() — 0 args, returns [0.0, 1.0)
            if name == "random" && args.is_empty() {
                return Ok(Value::Float(next_random(rng)));
            }
            // Handle len() — works on arrays, strings, and maps
            if name == "len" && args.len() == 1 {
                if let ScalarExpr::Ref(ref_name) = &args[0] {
                    // Check GPU arrays first (zero-copy)
                    if let Some(n) = gpu_array_len(ref_name) {
                        return Ok(Value::Float(n as f32));
                    }
                    if let Some(arr) = arrays.get(ref_name) {
                        return Ok(Value::Float(arr.len() as f32));
                    }
                    if let Some(hm) = hashmaps.get(ref_name) {
                        return Ok(Value::Float(hm.len() as f32));
                    }
                }
                // len() on a string or map value
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                if let Value::Str(s) = &val {
                    return Ok(Value::Float(s.chars().count() as f32));
                }
                if let Value::Map(m) = &val {
                    return Ok(Value::Float(m.len() as f32));
                }
                return Err(CliError::Compile("len() requires an array, map, or string argument".into()));
            }
            // Handle pop(arr) — remove and return last element
            if name == "pop" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    gpu_array_materialize(arr_name, arrays);
                    let arr = arrays.get_mut(arr_name)
                        .ok_or_else(|| CliError::Compile(format!("undefined array '{}'", arr_name)))?;
                    return arr.pop()
                        .ok_or_else(|| CliError::Compile(format!("cannot pop from empty array '{}'", arr_name)));
                }
                return Err(CliError::Compile("pop() argument must be an array name".into()));
            }
            // Handle contains(haystack, needle) — string search
            if name == "contains" && args.len() == 2 {
                let haystack = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let needle = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                match (&haystack, &needle) {
                    (Value::Str(h), Value::Str(n)) => {
                        return Ok(Value::Float(if h.contains(n.as_str()) { 1.0 } else { 0.0 }));
                    }
                    _ => return Err(CliError::Compile("contains() requires two string arguments".into())),
                }
            }
            // Handle map_get(map, key) — retrieve value from hashmap
            if name == "map_get" && args.len() == 2 {
                if let ScalarExpr::Ref(map_name) = &args[0] {
                    let key_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_str()?.to_string();
                    let hm = hashmaps.get(map_name)
                        .ok_or_else(|| CliError::Compile(format!("undefined map '{}'", map_name)))?;
                    return hm.get(&key_val)
                        .cloned()
                        .ok_or_else(|| CliError::Compile(format!("key '{}' not found in map '{}'", key_val, map_name)));
                }
                return Err(CliError::Compile("map_get() first argument must be a map name".into()));
            }
            // Handle map_has(map, key) — check if key exists, returns 1.0 or 0.0
            if name == "map_has" && args.len() == 2 {
                if let ScalarExpr::Ref(map_name) = &args[0] {
                    let key_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_str()?.to_string();
                    let hm = hashmaps.get(map_name)
                        .ok_or_else(|| CliError::Compile(format!("undefined map '{}'", map_name)))?;
                    return Ok(Value::Float(if hm.contains_key(&key_val) { 1.0 } else { 0.0 }));
                }
                return Err(CliError::Compile("map_has() first argument must be a map name".into()));
            }
            // Handle map_remove(map, key) — remove key, returns removed value
            if name == "map_remove" && args.len() == 2 {
                if let ScalarExpr::Ref(map_name) = &args[0] {
                    let key_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_str()?.to_string();
                    let hm = hashmaps.get_mut(map_name)
                        .ok_or_else(|| CliError::Compile(format!("undefined map '{}'", map_name)))?;
                    return hm.remove(&key_val)
                        .ok_or_else(|| CliError::Compile(format!("key '{}' not found in map '{}'", key_val, map_name)));
                }
                return Err(CliError::Compile("map_remove() first argument must be a map name".into()));
            }
            // map() — create empty hashmap as inline expression
            if name == "map" && args.is_empty() {
                return Ok(Value::Map(std::collections::HashMap::new()));
            }
            // Handle map_keys(map) — returns comma-separated string of keys
            if name == "map_keys" && args.len() == 1 {
                if let ScalarExpr::Ref(map_name) = &args[0] {
                    let hm = hashmaps.get(map_name)
                        .ok_or_else(|| CliError::Compile(format!("undefined map '{}'", map_name)))?;
                    let mut keys: Vec<&String> = hm.keys().collect();
                    keys.sort();
                    return Ok(Value::Str(keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(",")));
                }
                return Err(CliError::Compile("map_keys() argument must be a map name".into()));
            }
            // Handle json_stringify(name) — serialize map or array to JSON string
            if name == "json_stringify" && args.len() == 1 {
                if let ScalarExpr::Ref(ref_name) = &args[0] {
                    // Check hashmaps first
                    if let Some(hm) = hashmaps.get(ref_name) {
                        let json_str = crate::json_io::stringify_map(hm)?;
                        return Ok(Value::Str(json_str));
                    }
                    // Then check arrays (GPU arrays need materializing)
                    gpu_array_materialize(ref_name, arrays);
                    if let Some(arr) = arrays.get(ref_name) {
                        let json_str = crate::json_io::stringify_array(arr)?;
                        return Ok(Value::Str(json_str));
                    }
                    return Err(CliError::Compile(format!("json_stringify(): '{}' is not a map or array", ref_name)));
                }
                return Err(CliError::Compile("json_stringify() argument must be a variable name".into()));
            }
            // ── File I/O functions (Phase 31) ──────────────────────
            // read_file(path) → string contents
            if (name == "read_file" || name == "read") && args.len() == 1 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("read_file() path must be a string".into()))?;
                check_read_permission_for(path)?;
                let contents = std::fs::read_to_string(path)
                    .map_err(|e| CliError::Io(format!("read_file(\"{}\"): {}", path, e)))?;
                return Ok(Value::Str(contents));
            }
            // read_image(path) → map {r: array, g: array, b: array, width: N, height: N}
            if name == "read_image" && args.len() == 1 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("read_image() path must be a string".into()))?;
                check_read_permission_for(path)?;
                let (pixels, w, h) = crate::image_io::read_image(path)?;
                // pixels is flat RGB: [r0,g0,b0, r1,g1,b1, ...]
                let n = (w as usize) * (h as usize);
                let mut r_arr = Vec::with_capacity(n);
                let mut g_arr = Vec::with_capacity(n);
                let mut b_arr = Vec::with_capacity(n);
                for i in 0..n {
                    r_arr.push(Value::Float(pixels[i * 3]));
                    g_arr.push(Value::Float(pixels[i * 3 + 1]));
                    b_arr.push(Value::Float(pixels[i * 3 + 2]));
                }
                // Store channel arrays in the arrays map with auto-generated names
                let r_name = format!("__img_r_{}", rng.get());
                let g_name = format!("__img_g_{}", rng.get());
                let b_name = format!("__img_b_{}", rng.get());
                arrays.insert(r_name.clone(), r_arr);
                arrays.insert(g_name.clone(), g_arr);
                arrays.insert(b_name.clone(), b_arr);
                let mut result = HashMap::new();
                result.insert("r".to_string(), Value::Str(r_name));
                result.insert("g".to_string(), Value::Str(g_name));
                result.insert("b".to_string(), Value::Str(b_name));
                result.insert("width".to_string(), Value::Float(w as f32));
                result.insert("height".to_string(), Value::Float(h as f32));
                return Ok(Value::Map(result));
            }
            // write_image(path, r_array, g_array, b_array, width, height) → writes PNG
            if name == "write_image" && args.len() == 6 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("write_image() path must be a string".into()))?.to_string();
                check_write_permission_for(&path)?;
                let w_val = eval_scalar(&args[4], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h_val = eval_scalar(&args[5], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let w = w_val.as_float().map_err(|_| CliError::Compile("write_image() width must be a number".into()))? as u32;
                let h = h_val.as_float().map_err(|_| CliError::Compile("write_image() height must be a number".into()))? as u32;
                // Get channel arrays by name
                let r_name = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let g_name = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let b_name = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let r_arr = if let Value::Str(ref s) = r_name {
                    arrays.get(s).ok_or_else(|| CliError::Compile(format!("write_image(): array '{}' not found", s)))?
                } else { return Err(CliError::Compile("write_image() r channel must be an array name".into())); };
                let g_arr = if let Value::Str(ref s) = g_name {
                    arrays.get(s).ok_or_else(|| CliError::Compile(format!("write_image(): array '{}' not found", s)))?
                } else { return Err(CliError::Compile("write_image() g channel must be an array name".into())); };
                let b_arr = if let Value::Str(ref s) = b_name {
                    arrays.get(s).ok_or_else(|| CliError::Compile(format!("write_image(): array '{}' not found", s)))?
                } else { return Err(CliError::Compile("write_image() b channel must be an array name".into())); };
                let n = (w as usize) * (h as usize);
                let mut pixels = Vec::with_capacity(n * 3);
                for i in 0..n {
                    pixels.push(r_arr.get(i).map(|v| v.as_float().unwrap_or(0.0)).unwrap_or(0.0));
                    pixels.push(g_arr.get(i).map(|v| v.as_float().unwrap_or(0.0)).unwrap_or(0.0));
                    pixels.push(b_arr.get(i).map(|v| v.as_float().unwrap_or(0.0)).unwrap_or(0.0));
                }
                crate::image_io::write_image(&path, &pixels, w, h)?;
                return Ok(Value::Float(1.0));
            }
            // file_exists(path) → 1.0 or 0.0
            if name == "file_exists" && args.len() == 1 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("file_exists() path must be a string".into()))?;
                check_read_permission_for(path)?;
                return Ok(Value::Float(if std::path::Path::new(path).exists() { 1.0 } else { 0.0 }));
            }
            // file_size(path) → size in bytes as float
            if name == "file_size" && args.len() == 1 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("file_size() path must be a string".into()))?;
                check_read_permission_for(path)?;
                let meta = std::fs::metadata(path)
                    .map_err(|e| CliError::Io(format!("file_size(\"{}\"): {}", path, e)))?;
                return Ok(Value::Float(meta.len() as f32));
            }
            // gguf_cache_file(path) → loads file into memory cache, returns size in bytes.
            // Subsequent gguf_load_tensor calls use cached bytes instead of re-reading.
            if name == "gguf_cache_file" && args.len() == 1 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("gguf_cache_file() path must be a string".into()))?.to_string();
                check_read_permission_for(&path)?;
                let size = FILE_CACHE.with(|cache| {
                    let mut c = cache.borrow_mut();
                    if let Some(existing) = c.get(&path) {
                        return existing.len();
                    }
                    let bytes = std::fs::read(&path).unwrap_or_default();
                    let len = bytes.len();
                    c.insert(path.clone(), bytes);
                    len
                });
                return Ok(Value::Float(size as f32));
            }
            // gguf_evict_layer(path, model, layer_idx) — evict layer from TENSOR_CACHE + GPU_BUFFER_CACHE
            if name == "gguf_evict_layer" && args.len() == 3 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("gguf_evict_layer: path must be string".into()))?.to_string();
                let model_name = if let ScalarExpr::Ref(name) = &args[1] {
                    name.clone()
                } else {
                    return Err(CliError::Compile("gguf_evict_layer: second arg must be map variable".into()));
                };
                let layer_idx = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;
                let li = layer_idx.to_string();

                // Check if model has bias tensors
                let has_bias = hashmaps.get(&model_name)
                    .map(|m| m.contains_key(&format!("t.blk.{}.attn_q.bias.type", li)))
                    .unwrap_or(false);

                // Evict all weight tensors for this layer
                let tensor_names = vec![
                    format!("blk.{}.attn_norm.weight", li),
                    format!("blk.{}.attn_q.weight", li),
                    format!("blk.{}.attn_k.weight", li),
                    format!("blk.{}.attn_v.weight", li),
                    format!("blk.{}.attn_output.weight", li),
                    format!("blk.{}.ffn_norm.weight", li),
                    format!("blk.{}.ffn_gate.weight", li),
                    format!("blk.{}.ffn_up.weight", li),
                    format!("blk.{}.ffn_down.weight", li),
                ];
                for tn in &tensor_names {
                    evict_tensor(&format!("{}:{}", path, tn));
                }
                if has_bias {
                    evict_tensor(&format!("{}:blk.{}.attn_q.bias", path, li));
                    evict_tensor(&format!("{}:blk.{}.attn_k.bias", path, li));
                    evict_tensor(&format!("{}:blk.{}.attn_v.bias", path, li));
                }
                LAYER_RESIDENCY.with(|lr| lr.borrow_mut().remove(&layer_idx));
                return Ok(Value::Float(1.0));
            }
            // gguf_evict_layer_ram(path, model, layer_idx) — evict layer from TENSOR_CACHE only
            // GPU_BUFFER_CACHE stays resident in VRAM for reuse on subsequent tokens.
            // Use this when weights should stay on GPU but system RAM should be freed.
            if name == "gguf_evict_layer_ram" && args.len() == 3 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("gguf_evict_layer_ram: path must be string".into()))?.to_string();
                let model_name = if let ScalarExpr::Ref(name) = &args[1] {
                    name.clone()
                } else {
                    return Err(CliError::Compile("gguf_evict_layer_ram: second arg must be map variable".into()));
                };
                let layer_idx = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;
                let li = layer_idx.to_string();

                let has_bias = hashmaps.get(&model_name)
                    .map(|m| m.contains_key(&format!("t.blk.{}.attn_q.bias.type", li)))
                    .unwrap_or(false);

                let tensor_names = vec![
                    format!("blk.{}.attn_norm.weight", li),
                    format!("blk.{}.attn_q.weight", li),
                    format!("blk.{}.attn_k.weight", li),
                    format!("blk.{}.attn_v.weight", li),
                    format!("blk.{}.attn_output.weight", li),
                    format!("blk.{}.ffn_norm.weight", li),
                    format!("blk.{}.ffn_gate.weight", li),
                    format!("blk.{}.ffn_up.weight", li),
                    format!("blk.{}.ffn_down.weight", li),
                ];
                for tn in &tensor_names {
                    evict_tensor_cache_only(&format!("{}:{}", path, tn));
                }
                if has_bias {
                    evict_tensor_cache_only(&format!("{}:blk.{}.attn_q.bias", path, li));
                    evict_tensor_cache_only(&format!("{}:blk.{}.attn_k.bias", path, li));
                    evict_tensor_cache_only(&format!("{}:blk.{}.attn_v.bias", path, li));
                }
                return Ok(Value::Float(1.0));
            }
            // gguf_prefetch_layer(path, model, layer_idx) — async prefetch next layer weights
            // Spawns background thread: file I/O + CPU dequant. Main thread continues.
            // gguf_prefetch_complete() joins and uploads to GPU_BUFFER_CACHE/TENSOR_CACHE.
            if name == "gguf_prefetch_layer" && args.len() == 3 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("gguf_prefetch_layer: path must be string".into()))?.to_string();
                check_read_permission_for(&path)?;
                let model_name = if let ScalarExpr::Ref(name) = &args[1] {
                    name.clone()
                } else {
                    return Err(CliError::Compile("gguf_prefetch_layer: second arg must be map variable".into()));
                };
                let layer_idx = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;

                let model_map = hashmaps.get(&model_name).ok_or_else(|| {
                    CliError::Compile(format!("gguf_prefetch_layer: map '{}' not found", model_name))
                })?;

                let li = layer_idx.to_string();
                // Build tensor list: (cache_key, tensor_type, file_offset, total_count, is_small)
                let mut tensor_metas: Vec<(String, u32, u64, usize, bool)> = Vec::new();

                // Large weight matrices (is_small=false)
                let weight_names = vec![
                    format!("blk.{}.attn_q.weight", li),
                    format!("blk.{}.attn_k.weight", li),
                    format!("blk.{}.attn_v.weight", li),
                    format!("blk.{}.attn_output.weight", li),
                    format!("blk.{}.ffn_gate.weight", li),
                    format!("blk.{}.ffn_up.weight", li),
                    format!("blk.{}.ffn_down.weight", li),
                ];
                // Small tensors (is_small=true)
                let small_names = vec![
                    format!("blk.{}.attn_norm.weight", li),
                    format!("blk.{}.ffn_norm.weight", li),
                ];
                // Optional bias tensors (is_small=true)
                let has_bias = model_map.contains_key(&format!("t.blk.{}.attn_q.bias.type", li));
                let bias_names = if has_bias {
                    vec![
                        format!("blk.{}.attn_q.bias", li),
                        format!("blk.{}.attn_k.bias", li),
                        format!("blk.{}.attn_v.bias", li),
                    ]
                } else {
                    vec![]
                };

                // Resolve _ds_buf and _hdr_buf on main thread (mem_table is thread-local)
                let ds_buf = model_map.get("_ds_buf")
                    .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                let hdr_buf = model_map.get("_hdr_buf")
                    .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                let ds_ptr = mem_table_get_ptr(ds_buf)?;
                let data_start = unsafe { (ds_ptr as *const u64).read_unaligned() };
                let hdr_ptr = mem_table_get_ptr(hdr_buf)?;

                // Extract metadata for all tensors
                let all_names: Vec<(&str, bool)> = weight_names.iter().map(|n| (n.as_str(), false))
                    .chain(small_names.iter().map(|n| (n.as_str(), true)))
                    .chain(bias_names.iter().map(|n| (n.as_str(), true)))
                    .collect();

                for (tname, is_small) in &all_names {
                    let prefix = format!("t.{}", tname);
                    let total_count = model_map.get(&format!("{}.count", prefix))
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                    if total_count == 0 { continue; }
                    let tensor_type = model_map.get(&format!("{}.type", prefix))
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as u32;
                    let off_pos = model_map.get(&format!("{}.off_pos", prefix))
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                    let tensor_offset = unsafe { (hdr_ptr.add(off_pos) as *const u64).read_unaligned() };
                    let file_offset = data_start + tensor_offset;
                    let cache_key = format!("{}:{}", path, tname);

                    // Skip if already cached
                    let in_gpu = GPU_BUFFER_CACHE.with(|gc| gc.borrow().contains_key(&cache_key));
                    let in_tc = TENSOR_CACHE.with(|tc| tc.borrow().contains_key(&cache_key));
                    if in_gpu || in_tc { continue; }

                    tensor_metas.push((cache_key, tensor_type, file_offset, total_count, *is_small));
                }

                if tensor_metas.is_empty() {
                    return Ok(Value::Float(1.0)); // All already cached
                }

                let path_clone = path.clone();
                let handle = std::thread::spawn(move || -> Result<Vec<(String, Vec<f32>, bool)>, String> {
                    use std::io::{Read, Seek, SeekFrom};
                    let mut file = std::fs::File::open(&path_clone)
                        .map_err(|e| format!("prefetch open: {}", e))?;
                    let mut results = Vec::with_capacity(tensor_metas.len());

                    for (cache_key, tensor_type, file_offset, total_count, is_small) in tensor_metas {
                        let byte_size = match tensor_type {
                            0 => total_count * 4,
                            1 => total_count * 2,
                            12 => (total_count / 256) * 144,
                            13 => (total_count / 256) * 176,
                            14 => (total_count / 256) * 210,
                            _ => total_count * 4,
                        };
                        file.seek(SeekFrom::Start(file_offset))
                            .map_err(|e| format!("prefetch seek: {}", e))?;
                        let mut raw = vec![0u8; byte_size];
                        file.read_exact(&mut raw)
                            .map_err(|e| format!("prefetch read: {}", e))?;

                        // CPU dequant (no GPU access from background thread)
                        let dequanted: Vec<f32> = match tensor_type {
                            0 => raw.chunks_exact(4).take(total_count)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                            1 => raw.chunks_exact(2).take(total_count)
                                .map(|c| gguf_f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                            12 => {
                                let mut out = Vec::with_capacity(total_count + 256);
                                for block in raw.chunks(144) {
                                    if block.len() < 144 { break; }
                                    gguf_dequant_q4k_block(block, &mut out);
                                }
                                out.truncate(total_count);
                                out
                            }
                            13 => {
                                let mut out = Vec::with_capacity(total_count + 256);
                                for block in raw.chunks(176) {
                                    if block.len() < 176 { break; }
                                    gguf_dequant_q5k_block(block, &mut out);
                                }
                                out.truncate(total_count);
                                out
                            }
                            14 => {
                                let mut out = Vec::with_capacity(total_count + 256);
                                for block in raw.chunks(210) {
                                    if block.len() < 210 { break; }
                                    gguf_dequant_q6k_block(block, &mut out);
                                }
                                out.truncate(total_count);
                                out
                            }
                            _ => return Err(format!("prefetch: unsupported type {}", tensor_type)),
                        };
                        results.push((cache_key, dequanted, is_small));
                    }
                    Ok(results)
                });

                PREFETCH_THREAD.with(|pt| *pt.borrow_mut() = Some(handle));
                return Ok(Value::Float(1.0));
            }
            // vm_gpu_usage() — returns total bytes currently in GPU_BUFFER_CACHE
            if name == "vm_gpu_usage" && args.is_empty() {
                let bytes = GPU_CACHE_BYTES.with(|c| c.get());
                return Ok(Value::Float(bytes as f64 as f32));
            }
            // vm_layer_resident(layer_idx) — returns 1.0 if layer is GPU-resident, 0.0 otherwise
            if name == "vm_layer_resident" && args.len() == 1 {
                let lidx = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;
                let resident = LAYER_RESIDENCY.with(|lr| {
                    lr.borrow().get(&lidx).copied().unwrap_or(false)
                });
                return Ok(Value::Float(if resident { 1.0 } else { 0.0 }));
            }
            // vm_layer_estimate(model, layer_idx) — estimated dequanted f32 bytes for one layer
            if name == "vm_layer_estimate" && args.len() == 2 {
                let model_name = if let ScalarExpr::Ref(name) = &args[0] {
                    name.clone()
                } else {
                    return Err(CliError::Compile("vm_layer_estimate: first arg must be map variable".into()));
                };
                let lidx = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;
                let model_map = hashmaps.get(&model_name).ok_or_else(|| {
                    CliError::Compile(format!("vm_layer_estimate: map '{}' not found", model_name))
                })?;
                let li = lidx.to_string();
                let large_names = [
                    format!("blk.{}.attn_q.weight", li),
                    format!("blk.{}.attn_k.weight", li),
                    format!("blk.{}.attn_v.weight", li),
                    format!("blk.{}.attn_output.weight", li),
                    format!("blk.{}.ffn_gate.weight", li),
                    format!("blk.{}.ffn_up.weight", li),
                    format!("blk.{}.ffn_down.weight", li),
                ];
                let mut total_bytes: u64 = 0;
                for tname in &large_names {
                    let count = model_map.get(&format!("t.{}.count", tname))
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as u64;
                    total_bytes += count * 4; // f32 = 4 bytes per element
                }
                return Ok(Value::Float(total_bytes as f64 as f32));
            }

            // ── Loom Engine builtins (loom_* primary, vm_* deprecated aliases) ──
            // loom_boot / vm_boot(n_instances, reg_size, globals_size) → unit handle ID
            if (name == "loom_boot" || name == "vm_boot") && args.len() == 3 {
                let n_instances = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let reg_size = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let globals_size = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;

                // Resource budget: check VM cap
                let total_vms = GPU_VMS.with(|v| v.borrow().len()) + PARKED_VMS.with(|p| p.borrow().len());
                let max_vms = LOOM_MAX_VMS.with(|m| m.get());
                if max_vms > 0 && total_vms >= max_vms as usize {
                    return Ok(Value::Float(-1.0)); // VM cap exceeded
                }

                // Resource budget: check VRAM
                let vram_cost = (reg_size as u64 * 4 * 32 * n_instances as u64) + (globals_size as u64 * 4);
                let budget = LOOM_VRAM_BUDGET.with(|b| b.get());
                if budget > 0 {
                    let used = LOOM_VRAM_USED.with(|u| u.get());
                    if used + vram_cost > budget {
                        return Ok(Value::Float(-2.0)); // VRAM budget exceeded
                    }
                }

                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Ok(Value::Float(-1.0)); // no GPU — graceful fallback
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                match octoflow_vulkan::vm::vm_create(gpu_dev, n_instances, reg_size, globals_size) {
                    Ok(vm) => {
                        let vm_id = VM_NEXT_ID.with(|c| { let id = c.get(); c.set(id + 1); id });
                        GPU_VMS.with(|vms| vms.borrow_mut().insert(vm_id, vm));
                        LOOM_VRAM_USED.with(|u| u.set(u.get() + vram_cost));
                        return Ok(Value::Float(vm_id as f32));
                    }
                    Err(_) => return Ok(Value::Float(-1.0)), // vm_create failed — graceful fallback
                }
            }
            // vm_write_register(vm_id, instance, reg_idx, array_name) → 0.0
            if name == "vm_write_register" && args.len() == 4 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let instance = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let reg_idx = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let arr_name = if let ScalarExpr::Ref(n) = &args[3] { n.clone() }
                    else { return Err(CliError::Compile("vm_write_register: 4th arg must be array name".into())); };
                let data = gpu_array_get(&arr_name).unwrap_or_else(|| {
                    arrays.get(&arr_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("vm_write_register: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_write_register: unknown VM {}", vm_id)))?;
                    octoflow_vulkan::vm::vm_write_reg(gpu_dev, vm, instance, reg_idx, &data)
                        .map_err(|e| CliError::Runtime(format!("vm_write_register: {}", e)))
                })?;
                return Ok(Value::Float(0.0));
            }
            // vm_write_globals(vm_id, offset, array_name) → 0.0
            if (name == "loom_write" || name == "loom_set_globals" || name == "vm_write_globals") && args.len() == 3 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let arr_name = if let ScalarExpr::Ref(n) = &args[2] { n.clone() }
                    else { return Err(CliError::Compile("vm_write_globals: 3rd arg must be array name".into())); };
                let data = gpu_array_get(&arr_name).unwrap_or_else(|| {
                    arrays.get(&arr_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("vm_write_globals: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                GPU_VMS.with(|vms| -> Result<(), CliError> {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_write_globals: unknown VM {}", vm_id)))?;
                    // Use deferred staging: CPU memcpy only, DMA batched at loom_build.
                    let pending = octoflow_vulkan::vm::vm_stage_write(gpu_dev, &vm.globals, (offset as u64) * 4, &data)
                        .map_err(|e| CliError::Runtime(format!("vm_write_globals: {}", e)))?;
                    if let Some(pu) = pending {
                        VM_PENDING_UPLOADS.with(|uploads| {
                            uploads.borrow_mut().entry(vm_id).or_default().push(pu);
                        });
                    }
                    Ok(())
                })?;
                return Ok(Value::Float(0.0));
            }
            // loom_copy(src_vm, src_offset, dst_vm, dst_offset, count) → 0.0
            // GPU→GPU buffer copy between VMs' globals (no CPU roundtrip).
            if name == "loom_copy" && args.len() == 5 {
                let src_vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let src_offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let dst_vm_id = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let dst_offset = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let count = eval_scalar(&args[4], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("loom_copy: no Vulkan GPU available".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let src_vm = vms.get(&src_vm_id).ok_or_else(||
                        CliError::Runtime(format!("loom_copy: unknown src VM {}", src_vm_id)))?;
                    let dst_vm = vms.get(&dst_vm_id).ok_or_else(||
                        CliError::Runtime(format!("loom_copy: unknown dst VM {}", dst_vm_id)))?;
                    octoflow_vulkan::vm::vm_copy_globals(gpu_dev, src_vm, src_offset, dst_vm, dst_offset, count)
                        .map_err(|e| CliError::Runtime(format!("loom_copy: {}", e)))
                })?;
                return Ok(Value::Float(0.0));
            }

            // ── Mailbox builtins (cross-loom communication) ──

            // loom_mailbox(slot_size, slot_count) → vm_id
            // Creates a mailbox VM with a ring buffer layout in its globals buffer.
            if name == "loom_mailbox" && args.len() == 2 {
                let slot_size = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let slot_count = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                // F02: validate non-zero dimensions
                if slot_size == 0 || slot_count == 0 {
                    return Ok(Value::Float(-1.0)); // error: invalid mailbox dimensions
                }
                let header = 6u32;
                let globals_size = header + slot_size * slot_count;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("loom_mailbox: no Vulkan GPU available".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                let vm = octoflow_vulkan::vm::vm_create(gpu_dev, 1, 1, globals_size)
                    .map_err(|e| CliError::Runtime(format!("loom_mailbox: {}", e)))?;
                let vm_id = VM_NEXT_ID.with(|c| { let id = c.get(); c.set(id + 1); id });
                // Initialize header: [seq_write, seq_read, payload_len, flags, slot_size, capacity]
                let header_data: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, slot_size as f32, (slot_size * slot_count) as f32];
                octoflow_vulkan::vm::vm_write_globals(gpu_dev, &vm, 0, &header_data)
                    .map_err(|e| CliError::Runtime(format!("loom_mailbox: init header: {}", e)))?;
                GPU_VMS.with(|vms| vms.borrow_mut().insert(vm_id, vm));
                MAILBOX_VMS.with(|m| m.borrow_mut().insert(vm_id));
                return Ok(Value::Float(vm_id as f32));
            }

            // loom_mail_send(mailbox_vm, src_vm, src_offset, count) → status (1.0=ok, 0.0=full)
            if name == "loom_mail_send" && args.len() == 4 {
                let mb_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let src_id = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let src_off = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let count = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("loom_mail_send: no Vulkan GPU available".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                // Read mailbox header (6 floats)
                let header = GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let mb = vms.get(&mb_id).ok_or_else(||
                        CliError::Runtime(format!("loom_mail_send: unknown mailbox VM {}", mb_id)))?;
                    octoflow_vulkan::vm::vm_read_globals(gpu_dev, mb, 0, 6)
                        .map_err(|e| CliError::Runtime(format!("loom_mail_send: read header: {}", e)))
                })?;
                let seq_w = header[0] as u32;
                let seq_r = header[1] as u32;
                let slot_size = header[4] as u32;
                let capacity = header[5] as u32;
                // F01: reject payload that exceeds slot capacity
                if count > slot_size {
                    return Ok(Value::Float(-1.0)); // error: payload exceeds slot_size
                }
                let slot_count = if slot_size > 0 { capacity / slot_size } else { 0 };
                // Check overflow (ring buffer full)
                if slot_count == 0 || seq_w.wrapping_sub(seq_r) >= slot_count {
                    return Ok(Value::Float(0.0)); // backpressure
                }
                // Compute write position in ring
                let write_pos = 6 + (seq_w % slot_count) * slot_size;
                // Copy payload: src_vm → mailbox
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let src_vm = vms.get(&src_id).ok_or_else(||
                        CliError::Runtime(format!("loom_mail_send: unknown src VM {}", src_id)))?;
                    let mb = vms.get(&mb_id).ok_or_else(||
                        CliError::Runtime(format!("loom_mail_send: unknown mailbox VM {}", mb_id)))?;
                    octoflow_vulkan::vm::vm_copy_globals(gpu_dev, src_vm, src_off, mb, write_pos, count)
                        .map_err(|e| CliError::Runtime(format!("loom_mail_send: copy: {}", e)))
                })?;
                // Update header: seq_write++, payload_len=count, flags=READY
                let new_header: Vec<f32> = vec![(seq_w + 1) as f32, seq_r as f32, count as f32, 1.0, slot_size as f32, capacity as f32];
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let mb = vms.get(&mb_id).ok_or_else(||
                        CliError::Runtime(format!("loom_mail_send: unknown mailbox VM {}", mb_id)))?;
                    octoflow_vulkan::vm::vm_write_globals(gpu_dev, mb, 0, &new_header)
                        .map_err(|e| CliError::Runtime(format!("loom_mail_send: write header: {}", e)))
                })?;
                return Ok(Value::Float(1.0));
            }

            // loom_mail_recv(mailbox_vm, dst_vm, dst_offset) → count (0.0 = empty)
            if name == "loom_mail_recv" && args.len() == 3 {
                let mb_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let dst_id = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let dst_off = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("loom_mail_recv: no Vulkan GPU available".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                // Read mailbox header
                let header = GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let mb = vms.get(&mb_id).ok_or_else(||
                        CliError::Runtime(format!("loom_mail_recv: unknown mailbox VM {}", mb_id)))?;
                    octoflow_vulkan::vm::vm_read_globals(gpu_dev, mb, 0, 6)
                        .map_err(|e| CliError::Runtime(format!("loom_mail_recv: read header: {}", e)))
                })?;
                let seq_w = header[0] as u32;
                let seq_r = header[1] as u32;
                let payload_len = header[2] as u32;
                let slot_size = header[4] as u32;
                let capacity = header[5] as u32;
                let slot_count = if slot_size > 0 { capacity / slot_size } else { 0 };
                // Check empty
                if seq_w == seq_r {
                    return Ok(Value::Float(0.0));
                }
                // Compute read position
                let read_pos = 6 + (seq_r % slot_count) * slot_size;
                // Copy: mailbox[read_pos] → dst_vm[dst_offset]
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let mb = vms.get(&mb_id).ok_or_else(||
                        CliError::Runtime(format!("loom_mail_recv: unknown mailbox VM {}", mb_id)))?;
                    let dst_vm = vms.get(&dst_id).ok_or_else(||
                        CliError::Runtime(format!("loom_mail_recv: unknown dst VM {}", dst_id)))?;
                    octoflow_vulkan::vm::vm_copy_globals(gpu_dev, mb, read_pos, dst_vm, dst_off, payload_len)
                        .map_err(|e| CliError::Runtime(format!("loom_mail_recv: copy: {}", e)))
                })?;
                // Update header: seq_read++
                let new_header: Vec<f32> = vec![seq_w as f32, (seq_r + 1) as f32, payload_len as f32, header[3], slot_size as f32, capacity as f32];
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let mb = vms.get(&mb_id).ok_or_else(||
                        CliError::Runtime(format!("loom_mail_recv: unknown mailbox VM {}", mb_id)))?;
                    octoflow_vulkan::vm::vm_write_globals(gpu_dev, mb, 0, &new_header)
                        .map_err(|e| CliError::Runtime(format!("loom_mail_recv: write header: {}", e)))
                })?;
                return Ok(Value::Float(payload_len as f32));
            }

            // loom_mail_poll(mailbox_vm) → has_message (1.0 or 0.0)
            if name == "loom_mail_poll" && args.len() == 1 {
                let mb_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("loom_mail_poll: no Vulkan GPU available".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                let header = GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let mb = vms.get(&mb_id).ok_or_else(||
                        CliError::Runtime(format!("loom_mail_poll: unknown mailbox VM {}", mb_id)))?;
                    octoflow_vulkan::vm::vm_read_globals(gpu_dev, mb, 0, 2)
                        .map_err(|e| CliError::Runtime(format!("loom_mail_poll: {}", e)))
                })?;
                let has_msg = if header[0] as u32 > header[1] as u32 { 1.0 } else { 0.0 };
                return Ok(Value::Float(has_msg));
            }

            // loom_mail_depth(mailbox_vm) → pending message count
            if name == "loom_mail_depth" && args.len() == 1 {
                let mb_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("loom_mail_depth: no Vulkan GPU available".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                let header = GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let mb = vms.get(&mb_id).ok_or_else(||
                        CliError::Runtime(format!("loom_mail_depth: unknown mailbox VM {}", mb_id)))?;
                    octoflow_vulkan::vm::vm_read_globals(gpu_dev, mb, 0, 2)
                        .map_err(|e| CliError::Runtime(format!("loom_mail_depth: {}", e)))
                })?;
                let depth = (header[0] as u32).wrapping_sub(header[1] as u32);
                return Ok(Value::Float(depth as f32));
            }

            // ── VM Park/Unpark reuse pool ──

            // loom_park(vm_id) → 0.0 — move VM from active to parked pool
            if name == "loom_park" && args.len() == 1 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let vm = GPU_VMS.with(|vms| vms.borrow_mut().remove(&vm_id));
                if let Some(vm) = vm {
                    PARKED_VMS.with(|p| p.borrow_mut().insert(vm_id, vm));
                }
                return Ok(Value::Float(0.0));
            }

            // loom_unpark(vm_id) → 0.0 — move VM from parked pool back to active
            if name == "loom_unpark" && args.len() == 1 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let vm = PARKED_VMS.with(|p| p.borrow_mut().remove(&vm_id));
                if let Some(vm) = vm {
                    GPU_VMS.with(|vms| vms.borrow_mut().insert(vm_id, vm));
                }
                return Ok(Value::Float(0.0));
            }

            // loom_pool_size() → count of parked VMs
            if name == "loom_pool_size" && args.is_empty() {
                let count = PARKED_VMS.with(|p| p.borrow().len());
                return Ok(Value::Float(count as f32));
            }

            // loom_auto_spawn(instances, reg_size, globals_size) → vm_id
            // Reuses a compatible parked VM if available, otherwise boots a new one.
            if name == "loom_auto_spawn" && args.len() == 3 {
                let n_instances = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let reg_size = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let globals_size = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;

                // Search parked pool for a compatible VM (reg_size >= requested, globals >= requested)
                let reuse_id = PARKED_VMS.with(|p| {
                    let parked = p.borrow();
                    for (&vid, vm) in parked.iter() {
                        if vm.n_instances >= n_instances
                            && vm.reg_size >= reg_size
                            && vm.globals.len() >= globals_size as usize
                        {
                            return Some(vid);
                        }
                    }
                    None
                });

                if let Some(vid) = reuse_id {
                    // Unpark the compatible VM
                    let vm = PARKED_VMS.with(|p| p.borrow_mut().remove(&vid)).unwrap();
                    GPU_VMS.with(|vms| vms.borrow_mut().insert(vid, vm));
                    return Ok(Value::Float(vid as f32));
                }

                // No compatible VM in pool — boot a new one

                // F04: check VM cap before booting
                let total_vms = GPU_VMS.with(|v| v.borrow().len()) + PARKED_VMS.with(|p| p.borrow().len());
                let max_vms = LOOM_MAX_VMS.with(|m| m.get());
                if max_vms > 0 && total_vms >= max_vms as usize {
                    return Ok(Value::Float(-1.0)); // VM cap exceeded
                }

                // F05: check VRAM budget before booting
                let vram_cost = (reg_size as u64 * 4 * 32 * n_instances as u64) + (globals_size as u64 * 4);
                let budget = LOOM_VRAM_BUDGET.with(|b| b.get());
                if budget > 0 {
                    let used = LOOM_VRAM_USED.with(|u| u.get());
                    if used + vram_cost > budget {
                        return Ok(Value::Float(-2.0)); // VRAM budget exceeded
                    }
                }

                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Ok(Value::Float(-1.0)); // no GPU — graceful fallback
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                match octoflow_vulkan::vm::vm_create(gpu_dev, n_instances, reg_size, globals_size) {
                    Ok(vm) => {
                        let vm_id = VM_NEXT_ID.with(|c| { let id = c.get(); c.set(id + 1); id });
                        GPU_VMS.with(|vms| vms.borrow_mut().insert(vm_id, vm));
                        LOOM_VRAM_USED.with(|u| u.set(u.get() + vram_cost));
                        return Ok(Value::Float(vm_id as f32));
                    }
                    Err(_) => return Ok(Value::Float(-1.0)), // vm_create failed — graceful fallback
                }
            }

            // loom_auto_release(vm_id) → 0.0
            // Parks the VM instead of destroying it. Destroys oldest parked if pool > 8.
            if name == "loom_auto_release" && args.len() == 1 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let vm = GPU_VMS.with(|vms| vms.borrow_mut().remove(&vm_id));
                if let Some(vm) = vm {
                    PARKED_VMS.with(|p| {
                        let mut parked = p.borrow_mut();
                        // If pool is at capacity (8), remove the oldest entry and decrement VRAM
                        if parked.len() >= 8 {
                            if let Some(&oldest_id) = parked.keys().next() {
                                if let Some(old_vm) = parked.remove(&oldest_id) {
                                    // F03: decrement VRAM for destroyed overflow VM
                                    let cost = (old_vm.reg_size as u64 * 4 * 32 * old_vm.n_instances as u64)
                                        + (old_vm.globals.len() as u64 * 4);
                                    LOOM_VRAM_USED.with(|u| u.set(u.get().saturating_sub(cost)));
                                }
                            }
                        }
                        parked.insert(vm_id, vm);
                    });
                }
                return Ok(Value::Float(0.0));
            }

            // loom_pool_warm(count, instances, reg_size, globals_size) → 0.0
            // Pre-boot `count` VMs and park them immediately for rapid swarm cycling.
            if name == "loom_pool_warm" && args.len() == 4 {
                let count = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let n_instances = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let reg_size = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let globals_size = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;

                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Ok(Value::Float(0.0)); // no GPU — skip warm-up silently
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };

                for _ in 0..count {
                    // Check VM cap
                    let total_vms = GPU_VMS.with(|v| v.borrow().len())
                        + PARKED_VMS.with(|p| p.borrow().len());
                    let max_vms = LOOM_MAX_VMS.with(|m| m.get());
                    if max_vms > 0 && total_vms >= max_vms as usize {
                        break; // Respect budget — warm as many as allowed
                    }

                    // Check VRAM budget
                    let vram_cost = (reg_size as u64 * 4 * 32 * n_instances as u64) + (globals_size as u64 * 4);
                    let budget = LOOM_VRAM_BUDGET.with(|b| b.get());
                    if budget > 0 {
                        let used = LOOM_VRAM_USED.with(|u| u.get());
                        if used + vram_cost > budget {
                            break; // VRAM budget exceeded
                        }
                    }

                    let vm = match octoflow_vulkan::vm::vm_create(gpu_dev, n_instances, reg_size, globals_size) {
                        Ok(vm) => vm,
                        Err(_) => break, // vm_create failed — stop warming
                    };
                    let vm_id = VM_NEXT_ID.with(|c| { let id = c.get(); c.set(id + 1); id });

                    // Track VRAM
                    LOOM_VRAM_USED.with(|u| u.set(u.get() + vram_cost));

                    // Park immediately
                    PARKED_VMS.with(|p| p.borrow_mut().insert(vm_id, vm));
                }

                return Ok(Value::Float(0.0));
            }

            // ── Resource budget controls ──

            // loom_max_vms(n) → 0.0 — set max active + parked VMs (0 = unlimited)
            if name == "loom_max_vms" && args.len() == 1 {
                let n = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                LOOM_MAX_VMS.with(|m| m.set(n));
                return Ok(Value::Float(0.0));
            }

            // loom_vram_budget(bytes) → 0.0 — set VRAM budget in bytes (0 = unlimited)
            if name == "loom_vram_budget" && args.len() == 1 {
                let bytes = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u64;
                LOOM_VRAM_BUDGET.with(|b| b.set(bytes));
                return Ok(Value::Float(0.0));
            }

            // loom_vram_used() → bytes — current estimated VRAM usage
            if name == "loom_vram_used" && args.is_empty() {
                let used = LOOM_VRAM_USED.with(|u| u.get());
                return Ok(Value::Float(used as f32));
            }

            // loom_vm_count() → count — total active + parked VMs
            if name == "loom_vm_count" && args.is_empty() {
                let active = GPU_VMS.with(|v| v.borrow().len());
                let parked = PARKED_VMS.with(|p| p.borrow().len());
                return Ok(Value::Float((active + parked) as f32));
            }

            // ── VM Introspection ──

            // loom_vm_info(vm_id) → map with reg_size, globals_size, instances, parked
            if name == "loom_vm_info" && args.len() == 1 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;

                // Check active VMs first
                let info = GPU_VMS.with(|vms| {
                    vms.borrow().get(&vm_id).map(|vm| {
                        (vm.reg_size as f32, vm.globals.len() as f32, vm.n_instances as f32, 0.0f32)
                    })
                });

                // Check parked VMs if not active
                let info = info.or_else(|| {
                    PARKED_VMS.with(|pvms| {
                        pvms.borrow().get(&vm_id).map(|vm| {
                            (vm.reg_size as f32, vm.globals.len() as f32, vm.n_instances as f32, 1.0f32)
                        })
                    })
                });

                match info {
                    Some((reg, glob, inst, parked)) => {
                        let mut map = HashMap::new();
                        map.insert("reg_size".to_string(), Value::Float(reg));
                        map.insert("globals_size".to_string(), Value::Float(glob));
                        map.insert("instances".to_string(), Value::Float(inst));
                        map.insert("parked".to_string(), Value::Float(parked));
                        return Ok(Value::Map(map));
                    }
                    None => return Ok(Value::Float(-1.0)), // VM not found
                }
            }

            // loom_elapsed_us() → microseconds since program start
            if name == "loom_elapsed_us" && args.is_empty() {
                use std::time::Instant;
                use std::sync::OnceLock;
                static START_ELAPSED: OnceLock<Instant> = OnceLock::new();
                let start = START_ELAPSED.get_or_init(Instant::now);
                let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;
                return Ok(Value::Float(elapsed_us as f32));
            }

            // loom_dispatch_time() → microseconds of last dispatch
            if name == "loom_dispatch_time" && args.is_empty() {
                let us = LAST_DISPATCH_US.with(|t| t.get());
                return Ok(Value::Float(us as f32));
            }

            // loom_pool_info() → map with active/parked/total/vram_used/vram_budget
            if name == "loom_pool_info" && args.is_empty() {
                let active = GPU_VMS.with(|v| v.borrow().len()) as f32;
                let parked = PARKED_VMS.with(|p| p.borrow().len()) as f32;
                let vram_used = LOOM_VRAM_USED.with(|u| u.get()) as f32;
                let vram_budget = LOOM_VRAM_BUDGET.with(|b| b.get()) as f32;
                let mut map = HashMap::new();
                map.insert("active".to_string(), Value::Float(active));
                map.insert("parked".to_string(), Value::Float(parked));
                map.insert("total".to_string(), Value::Float(active + parked));
                map.insert("vram_used".to_string(), Value::Float(vram_used));
                map.insert("vram_budget".to_string(), Value::Float(vram_budget));
                return Ok(Value::Map(map));
            }

            // ── OctoPress GPU Compression Foundation ──

            // octopress_init(block_size) → 0.0 (block_size must be power of 2)
            if name == "octopress_init" && args.len() == 1 {
                let block_size = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                if block_size == 0 || (block_size & (block_size - 1)) != 0 {
                    return Ok(Value::Float(-1.0)); // must be power of 2
                }
                OCTOPRESS_STATE.with(|s| {
                    *s.borrow_mut() = Some(OctoPressState { block_size });
                });
                return Ok(Value::Float(0.0));
            }

            // octopress_analyze(data_array_name) → map with statistics
            if name == "octopress_analyze" && args.len() == 1 {
                let arr_name = if let ScalarExpr::Ref(n) = &args[0] { n.clone() }
                    else {
                        eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                            .as_str().map_err(|_| CliError::Compile("octopress_analyze: arg must be array name".into()))?.to_string()
                    };
                let data: Vec<f32> = arrays.get(&arr_name)
                    .map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect())
                    .unwrap_or_default();

                if data.is_empty() {
                    let mut map = HashMap::new();
                    map.insert("mean".to_string(), Value::Float(0.0));
                    map.insert("variance".to_string(), Value::Float(0.0));
                    map.insert("self_similarity".to_string(), Value::Float(0.0));
                    map.insert("delta_ratio".to_string(), Value::Float(0.0));
                    map.insert("recommended_method".to_string(), Value::Float(0.0));
                    return Ok(Value::Map(map));
                }

                let n = data.len() as f32;
                let mean = data.iter().sum::<f32>() / n;
                let variance = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;

                // Self-similarity: correlation between first half and second half
                let half = data.len() / 2;
                let self_similarity = if half > 1 {
                    let a = &data[..half];
                    let b = &data[half..half * 2];
                    let mean_a = a.iter().sum::<f32>() / half as f32;
                    let mean_b = b.iter().sum::<f32>() / half as f32;
                    let mut cov = 0.0f32;
                    let mut var_a = 0.0f32;
                    let mut var_b = 0.0f32;
                    for i in 0..half {
                        let da = a[i] - mean_a;
                        let db = b[i] - mean_b;
                        cov += da * db;
                        var_a += da * da;
                        var_b += db * db;
                    }
                    let denom = (var_a * var_b).sqrt();
                    if denom > 1e-12 { (cov / denom).clamp(-1.0, 1.0) } else { 0.0 }
                } else {
                    0.0
                };

                // Delta ratio: fraction of deltas smaller than sqrt(variance)
                let threshold = variance.sqrt().max(1e-6);
                let delta_count = data.windows(2)
                    .filter(|w| (w[1] - w[0]).abs() < threshold)
                    .count();
                let delta_ratio = if data.len() > 1 { delta_count as f32 / (data.len() - 1) as f32 } else { 0.0 };

                let recommended = if delta_ratio > 0.5 { 1.0 } else { 0.0 };

                let mut map = HashMap::new();
                map.insert("mean".to_string(), Value::Float(mean));
                map.insert("variance".to_string(), Value::Float(variance));
                map.insert("self_similarity".to_string(), Value::Float(self_similarity));
                map.insert("delta_ratio".to_string(), Value::Float(delta_ratio));
                map.insert("recommended_method".to_string(), Value::Float(recommended));
                return Ok(Value::Map(map));
            }

            // octopress_encode(data_array_name, method) → count (compressed stored via RETURNED_ARRAY)
            // method 0 = raw, method 1 = delta
            if name == "octopress_encode" && args.len() == 2 {
                let arr_name = if let ScalarExpr::Ref(n) = &args[0] { n.clone() }
                    else {
                        eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                            .as_str().map_err(|_| CliError::Compile("octopress_encode: 1st arg must be array name".into()))?.to_string()
                    };
                let method = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;

                let data: Vec<f32> = arrays.get(&arr_name)
                    .map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect())
                    .unwrap_or_default();

                // Sanitize: replace NaN/Inf (audit N5)
                let data: Vec<f32> = data.into_iter().map(|v| {
                    if v.is_nan() { 0.0 }
                    else if v.is_infinite() { if v > 0.0 { f32::MAX } else { f32::MIN } }
                    else { v }
                }).collect();

                let compressed: Vec<f32> = match method {
                    0 => {
                        // Raw: header [method=0, count] + data
                        let mut out = vec![0.0, data.len() as f32];
                        out.extend_from_slice(&data);
                        out
                    }
                    1 => {
                        // Delta: header [method=1, count, first_value] + deltas
                        if data.is_empty() {
                            vec![1.0, 0.0, 0.0]
                        } else {
                            let mut out = vec![1.0, data.len() as f32, data[0]];
                            for i in 0..data.len() - 1 {
                                out.push(data[i + 1] - data[i]);
                            }
                            out
                        }
                    }
                    2 => {
                        // Fractal: IFS attractor fitting (CPU path)
                        let block_size = OCTOPRESS_STATE.with(|s| {
                            s.borrow().as_ref().map(|st| st.block_size).unwrap_or(256)
                        }) as usize;

                        if data.len() < block_size * 2 {
                            // Too small for fractal — fall back to delta
                            if data.is_empty() {
                                vec![1.0, 0.0, 0.0]
                            } else {
                                let mut out = vec![1.0, data.len() as f32, data[0]];
                                for i in 0..data.len() - 1 {
                                    out.push(data[i + 1] - data[i]);
                                }
                                out
                            }
                        } else {
                            let range_size = block_size / 2;
                            let n_ranges = data.len() / range_size;
                            let n_domains = data.len() / block_size;

                            // Header: [method=2, original_count, block_size, n_ranges]
                            let mut out = vec![2.0, data.len() as f32, block_size as f32, n_ranges as f32];

                            for r in 0..n_ranges {
                                let r_start = r * range_size;
                                let r_end = (r_start + range_size).min(data.len());
                                let range_block = &data[r_start..r_end];

                                let mut best_domain = 0u32;
                                let mut best_scale = 1.0f32;
                                let mut best_offset = 0.0f32;
                                let mut best_mse = f32::MAX;

                                for d in 0..n_domains {
                                    let d_start = d * block_size;
                                    let d_end = (d_start + block_size).min(data.len());
                                    let domain_block = &data[d_start..d_end];

                                    // Downsample domain block to range_size by averaging pairs
                                    let downsampled: Vec<f32> = domain_block.chunks(2)
                                        .map(|c| if c.len() == 2 { (c[0] + c[1]) / 2.0 } else { c[0] })
                                        .collect();

                                    // Least-squares fit: range ≈ scale * downsampled + offset
                                    let n = range_block.len().min(downsampled.len()) as f32;
                                    let sum_d: f32 = downsampled.iter().take(range_block.len()).sum();
                                    let sum_r: f32 = range_block.iter().sum();
                                    let sum_dd: f32 = downsampled.iter().take(range_block.len())
                                        .map(|x| x * x).sum();
                                    let sum_dr: f32 = downsampled.iter().zip(range_block.iter())
                                        .map(|(d, r)| d * r).sum();

                                    let denom = n * sum_dd - sum_d * sum_d;
                                    let (scale, offset) = if denom.abs() > 1e-12 {
                                        let s = (n * sum_dr - sum_d * sum_r) / denom;
                                        let o = (sum_r - s * sum_d) / n;
                                        (s, o)
                                    } else {
                                        (0.0, sum_r / n)
                                    };

                                    // Compute MSE
                                    let mse: f32 = downsampled.iter().zip(range_block.iter())
                                        .map(|(d, r)| {
                                            let pred = scale * d + offset;
                                            (pred - r) * (pred - r)
                                        })
                                        .sum::<f32>() / n;

                                    if mse < best_mse {
                                        best_mse = mse;
                                        best_domain = d as u32;
                                        best_scale = scale;
                                        best_offset = offset;
                                    }
                                }

                                out.push(best_domain as f32);
                                out.push(best_scale);
                                out.push(best_offset);
                            }
                            out
                        }
                    }
                    _ => return Ok(Value::Float(-1.0)), // unsupported method
                };

                let count = compressed.len();
                RETURNED_ARRAY.with(|r| {
                    *r.borrow_mut() = Some(compressed.into_iter().map(|f| Value::Float(f)).collect());
                });
                return Ok(Value::Float(count as f32));
            }

            // octopress_decode(compressed_array_name) → count (decoded stored via RETURNED_ARRAY)
            if name == "octopress_decode" && args.len() == 1 {
                let arr_name = if let ScalarExpr::Ref(n) = &args[0] { n.clone() }
                    else {
                        eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                            .as_str().map_err(|_| CliError::Compile("octopress_decode: arg must be array name".into()))?.to_string()
                    };

                let compressed: Vec<f32> = arrays.get(&arr_name)
                    .map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect())
                    .unwrap_or_default();

                if compressed.len() < 2 {
                    return Ok(Value::Float(-1.0)); // invalid compressed data
                }

                let decoded = match octopress_decode_internal(&compressed) {
                    Ok(d) => d,
                    Err(_) => return Ok(Value::Float(-1.0)),
                };

                let out_count = decoded.len();
                RETURNED_ARRAY.with(|r| {
                    *r.borrow_mut() = Some(decoded.into_iter().map(|f| Value::Float(f)).collect());
                });
                return Ok(Value::Float(out_count as f32));
            }

            // octopress_save(compressed_array_name, path) → 0.0
            // .ocp format: 16-byte header [OCP1 magic, method u32 LE, count u32 LE, block_size u32 LE] + f32 body
            if name == "octopress_save" && args.len() == 2 {
                let arr_name = if let ScalarExpr::Ref(n) = &args[0] { n.clone() }
                    else {
                        eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                            .as_str().map_err(|_| CliError::Compile("octopress_save: 1st arg must be array name".into()))?.to_string()
                    };
                let path = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("octopress_save: 2nd arg must be path string".into()))?.to_string();

                let compressed: Vec<f32> = arrays.get(&arr_name)
                    .map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect())
                    .unwrap_or_default();

                if compressed.len() < 2 {
                    return Ok(Value::Float(-1.0));
                }

                let method = compressed[0] as u32;
                let count = compressed[1] as u32;
                let block_size = OCTOPRESS_STATE.with(|s| {
                    s.borrow().as_ref().map(|st| st.block_size).unwrap_or(256)
                });

                // Build binary output
                let mut bytes = Vec::with_capacity(16 + compressed.len() * 4);
                // Magic
                bytes.extend_from_slice(b"OCP1");
                // Header fields (u32 LE)
                bytes.extend_from_slice(&method.to_le_bytes());
                bytes.extend_from_slice(&count.to_le_bytes());
                bytes.extend_from_slice(&block_size.to_le_bytes());
                // Body: f32 values
                for &val in &compressed {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }

                std::fs::write(&path, &bytes)
                    .map_err(|e| CliError::Io(format!("octopress_save: {}", e)))?;
                return Ok(Value::Float(0.0));
            }

            // octopress_load(path) → count (compressed array stored via RETURNED_ARRAY)
            if name == "octopress_load" && args.len() == 1 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("octopress_load: arg must be path string".into()))?.to_string();

                let bytes = std::fs::read(&path)
                    .map_err(|e| CliError::Io(format!("octopress_load: {}", e)))?;

                if bytes.len() < 16 || &bytes[0..4] != b"OCP1" {
                    return Ok(Value::Float(-1.0)); // invalid file
                }

                // Read f32 values from body (after 16-byte header)
                let body = &bytes[16..];
                let float_count = body.len() / 4;
                let mut compressed = Vec::with_capacity(float_count);
                for i in 0..float_count {
                    let off = i * 4;
                    let val = f32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]);
                    compressed.push(val);
                }

                let count = compressed.len();
                RETURNED_ARRAY.with(|r| {
                    *r.borrow_mut() = Some(compressed.into_iter().map(|f| Value::Float(f)).collect());
                });
                return Ok(Value::Float(count as f32));
            }

            // octopress_info(path) → map {method, count, block_size, compressed_size, ratio} or -1.0
            // Reads only the 16-byte .ocp header — no full file load.
            if name == "octopress_info" && args.len() == 1 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("octopress_info: arg must be path string".into()))?.to_string();
                let mut file = std::fs::File::open(&path)
                    .map_err(|e| CliError::Io(format!("octopress_info: {}", e)))?;
                let mut header = [0u8; 16];
                use std::io::Read as _;
                file.read_exact(&mut header)
                    .map_err(|e| CliError::Io(format!("octopress_info: {}", e)))?;
                if &header[0..4] != b"OCP1" {
                    return Ok(Value::Float(-1.0));
                }
                let method = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
                let count = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
                let block_size = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
                let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let body_floats = (file_size.saturating_sub(16)) / 4;

                let mut map = HashMap::new();
                map.insert("method".to_string(), Value::Float(method as f32));
                map.insert("count".to_string(), Value::Float(count as f32));
                map.insert("block_size".to_string(), Value::Float(block_size as f32));
                map.insert("compressed_size".to_string(), Value::Float(body_floats as f32));
                let ratio = if count > 0 { body_floats as f32 / count as f32 } else { 0.0 };
                map.insert("ratio".to_string(), Value::Float(ratio));
                RETURNED_MAP.with(|r| *r.borrow_mut() = Some(map.clone()));
                return Ok(Value::Map(map));
            }

            // ── OctoPress Streaming Decompression ──

            // octopress_stream_open(path) → handle (or -1.0 on error)
            // Opens an .ocp file, reads compressed data, returns a stream handle.
            if name == "octopress_stream_open" && args.len() == 1 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("octopress_stream_open: arg must be path string".into()))?.to_string();
                let bytes = std::fs::read(&path)
                    .map_err(|e| CliError::Io(format!("octopress_stream_open: {}", e)))?;
                if bytes.len() < 16 || &bytes[0..4] != b"OCP1" {
                    return Ok(Value::Float(-1.0));
                }
                let method = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
                let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
                let block_size = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
                let body = &bytes[16..];
                let compressed: Vec<f32> = (0..body.len() / 4).map(|i| {
                    let off = i * 4;
                    f32::from_le_bytes([body[off], body[off+1], body[off+2], body[off+3]])
                }).collect();

                let id = OCTOPRESS_STREAM_NEXT_ID.with(|c| { let v = c.get(); c.set(v + 1); v });
                OCTOPRESS_STREAMS.with(|s| {
                    s.borrow_mut().insert(id, OctoPressStream {
                        compressed, method, original_count: count, block_size, cursor: 0,
                    });
                });
                return Ok(Value::Float(id as f32));
            }

            // octopress_stream_next(handle) → count (block data via RETURNED_ARRAY), 0.0 if done
            if name == "octopress_stream_next" && args.len() == 1 {
                let handle = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let result = OCTOPRESS_STREAMS.with(|s| {
                    let mut streams = s.borrow_mut();
                    let stream = streams.get_mut(&handle).ok_or_else(|| CliError::Runtime("octopress_stream_next: invalid stream handle".into()))?;
                    if stream.cursor >= stream.original_count as usize {
                        return Ok(Vec::new()); // done
                    }
                    let all_decoded = octopress_decode_internal(&stream.compressed)?;
                    let end = (stream.cursor + stream.block_size as usize).min(all_decoded.len());
                    let block: Vec<f32> = all_decoded[stream.cursor..end].to_vec();
                    stream.cursor = end;
                    Ok(block)
                })?;
                if result.is_empty() {
                    return Ok(Value::Float(0.0));
                }
                let count = result.len();
                RETURNED_ARRAY.with(|r| {
                    *r.borrow_mut() = Some(result.into_iter().map(|f| Value::Float(f)).collect());
                });
                return Ok(Value::Float(count as f32));
            }

            // octopress_stream_info(handle) → map {method, count, block_size, blocks, cursor}
            if name == "octopress_stream_info" && args.len() == 1 {
                let handle = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let info = OCTOPRESS_STREAMS.with(|s| {
                    let streams = s.borrow();
                    let stream = streams.get(&handle).ok_or_else(|| CliError::Runtime("octopress_stream_info: invalid stream handle".into()))?;
                    let n_blocks = (stream.original_count as f32 / stream.block_size as f32).ceil();
                    let mut map = HashMap::new();
                    map.insert("method".to_string(), Value::Float(stream.method as f32));
                    map.insert("count".to_string(), Value::Float(stream.original_count as f32));
                    map.insert("block_size".to_string(), Value::Float(stream.block_size as f32));
                    map.insert("blocks".to_string(), Value::Float(n_blocks));
                    map.insert("cursor".to_string(), Value::Float(stream.cursor as f32));
                    Ok(Value::Map(map))
                })?;
                RETURNED_MAP.with(|r| {
                    if let Value::Map(ref m) = info { *r.borrow_mut() = Some(m.clone()); }
                });
                return Ok(info);
            }

            // octopress_stream_reset(handle) → 0.0
            if name == "octopress_stream_reset" && args.len() == 1 {
                let handle = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                OCTOPRESS_STREAMS.with(|s| {
                    let mut streams = s.borrow_mut();
                    if let Some(stream) = streams.get_mut(&handle) {
                        stream.cursor = 0;
                    }
                });
                return Ok(Value::Float(0.0));
            }

            // octopress_stream_close(handle) → 0.0
            if name == "octopress_stream_close" && args.len() == 1 {
                let handle = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                OCTOPRESS_STREAMS.with(|s| s.borrow_mut().remove(&handle));
                return Ok(Value::Float(0.0));
            }

            // octopress_gpu_encode(data_array_name, block_size) → count
            // GPU-accelerated fractal compression. Falls back to CPU method 2 if no GPU or data too small.
            if name == "octopress_gpu_encode" && args.len() == 2 {
                let arr_name = if let ScalarExpr::Ref(n) = &args[0] { n.clone() }
                    else {
                        eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                            .as_str().map_err(|_| CliError::Compile("octopress_gpu_encode: 1st arg must be array name".into()))?.to_string()
                    };
                let block_size = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;

                let data: Vec<f32> = arrays.get(&arr_name)
                    .map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect())
                    .unwrap_or_default();

                // Sanitize: replace NaN/Inf (audit N5)
                let data: Vec<f32> = data.into_iter().map(|v| {
                    if v.is_nan() { 0.0 }
                    else if v.is_infinite() { if v > 0.0 { f32::MAX } else { f32::MIN } }
                    else { v }
                }).collect();

                let n_elements = data.len() as u32;
                let range_size = block_size / 2;
                let n_ranges = if range_size > 0 { n_elements / range_size } else { 0 };
                let n_domains = if block_size > 0 { n_elements / block_size } else { 0 };

                // Fall back to CPU method 2 if data too small or no GPU
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if n_domains == 0 || n_ranges == 0 || n_elements < block_size * 2 || device_ptr == 0 {
                    // CPU fallback: use existing method 2 path
                    let bs = block_size as usize;
                    let rs = bs / 2;
                    let nr = data.len() / rs;
                    let nd = data.len() / bs;
                    let mut out = vec![2.0, data.len() as f32, bs as f32, nr as f32];
                    for r in 0..nr {
                        let r_start = r * rs;
                        let r_end = (r_start + rs).min(data.len());
                        let range_block = &data[r_start..r_end];
                        let mut best_d = 0u32;
                        let mut best_s = 1.0f32;
                        let mut best_o = 0.0f32;
                        let mut best_mse = f32::MAX;
                        for d in 0..nd {
                            let d_start = d * bs;
                            let d_end = (d_start + bs).min(data.len());
                            let dom = &data[d_start..d_end];
                            let down: Vec<f32> = dom.chunks(2).map(|c| if c.len() == 2 { (c[0]+c[1])/2.0 } else { c[0] }).collect();
                            let n = range_block.len().min(down.len()) as f32;
                            let sd: f32 = down.iter().take(range_block.len()).sum();
                            let sr: f32 = range_block.iter().sum();
                            let sdd: f32 = down.iter().take(range_block.len()).map(|x| x*x).sum();
                            let sdr: f32 = down.iter().zip(range_block.iter()).map(|(a,b)| a*b).sum();
                            let denom = n * sdd - sd * sd;
                            let (sc, of) = if denom.abs() > 1e-12 {
                                ((n*sdr - sd*sr)/denom, (sr - (n*sdr - sd*sr)/denom * sd)/n)
                            } else { (0.0, sr / n) };
                            let mse: f32 = down.iter().zip(range_block.iter()).map(|(a,b)| { let p = sc*a + of; (p-b)*(p-b) }).sum::<f32>() / n;
                            if mse < best_mse { best_mse = mse; best_d = d as u32; best_s = sc; best_o = of; }
                        }
                        out.push(best_d as f32);
                        out.push(best_s);
                        out.push(best_o);
                    }
                    let count = out.len();
                    RETURNED_ARRAY.with(|r| { *r.borrow_mut() = Some(out.into_iter().map(|f| Value::Float(f)).collect()); });
                    return Ok(Value::Float(count as f32));
                }

                // --- GPU path ---
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                let total_threads = n_ranges * n_domains;
                let out_offset = n_elements; // SSD output starts after input data
                let globals_size = n_elements + total_threads; // input data + SSD results

                // 1. Boot temporary VM (1 instance, 1 register, globals_size floats)
                let vm = octoflow_vulkan::vm::vm_create(gpu_dev, 1, 1, globals_size)
                    .map_err(|e| CliError::Runtime(format!("octopress_gpu_encode: vm_create: {}", e)))?;
                let vm_id = VM_NEXT_ID.with(|c| { let id = c.get(); c.set(id + 1); id });
                GPU_VMS.with(|vms| vms.borrow_mut().insert(vm_id, vm));

                // 2. Stage data upload to globals[0..N]
                GPU_VMS.with(|vms| -> Result<(), CliError> {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).unwrap();
                    let pending = octoflow_vulkan::vm::vm_stage_write(gpu_dev, &vm.globals, 0, &data)
                        .map_err(|e| CliError::Runtime(format!("octopress_gpu_encode: stage_write: {}", e)))?;
                    if let Some(pu) = pending {
                        VM_PENDING_UPLOADS.with(|uploads| {
                            uploads.borrow_mut().entry(vm_id).or_default().push(pu);
                        });
                    }
                    Ok(())
                })?;

                // 3. Build VmOp with SPIR-V kernel + push constants
                let wg_size = 64u32;
                let spirv = emit_octopress_fractal_spirv(wg_size);
                let pc = vec![
                    f32::from_bits(n_domains),
                    f32::from_bits(range_size),
                    f32::from_bits(block_size),
                    f32::from_bits(out_offset),
                    f32::from_bits(total_threads),
                ];
                let n_workgroups = (total_threads + wg_size - 1) / wg_size;
                let op = octoflow_vulkan::vm::VmOp {
                    spirv,
                    push_constants: pc,
                    workgroups: (n_workgroups, 1, 1),
                    indirect_offset: None,
                };
                let ops = vec![op];

                // 4. Build program (flushes staged uploads + compiles dispatches)
                let uploads = VM_PENDING_UPLOADS.with(|u| {
                    u.borrow_mut().remove(&vm_id).unwrap_or_default()
                });
                let prog = GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).unwrap();
                    octoflow_vulkan::vm::vm_build_program_with_uploads(gpu_dev, vm, &ops, uploads)
                        .map_err(|e| CliError::Runtime(format!("octopress_gpu_encode: build: {}", e)))
                })?;

                // 5. Execute on GPU
                octoflow_vulkan::vm::vm_execute_program(gpu_dev, &prog)
                    .map_err(|e| CliError::Runtime(format!("octopress_gpu_encode: execute: {}", e)))?;

                // 6. Read SSD results back from globals[out_offset .. out_offset + total_threads]
                let ssd_data = GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).unwrap();
                    octoflow_vulkan::vm::vm_read_globals(gpu_dev, vm, out_offset, total_threads)
                        .map_err(|e| CliError::Runtime(format!("octopress_gpu_encode: read: {}", e)))
                })?;

                // 7. Shutdown temporary VM (drop program first)
                drop(prog);
                GPU_VMS.with(|vms| { vms.borrow_mut().remove(&vm_id); });

                // 8. CPU post-process: for each range, find best domain (min SSD),
                //    then compute least-squares scale/offset
                let mut out = vec![2.0, data.len() as f32, block_size as f32, n_ranges as f32];
                let rs = range_size as usize;
                let bs = block_size as usize;
                let nd = n_domains as usize;

                for r in 0..n_ranges as usize {
                    // Find domain with minimum SSD for this range
                    let ssd_base = r * nd;
                    let mut best_d = 0usize;
                    let mut best_ssd = f32::MAX;
                    for d in 0..nd {
                        let ssd = ssd_data[ssd_base + d];
                        if ssd < best_ssd {
                            best_ssd = ssd;
                            best_d = d;
                        }
                    }

                    // Compute least-squares scale/offset for best domain
                    let r_start = r * rs;
                    let r_end = (r_start + rs).min(data.len());
                    let range_block = &data[r_start..r_end];
                    let d_start = best_d * bs;
                    let d_end = (d_start + bs).min(data.len());
                    let dom = &data[d_start..d_end];
                    let down: Vec<f32> = dom.chunks(2).map(|c| if c.len() == 2 { (c[0]+c[1])/2.0 } else { c[0] }).collect();
                    let n = range_block.len().min(down.len()) as f32;
                    let sd: f32 = down.iter().take(range_block.len()).sum();
                    let sr: f32 = range_block.iter().sum();
                    let sdd: f32 = down.iter().take(range_block.len()).map(|x| x*x).sum();
                    let sdr: f32 = down.iter().zip(range_block.iter()).map(|(a,b)| a*b).sum();
                    let denom = n * sdd - sd * sd;
                    let (scale, offset) = if denom.abs() > 1e-12 {
                        let s = (n * sdr - sd * sr) / denom;
                        let o = (sr - s * sd) / n;
                        (s, o)
                    } else {
                        (0.0, sr / n)
                    };

                    out.push(best_d as f32);
                    out.push(scale);
                    out.push(offset);
                }

                let count = out.len();
                RETURNED_ARRAY.with(|r| {
                    *r.borrow_mut() = Some(out.into_iter().map(|f| Value::Float(f)).collect());
                });
                return Ok(Value::Float(count as f32));
            }

            // ── Array Utility Builtins ──

            // array_new(count, value) → count (array via RETURNED_ARRAY)
            // Create a pre-filled array. Cap at 10M elements.
            if name == "array_new" && args.len() == 2 {
                let count = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;
                let value = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                if count > 10_000_000 {
                    return Err(CliError::Runtime("array_new: count exceeds 10M limit".into()));
                }
                if count > 0 {
                    let arr = vec![value; count];
                    RETURNED_ARRAY.with(|r| {
                        *r.borrow_mut() = Some(arr);
                    });
                }
                return Ok(Value::Float(count as f32));
            }

            // extend(dest_array, source_array) → count_added
            // Append all elements of source_array to dest_array.
            if name == "extend" && args.len() == 2 {
                if let (ScalarExpr::Ref(dest_name), ScalarExpr::Ref(src_name)) = (&args[0], &args[1]) {
                    let src_vals: Vec<Value> = arrays.get(src_name.as_str())
                        .map(|a| a.clone())
                        .unwrap_or_default();
                    let count = src_vals.len() as f32;
                    if let Some(dest) = arrays.get_mut(dest_name.as_str()) {
                        dest.extend(src_vals);
                    } else {
                        arrays.insert(dest_name.clone(), src_vals);
                    }
                    return Ok(Value::Float(count));
                }
                return Err(CliError::Runtime("extend() requires two array names".into()));
            }

            // array_copy(dest, dest_offset, source, src_offset, count) → count_copied
            // Bulk copy count elements from source[src_offset..] into dest[dest_offset..].
            if name == "array_copy" && args.len() == 5 {
                if let ScalarExpr::Ref(dest_name) = &args[0] {
                    let dest_off = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                        .as_float()? as usize;
                    if let ScalarExpr::Ref(src_name) = &args[2] {
                        let src_off = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                            .as_float()? as usize;
                        let count = eval_scalar(&args[4], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                            .as_float()? as usize;

                        let src_vals: Vec<Value> = arrays.get(src_name.as_str())
                            .map(|a| {
                                let start = src_off.min(a.len());
                                let end = (src_off + count).min(a.len());
                                a[start..end].to_vec()
                            })
                            .unwrap_or_default();

                        let copied = src_vals.len();
                        if let Some(dest) = arrays.get_mut(dest_name.as_str()) {
                            if dest_off > dest.len() {
                                return Err(CliError::Runtime(format!(
                                    "array_copy: dest_offset {} out of bounds (dest len {})", dest_off, dest.len()
                                )));
                            }
                            for (i, val) in src_vals.into_iter().enumerate() {
                                let idx = dest_off + i;
                                if idx < dest.len() {
                                    dest[idx] = val;
                                }
                            }
                        } else {
                            return Err(CliError::Runtime(format!(
                                "array_copy: destination array '{}' does not exist", dest_name
                            )));
                        }
                        return Ok(Value::Float(copied as f32));
                    }
                }
                return Err(CliError::Runtime("array_copy(dest, dest_off, source, src_off, count)".into()));
            }

            // array_extract(source, offset, count) → new array via RETURNED_ARRAY
            if name == "array_extract" && args.len() == 3 {
                if let ScalarExpr::Ref(src_name) = &args[0] {
                    let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                        .as_float()? as usize;
                    let count = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                        .as_float()? as usize;

                    let result: Vec<Value> = arrays.get(src_name.as_str())
                        .map(|a| {
                            let start = offset.min(a.len());
                            let end = (offset + count).min(a.len());
                            a[start..end].to_vec()
                        })
                        .unwrap_or_default();

                    let n = result.len() as f32;
                    RETURNED_ARRAY.with(|r| *r.borrow_mut() = Some(result));
                    return Ok(Value::Float(n));
                }
                return Err(CliError::Runtime("array_extract(source, offset, count)".into()));
            }

            // ── CPU Thread Pool (Primitive 7) ──

            // loom_threads(n) → 0.0 — initialize thread pool with n workers (0 = auto)
            if name == "loom_threads" && args.len() == 1 {
                let n = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;

                let already = LOOM_THREAD_POOL.with(|p| p.borrow().is_some());
                if already {
                    return Ok(Value::Float(0.0)); // already initialized
                }

                let count = if n == 0 {
                    std::thread::available_parallelism()
                        .map(|p| p.get().saturating_sub(1).max(1))
                        .unwrap_or(2) as u32
                } else {
                    n
                };

                // Result channel: workers → main
                let (result_tx, result_rx) = std::sync::mpsc::channel();

                // Task channel: main → workers (shared via Arc<Mutex>)
                let (task_tx, task_rx) = std::sync::mpsc::channel();
                let task_rx = std::sync::Arc::new(std::sync::Mutex::new(task_rx));

                let mut workers = Vec::new();
                for _ in 0..count {
                    let rx = task_rx.clone();
                    let tx = result_tx.clone();
                    workers.push(std::thread::spawn(move || {
                        loop {
                            let task = match rx.lock().unwrap().recv() {
                                Ok(t) => t,
                                Err(_) => break,
                            };
                            match task {
                                PoolTask::FileRead { path, result_id } => {
                                    match std::fs::read(&path) {
                                        Ok(bytes) => { tx.send((result_id, AsyncResult::FileData(bytes))).ok(); }
                                        Err(e) => { tx.send((result_id, AsyncResult::Error(e.to_string()))).ok(); }
                                    }
                                }
                                PoolTask::FileWrite { path, data } => {
                                    std::fs::write(&path, &data).ok();
                                    // fire-and-forget, no result needed
                                }
                                PoolTask::Shutdown => break,
                            }
                        }
                    }));
                }

                LOOM_THREAD_POOL.with(|p| {
                    *p.borrow_mut() = Some(LoomPool { workers, task_tx, result_rx });
                });

                return Ok(Value::Float(0.0));
            }

            // loom_cpu_count() → available CPU cores
            if name == "loom_cpu_count" && args.is_empty() {
                let count = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1);
                return Ok(Value::Float(count as f32));
            }

            // loom_async_read(path) → handle_id
            if name == "loom_async_read" && args.len() == 1 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("loom_async_read: path must be string".into()))?.to_string();

                let handle_id = LOOM_ASYNC_NEXT_ID.with(|id| {
                    let current = id.get();
                    id.set(current + 1);
                    current
                });

                let submitted = LOOM_THREAD_POOL.with(|pool| {
                    if let Some(ref pool) = *pool.borrow() {
                        pool.task_tx.send(PoolTask::FileRead {
                            path: path.clone(),
                            result_id: handle_id,
                        }).ok();
                        true
                    } else {
                        false
                    }
                });

                if !submitted {
                    // Fallback: synchronous read
                    let result = match std::fs::read(&path) {
                        Ok(bytes) => AsyncResult::FileData(bytes),
                        Err(e) => AsyncResult::Error(e.to_string()),
                    };
                    LOOM_ASYNC_RESULTS.with(|r| r.borrow_mut().insert(handle_id, result));
                }

                return Ok(Value::Float(handle_id as f32));
            }

            // loom_await(handle_id) → result (string for file reads, -1.0 for error)
            if name == "loom_await" && args.len() == 1 {
                let handle_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;

                loop {
                    // Drain pending results from pool
                    LOOM_THREAD_POOL.with(|pool| {
                        if let Some(ref pool) = *pool.borrow() {
                            while let Ok((id, result)) = pool.result_rx.try_recv() {
                                LOOM_ASYNC_RESULTS.with(|r| r.borrow_mut().insert(id, result));
                            }
                        }
                    });

                    let result = LOOM_ASYNC_RESULTS.with(|r| r.borrow_mut().remove(&handle_id));
                    if let Some(result) = result {
                        return match result {
                            AsyncResult::FileData(bytes) => {
                                Ok(Value::Str(String::from_utf8_lossy(&bytes).into_owned()))
                            }
                            AsyncResult::Done => Ok(Value::Float(1.0)),
                            AsyncResult::Error(_) => Ok(Value::Float(-1.0)),
                        };
                    }

                    std::thread::yield_now();
                }
            }

            // vm_write_metrics(vm_id, offset, array_name) → 0.0
            if (name == "vm_write_metrics" || name == "loom_write_metrics") && args.len() == 3 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let arr_name = if let ScalarExpr::Ref(n) = &args[2] { n.clone() }
                    else { return Err(CliError::Compile("vm_write_metrics: 3rd arg must be array name".into())); };
                let data = gpu_array_get(&arr_name).unwrap_or_else(|| {
                    arrays.get(&arr_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("vm_write_metrics: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_write_metrics: unknown VM {}", vm_id)))?;
                    octoflow_vulkan::vm::vm_write_metrics(gpu_dev, vm, offset, &data)
                        .map_err(|e| CliError::Runtime(format!("vm_write_metrics: {}", e)))
                })?;
                return Ok(Value::Float(0.0));
            }
            // vm_write_control(vm_id, offset, array_name) → 0.0
            if (name == "vm_write_control" || name == "loom_write_control") && args.len() == 3 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let arr_name = if let ScalarExpr::Ref(n) = &args[2] { n.clone() }
                    else { return Err(CliError::Compile("vm_write_control: 3rd arg must be array name".into())); };
                let data = gpu_array_get(&arr_name).unwrap_or_else(|| {
                    arrays.get(&arr_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("vm_write_control: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_write_control: unknown VM {}", vm_id)))?;
                    octoflow_vulkan::vm::vm_write_control(gpu_dev, vm, offset, &data)
                        .map_err(|e| CliError::Runtime(format!("vm_write_control: {}", e)))
                })?;
                return Ok(Value::Float(0.0));
            }
            // vm_load_weights(vm_id, offset, gpu_array_name) → count of floats loaded
            // Loads weights into VM globals. GPU-resident arrays use vkCmdCopyBuffer (fast).
            // CPU arrays fall back to host-visible upload (still correct, just slower).
            if name == "vm_load_weights" && args.len() == 3 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let arr_name = if let ScalarExpr::Ref(n) = &args[2] { n.clone() }
                    else { return Err(CliError::Compile("vm_load_weights: 3rd arg must be GPU array name".into())); };
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("vm_load_weights: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                // Try GPU_ARRAYS first (gguf_load_tensor output)
                enum WeightSource { GpuBuffer(u64, u32), CpuData(Vec<f32>) }
                let source = GPU_ARRAYS.with(|ga| {
                    let arrays = ga.borrow();
                    if let Some(storage) = arrays.get(&arr_name) {
                        match storage {
                            GpuArrayStorage::Resident(buf) => {
                                let r = buf.as_ref();
                                Ok(WeightSource::GpuBuffer(r.buffer, r.len as u32))
                            }
                            GpuArrayStorage::Cpu(data) => {
                                Ok(WeightSource::CpuData(data.clone()))
                            }
                        }
                    } else {
                        Err(arr_name.clone())
                    }
                });
                let (count, source) = match source {
                    Ok(s) => match &s {
                        WeightSource::GpuBuffer(_, c) => (*c, s),
                        WeightSource::CpuData(d) => (d.len() as u32, s),
                    },
                    Err(_) => return Err(CliError::Runtime(format!(
                        "vm_load_weights: GPU array '{}' not found", arr_name))),
                };
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_load_weights: unknown VM {}", vm_id)))?;
                    match &source {
                        WeightSource::GpuBuffer(vk_buf, cnt) => {
                            octoflow_vulkan::vm::vm_gpu_to_globals(gpu_dev, vm, offset, *vk_buf, *cnt)
                                .map_err(|e| CliError::Runtime(format!("vm_load_weights: {}", e)))
                        }
                        WeightSource::CpuData(data) => {
                            octoflow_vulkan::vm::vm_write_globals(gpu_dev, vm, offset, data)
                                .map_err(|e| CliError::Runtime(format!("vm_load_weights: {}", e)))
                        }
                    }
                })?;
                return Ok(Value::Float(count as f32));
            }
            // vm_set_heap(vm_id, data_array) → count of floats loaded
            // Sets the VM's heap buffer (binding 4) for immutable bulk data.
            if (name == "vm_set_heap" || name == "loom_set_heap") && args.len() == 2 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("vm_set_heap: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                let data: Vec<f32> = extract_array_arg("vm_set_heap", &args[1], arrays)?;
                let count = data.len();
                GPU_VMS.with(|vms| {
                    let mut vms = vms.borrow_mut();
                    let vm = vms.get_mut(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_set_heap: unknown VM {}", vm_id)))?;
                    octoflow_vulkan::vm::vm_set_heap(gpu_dev, vm, &data)
                        .map_err(|e| CliError::Runtime(format!("vm_set_heap: {}", e)))
                })?;
                return Ok(Value::Float(count as f32));
            }
            // vm_shutdown(vm_id) → 0.0
            if (name == "vm_shutdown" || name == "loom_shutdown") && args.len() == 1 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                // Drop all VmPrograms owned by this VM while the device is still alive.
                // This prevents VmProgram::Drop from calling vkDestroyPipeline after the
                // device is destroyed at process exit.
                let owned_prog_ids: Vec<u32> = VM_PROG_OWNERS.with(|owners| {
                    owners.borrow().iter()
                        .filter(|(_, &vid)| vid == vm_id)
                        .map(|(&pid, _)| pid)
                        .collect()
                });
                VM_PROGRAMS.with(|progs| {
                    let mut progs = progs.borrow_mut();
                    for pid in &owned_prog_ids {
                        progs.remove(pid); // Drop calls vkDestroyPipeline while device is valid
                    }
                });
                VM_PROG_OWNERS.with(|owners| {
                    let mut owners = owners.borrow_mut();
                    for pid in &owned_prog_ids {
                        owners.remove(pid);
                    }
                });
                GPU_VMS.with(|vms| {
                    let mut vms = vms.borrow_mut();
                    if let Some(vm) = vms.remove(&vm_id) {
                        // Decrement VRAM tracking
                        let cost = (vm.reg_size as u64 * 4 * 32 * vm.n_instances as u64) + (vm.globals.len() as u64 * 4);
                        LOOM_VRAM_USED.with(|u| u.set(u.get().saturating_sub(cost)));
                        Ok(())
                    } else {
                        Err(CliError::Runtime(format!("vm_shutdown: unknown VM {}", vm_id)))
                    }
                })?;
                return Ok(Value::Float(0.0));
            }
            // vm_dispatch(vm_id, spv_path, push_constants_array, workgroups) → 0.0
            // Stages a dispatch op. Accumulated until vm_build is called.
            if (name == "loom_dispatch" || name == "vm_dispatch") && args.len() == 4 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let spv_path = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("vm_dispatch: spv_path must be string".into()))?.to_string();
                let pc_name = if let ScalarExpr::Ref(n) = &args[2] { n.clone() }
                    else { return Err(CliError::Compile("vm_dispatch: 3rd arg must be array name".into())); };
                let workgroups = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let pc_data = gpu_array_get(&pc_name).unwrap_or_else(|| {
                    arrays.get(&pc_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                let spirv = SPIRV_FILE_CACHE.with(|cache| {
                    let c = cache.borrow();
                    if let Some(data) = c.get(&spv_path) {
                        return Ok(data.clone());
                    }
                    drop(c);
                    // Check prefetch thread
                    let prefetched = SPIRV_PREFETCH.with(|pf| pf.borrow_mut().remove(&spv_path));
                    if let Some(handle) = prefetched {
                        let data = handle.join().map_err(|_| CliError::Runtime("prefetch thread panicked".into()))?
                            .map_err(|e| CliError::Io(e))?;
                        cache.borrow_mut().insert(spv_path.clone(), data.clone());
                        return Ok(data);
                    }
                    let data = std::fs::read(&spv_path)
                        .map_err(|e| CliError::Io(format!("vm_dispatch: read {}: {}", spv_path, e)))?;
                    cache.borrow_mut().insert(spv_path.clone(), data.clone());
                    Ok(data)
                })?;
                let op = octoflow_vulkan::vm::VmOp {
                    spirv,
                    push_constants: pc_data,
                    workgroups: (workgroups, 1, 1),
                    indirect_offset: None,
                };
                VM_STAGED_OPS.with(|staged| {
                    staged.borrow_mut().entry(vm_id).or_insert_with(Vec::new).push(op);
                });
                return Ok(Value::Float(0.0));
            }
            // vm_dispatch_indirect(vm_id, spv_path, push_constants_array, control_offset_floats)
            // Stages an indirect dispatch: GPU reads workgroup counts from control buffer.
            // control_offset_floats = float offset in control SSBO where 3 uint32 {x,y,z} live.
            if name == "vm_dispatch_indirect" && args.len() == 4 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let spv_path = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("vm_dispatch_indirect: spv_path must be string".into()))?.to_string();
                let pc_name = if let ScalarExpr::Ref(n) = &args[2] { n.clone() }
                    else { return Err(CliError::Compile("vm_dispatch_indirect: 3rd arg must be array name".into())); };
                let control_offset_floats = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let pc_data = gpu_array_get(&pc_name).unwrap_or_else(|| {
                    arrays.get(&pc_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                let spirv = SPIRV_FILE_CACHE.with(|cache| {
                    let c = cache.borrow();
                    if let Some(data) = c.get(&spv_path) {
                        return Ok(data.clone());
                    }
                    drop(c);
                    // Check prefetch thread
                    let prefetched = SPIRV_PREFETCH.with(|pf| pf.borrow_mut().remove(&spv_path));
                    if let Some(handle) = prefetched {
                        let data = handle.join().map_err(|_| CliError::Runtime("prefetch thread panicked".into()))?
                            .map_err(|e| CliError::Io(e))?;
                        cache.borrow_mut().insert(spv_path.clone(), data.clone());
                        return Ok(data);
                    }
                    let data = std::fs::read(&spv_path)
                        .map_err(|e| CliError::Io(format!("vm_dispatch_indirect: read {}: {}", spv_path, e)))?;
                    cache.borrow_mut().insert(spv_path.clone(), data.clone());
                    Ok(data)
                })?;
                let byte_offset = (control_offset_floats as u64) * 4;
                let op = octoflow_vulkan::vm::VmOp {
                    spirv,
                    push_constants: pc_data,
                    workgroups: (1, 1, 1), // ignored for indirect dispatch
                    indirect_offset: Some(byte_offset),
                };
                VM_STAGED_OPS.with(|staged| {
                    staged.borrow_mut().entry(vm_id).or_insert_with(Vec::new).push(op);
                });
                return Ok(Value::Float(0.0));
            }
            // vm_dispatch_mem(vm_id, spirv_byte_array, push_constants_array, workgroups) → 0.0
            // Like vm_dispatch but takes in-memory SPIR-V bytes instead of file path.
            // spirv_byte_array: OctoFlow array where each element is one byte (0-255).
            if (name == "loom_dispatch_jit" || name == "vm_dispatch_mem") && args.len() == 4 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let spirv_name = if let ScalarExpr::Ref(n) = &args[1] { n.clone() }
                    else { return Err(CliError::Compile("vm_dispatch_mem: 2nd arg must be array name".into())); };
                let pc_name = if let ScalarExpr::Ref(n) = &args[2] { n.clone() }
                    else { return Err(CliError::Compile("vm_dispatch_mem: 3rd arg must be array name".into())); };
                let workgroups = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let spirv_arr = gpu_array_get(&spirv_name).unwrap_or_else(|| {
                    arrays.get(&spirv_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                let spirv: Vec<u8> = spirv_arr.iter().map(|f| *f as u8).collect();
                let pc_data = gpu_array_get(&pc_name).unwrap_or_else(|| {
                    arrays.get(&pc_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                let op = octoflow_vulkan::vm::VmOp {
                    spirv,
                    push_constants: pc_data,
                    workgroups: (workgroups, 1, 1),
                    indirect_offset: None,
                };
                VM_STAGED_OPS.with(|staged| {
                    staged.borrow_mut().entry(vm_id).or_insert_with(Vec::new).push(op);
                });
                return Ok(Value::Float(0.0));
            }
            // vm_dispatch_indirect_mem(vm_id, spirv_byte_array, push_constants_array, control_offset)
            // Like vm_dispatch_indirect but takes in-memory SPIR-V bytes.
            if name == "vm_dispatch_indirect_mem" && args.len() == 4 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let spirv_name = if let ScalarExpr::Ref(n) = &args[1] { n.clone() }
                    else { return Err(CliError::Compile("vm_dispatch_indirect_mem: 2nd arg must be array name".into())); };
                let pc_name = if let ScalarExpr::Ref(n) = &args[2] { n.clone() }
                    else { return Err(CliError::Compile("vm_dispatch_indirect_mem: 3rd arg must be array name".into())); };
                let control_offset_floats = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let spirv_arr = gpu_array_get(&spirv_name).unwrap_or_else(|| {
                    arrays.get(&spirv_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                let spirv: Vec<u8> = spirv_arr.iter().map(|f| *f as u8).collect();
                let pc_data = gpu_array_get(&pc_name).unwrap_or_else(|| {
                    arrays.get(&pc_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                let byte_offset = (control_offset_floats as u64) * 4;
                let op = octoflow_vulkan::vm::VmOp {
                    spirv,
                    push_constants: pc_data,
                    workgroups: (1, 1, 1),
                    indirect_offset: Some(byte_offset),
                };
                VM_STAGED_OPS.with(|staged| {
                    staged.borrow_mut().entry(vm_id).or_insert_with(Vec::new).push(op);
                });
                return Ok(Value::Float(0.0));
            }
            // vm_build(vm_id) → program_id
            // Builds VkCommandBuffer from all staged dispatches.
            if (name == "loom_build" || name == "vm_build") && args.len() == 1 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("vm_build: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                let ops = VM_STAGED_OPS.with(|staged| {
                    staged.borrow_mut().remove(&vm_id).unwrap_or_default()
                });
                if ops.is_empty() {
                    return Err(CliError::Runtime(format!("vm_build: no staged ops for VM {}", vm_id)));
                }
                // Consume pending uploads for this VM (batched into command buffer).
                let uploads = VM_PENDING_UPLOADS.with(|u| {
                    u.borrow_mut().remove(&vm_id).unwrap_or_default()
                });
                let prog = GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_build: unknown VM {}", vm_id)))?;
                    octoflow_vulkan::vm::vm_build_program_with_uploads(gpu_dev, vm, &ops, uploads)
                        .map_err(|e| CliError::Runtime(format!("vm_build: {}", e)))
                })?;
                let prog_id = VM_PROG_NEXT_ID.with(|c| { let id = c.get(); c.set(id + 1); id });
                VM_PROGRAMS.with(|progs| progs.borrow_mut().insert(prog_id, prog));
                VM_PROG_OWNERS.with(|owners| owners.borrow_mut().insert(prog_id, vm_id));
                return Ok(Value::Float(prog_id as f32));
            }
            // vm_execute(program_id) → 0.0
            // Submits command buffer, waits for completion. Homeostasis pacing applied.
            if (name == "loom_run" || name == "vm_execute") && args.len() == 1 {
                let prog_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("vm_execute: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                // Look up VM ID from program ownership for homeostasis
                let vm_id = VM_PROG_OWNERS.with(|o| o.borrow().get(&prog_id).copied()).unwrap_or(0);
                let t0 = std::time::Instant::now();
                VM_PROGRAMS.with(|progs| {
                    let progs = progs.borrow();
                    let prog = progs.get(&prog_id).ok_or_else(||
                        CliError::Runtime(format!("vm_execute: unknown program {}", prog_id)))?;
                    octoflow_vulkan::vm::vm_execute_program(gpu_dev, prog)
                        .map_err(|e| CliError::Runtime(format!("vm_execute: {}", e)))
                })?;
                let elapsed_us = t0.elapsed().as_micros() as u64;
                LAST_DISPATCH_US.with(|t| t.set(elapsed_us));
                // Homeostasis: track dispatch timing and apply pacing
                LOOM_DISPATCH_COUNT.with(|c| {
                    *c.borrow_mut().entry(vm_id).or_insert(0) += 1;
                });
                LOOM_BASELINE_US.with(|b| {
                    let mut b = b.borrow_mut();
                    let baseline = b.entry(vm_id).or_insert(elapsed_us);
                    if elapsed_us > 0 {
                        // Exponential moving average of dispatch time
                        *baseline = (*baseline * 7 + elapsed_us) / 8;
                    }
                });
                // If dispatch took >20% longer than baseline, GPU may be throttling
                let (baseline, pace) = LOOM_BASELINE_US.with(|b| {
                    let b = b.borrow();
                    let bl = *b.get(&vm_id).unwrap_or(&0);
                    let p = LOOM_PACE_US.with(|p| *p.borrow().get(&vm_id).unwrap_or(&0));
                    (bl, p)
                });
                if baseline > 0 && elapsed_us > baseline + baseline / 5 {
                    // Throttling detected: increase pace (cap at 2ms)
                    let new_pace = (pace + 50).min(2000);
                    LOOM_PACE_US.with(|p| p.borrow_mut().insert(vm_id, new_pace));
                } else if pace > 0 {
                    // Stable: decrease pace toward 0
                    let new_pace = pace.saturating_sub(10);
                    LOOM_PACE_US.with(|p| p.borrow_mut().insert(vm_id, new_pace));
                }
                // Accumulate pacing debt (settled at vm_present) instead of per-dispatch sleep
                let current_pace = LOOM_PACE_US.with(|p| *p.borrow().get(&vm_id).unwrap_or(&0));
                if current_pace > 0 {
                    LOOM_PACE_DEBT_US.with(|d| d.set(d.get() + current_pace));
                    LOOM_PACED_COUNT.with(|c| {
                        *c.borrow_mut().entry(vm_id).or_insert(0) += 1;
                    });
                }
                return Ok(Value::Float(0.0));
            }

            // vm_execute_async(program_id) → 0.0
            // Submits command buffer without waiting — GPU runs autonomously.
            if (name == "loom_launch" || name == "vm_execute_async") && args.len() == 1 {
                let prog_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("vm_execute_async: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                VM_PROGRAMS.with(|progs| {
                    let progs = progs.borrow();
                    let prog = progs.get(&prog_id).ok_or_else(||
                        CliError::Runtime(format!("vm_execute_async: unknown program {}", prog_id)))?;
                    octoflow_vulkan::vm::vm_execute_async(gpu_dev, prog)
                        .map_err(|e| CliError::Runtime(format!("vm_execute_async: {}", e)))
                })?;
                return Ok(Value::Float(0.0));
            }

            // vm_poll(program_id) → 1.0 (done) or 0.0 (still running)
            // Non-blocking fence status check.
            if (name == "loom_poll" || name == "vm_poll") && args.len() == 1 {
                let prog_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let done = VM_PROGRAMS.with(|progs| {
                    let progs = progs.borrow();
                    let prog = progs.get(&prog_id).ok_or_else(||
                        CliError::Runtime(format!("vm_poll: unknown program {}", prog_id)))?;
                    Ok(octoflow_vulkan::vm::vm_poll_program(prog))
                })?;
                return Ok(Value::Float(if done { 1.0 } else { 0.0 }));
            }

            // vm_wait(program_id) → 0.0
            // Blocks until GPU work is complete.
            if (name == "vm_wait" || name == "loom_wait") && args.len() == 1 {
                let prog_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                VM_PROGRAMS.with(|progs| {
                    let progs = progs.borrow();
                    let prog = progs.get(&prog_id).ok_or_else(||
                        CliError::Runtime(format!("vm_wait: unknown program {}", prog_id)))?;
                    octoflow_vulkan::vm::vm_wait_program(prog)
                        .map_err(|e| CliError::Runtime(format!("vm_wait: {}", e)))
                })?;
                return Ok(Value::Float(0.0));
            }

            // vm_free_prog(program_id) → 0.0
            // Frees GPU resources for a completed VmProgram.
            // Call after vm_execute or vm_wait to prevent resource accumulation.
            if (name == "loom_free" || name == "vm_free_prog") && args.len() == 1 {
                let prog_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                // Remove from both maps — Drop on VmProgram calls vkDestroyPipeline
                VM_PROGRAMS.with(|progs| progs.borrow_mut().remove(&prog_id));
                VM_PROG_OWNERS.with(|owners| owners.borrow_mut().remove(&prog_id));
                return Ok(Value::Float(0.0));
            }

            // loom_status(vm_id) → pace_delay_ms (current homeostasis pacing)
            if name == "loom_status" && args.len() == 1 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let pace = LOOM_PACE_US.with(|p| *p.borrow().get(&vm_id).unwrap_or(&0));
                let dispatches = LOOM_DISPATCH_COUNT.with(|c| *c.borrow().get(&vm_id).unwrap_or(&0));
                let paced = LOOM_PACED_COUNT.with(|c| *c.borrow().get(&vm_id).unwrap_or(&0));
                let baseline = LOOM_BASELINE_US.with(|b| *b.borrow().get(&vm_id).unwrap_or(&0));
                // Return pace_delay_ms as primary value; use print for detail
                // Pack: pace_ms + dispatches*1e-6 for easy inspection
                let _ = (dispatches, paced, baseline); // available for future array return
                return Ok(Value::Float(pace as f32 / 1000.0));
            }

            // loom_pace(vm_id, pace_us) → 0.0
            // Manually set homeostasis pace delay in microseconds. 0 = disable.
            if name == "loom_pace" && args.len() == 2 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let pace_us = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u64;
                LOOM_PACE_US.with(|p| p.borrow_mut().insert(vm_id, pace_us));
                return Ok(Value::Float(0.0));
            }

            // loom_prefetch(path) → 0.0
            // Spawn background thread to read .spv file from disk into cache.
            if name == "loom_prefetch" && args.len() == 1 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str()?.to_string();
                let cached = SPIRV_FILE_CACHE.with(|c| c.borrow().contains_key(&path));
                if !cached {
                    let path_clone = path.clone();
                    let handle = std::thread::spawn(move || {
                        std::fs::read(&path_clone)
                            .map_err(|e| format!("loom_prefetch: {}", e))
                    });
                    SPIRV_PREFETCH.with(|pf| pf.borrow_mut().insert(path, handle));
                }
                return Ok(Value::Float(0.0));
            }

            // vm_poll_status(vm_id, instance) → status float (zero-copy HOST_VISIBLE read)
            // Reads Metrics[instance * 8 + 0]. Non-blocking, ~1μs.
            if name == "vm_poll_status" && args.len() == 2 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let instance = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let status = GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_poll_status: unknown VM {}", vm_id)))?;
                    octoflow_vulkan::vm::vm_poll_status(vm, instance)
                        .map_err(|e| CliError::Runtime(format!("vm_poll_status: {}", e)))
                })?;
                return Ok(Value::Float(status));
            }

            // vm_write_control_live(vm_id, offset, array_name) → 0.0
            // Direct HOST_VISIBLE write to Control SSBO (no staging, no fence).
            // Used by CPU poll loop to activate dormant VMs.
            if name == "vm_write_control_live" && args.len() == 3 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let arr_name = if let ScalarExpr::Ref(n) = &args[2] { n.clone() }
                    else { return Err(CliError::Compile("vm_write_control_live: 3rd arg must be array name".into())); };
                let data = gpu_array_get(&arr_name).unwrap_or_else(|| {
                    arrays.get(&arr_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_write_control_live: unknown VM {}", vm_id)))?;
                    octoflow_vulkan::vm::vm_write_control_live(vm, offset, &data)
                        .map_err(|e| CliError::Runtime(format!("vm_write_control_live: {}", e)))
                })?;
                return Ok(Value::Float(0.0));
            }

            // vm_write_control_u32(vm_id, offset, array_name) → 0.0
            // Direct HOST_VISIBLE write of uint32 values to Control SSBO.
            // Float values are truncated to u32 — for writing dispatch workgroup counts
            // that vkCmdDispatchIndirect reads as uint32.
            if name == "vm_write_control_u32" && args.len() == 3 {
                let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let arr_name = if let ScalarExpr::Ref(n) = &args[2] { n.clone() }
                    else { return Err(CliError::Compile("vm_write_control_u32: 3rd arg must be array name".into())); };
                let data = gpu_array_get(&arr_name).unwrap_or_else(|| {
                    arrays.get(&arr_name).map(|a| a.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()).unwrap_or_default()
                });
                GPU_VMS.with(|vms| {
                    let vms = vms.borrow();
                    let vm = vms.get(&vm_id).ok_or_else(||
                        CliError::Runtime(format!("vm_write_control_u32: unknown VM {}", vm_id)))?;
                    octoflow_vulkan::vm::vm_write_control_u32_live(vm, offset, &data)
                        .map_err(|e| CliError::Runtime(format!("vm_write_control_u32: {}", e)))
                })?;
                return Ok(Value::Float(0.0));
            }

            // gguf_prefetch_complete() — join prefetch thread, upload to caches
            if name == "gguf_prefetch_complete" && args.is_empty() {
                let handle = PREFETCH_THREAD.with(|pt| pt.borrow_mut().take());
                if let Some(h) = handle {
                    let results = h.join()
                        .map_err(|_| CliError::Runtime("gguf_prefetch_complete: thread panicked".into()))?
                        .map_err(|e| CliError::Runtime(format!("gguf_prefetch_complete: {}", e)))?;

                    // Atomic GPU upload — try ALL large tensors, commit or rollback
                    let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                    let mut gpu_resident = false;

                    if device_ptr != 0 {
                        let gpu = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                        let mut gpu_uploads: Vec<(String, octoflow_vulkan::GpuBuffer)> = Vec::new();
                        let mut all_ok = true;
                        for (cache_key, data, is_small) in &results {
                            if *is_small { continue; }
                            match octoflow_vulkan::upload_buffer(gpu, data) {
                                Ok(buf) => gpu_uploads.push((cache_key.clone(), buf)),
                                Err(_) => { all_ok = false; break; }
                            }
                        }
                        if all_ok && !gpu_uploads.is_empty() {
                            let mut added_bytes: u64 = 0;
                            for (key, buf) in gpu_uploads {
                                added_bytes += buf.len() as u64 * 4;
                                GPU_BUFFER_CACHE.with(|gc| gc.borrow_mut().insert(key, buf));
                            }
                            GPU_CACHE_BYTES.with(|c| c.set(c.get() + added_bytes));
                            gpu_resident = true;
                        }
                    }

                    // All tensors get CPU backup in TENSOR_CACHE
                    for (cache_key, data, _is_small) in results {
                        TENSOR_CACHE.with(|tc| tc.borrow_mut().insert(cache_key, data));
                    }

                    // Note: layer_idx not available here — LAYER_RESIDENCY set by caller if needed
                    let _ = gpu_resident;
                }
                return Ok(Value::Float(1.0));
            }
            // decomposed_load_layer(bin_path, model, layer_idx, virtual_path)
            // Synchronously loads all tensors for one layer from a decomposed .bin file.
            // Populates GPU_BUFFER_CACHE (large weights) and TENSOR_CACHE (norms/biases).
            // Cache keys use virtual_path so gguf_infer_layer finds 100% cache hits.
            if name == "decomposed_load_layer" && args.len() == 4 {
                let bin_path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("decomposed_load_layer: bin_path must be string".into()))?.to_string();
                check_read_permission_for(&bin_path)?;
                let model_name = if let ScalarExpr::Ref(name) = &args[1] {
                    name.clone()
                } else {
                    return Err(CliError::Compile("decomposed_load_layer: second arg must be map variable".into()));
                };
                let layer_idx = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;
                let virtual_path = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("decomposed_load_layer: virtual_path must be string".into()))?.to_string();

                let model_map = hashmaps.get(&model_name).ok_or_else(|| {
                    CliError::Compile(format!("decomposed_load_layer: map '{}' not found", model_name))
                })?;

                let li = layer_idx.to_string();

                // Build tensor list in decompose_gguf.flow packing order:
                // attn_norm, Q, K, V, O, ffn_norm, gate, up, down, [q_bias, k_bias, v_bias]
                let mut tensor_list: Vec<(String, bool)> = vec![
                    (format!("blk.{}.attn_norm.weight", li), true),
                    (format!("blk.{}.attn_q.weight", li), false),
                    (format!("blk.{}.attn_k.weight", li), false),
                    (format!("blk.{}.attn_v.weight", li), false),
                    (format!("blk.{}.attn_output.weight", li), false),
                    (format!("blk.{}.ffn_norm.weight", li), true),
                    (format!("blk.{}.ffn_gate.weight", li), false),
                    (format!("blk.{}.ffn_up.weight", li), false),
                    (format!("blk.{}.ffn_down.weight", li), false),
                ];
                let has_bias = model_map.contains_key(&format!("t.blk.{}.attn_q.bias.type", li));
                if has_bias {
                    tensor_list.push((format!("blk.{}.attn_q.bias", li), true));
                    tensor_list.push((format!("blk.{}.attn_k.bias", li), true));
                    tensor_list.push((format!("blk.{}.attn_v.bias", li), true));
                }

                // Compute cumulative byte offsets and collect metadata
                let mut file_offset: u64 = 0;
                // (cache_key, tensor_type, offset_in_bin, element_count, raw_byte_size, is_small, skip)
                let mut tensor_metas: Vec<(String, u32, u64, usize, usize, bool, bool)> = Vec::new();

                for (tname, is_small) in &tensor_list {
                    let prefix = format!("t.{}", tname);
                    let total_count = model_map.get(&format!("{}.count", prefix))
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                    if total_count == 0 { continue; }
                    let tensor_type = model_map.get(&format!("{}.type", prefix))
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as u32;
                    let byte_size: usize = match tensor_type {
                        0 => total_count * 4,
                        1 => total_count * 2,
                        12 => (total_count / 256) * 144,
                        13 => (total_count / 256) * 176,
                        14 => (total_count / 256) * 210,
                        _ => total_count * 4,
                    };
                    let cache_key = format!("{}:{}", virtual_path, tname);
                    let in_gpu = GPU_BUFFER_CACHE.with(|gc| gc.borrow().contains_key(&cache_key));
                    let in_tc = TENSOR_CACHE.with(|tc| tc.borrow().contains_key(&cache_key));
                    tensor_metas.push((cache_key, tensor_type, file_offset, total_count, byte_size, *is_small, in_gpu || in_tc));
                    file_offset += byte_size as u64;
                }

                // Streaming atomic load: dequant each tensor one at a time (saves RAM),
                // upload to GPU immediately, but track success for atomic rollback.
                // All tensors also get a CPU backup in TENSOR_CACHE (prevents split-brain).
                use std::io::{Read as _, Seek as _, SeekFrom};
                let mut file = std::fs::File::open(&bin_path)
                    .map_err(|e| CliError::Runtime(format!("decomposed_load_layer: open '{}': {}", bin_path, e)))?;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                let mut gpu_keys_uploaded: Vec<String> = Vec::new();
                let mut all_gpu_ok = true;
                let mut added_bytes: u64 = 0;

                for (cache_key, tensor_type, offset, total_count, byte_size, is_small, skip) in &tensor_metas {
                    if *skip { continue; }
                    file.seek(SeekFrom::Start(*offset))
                        .map_err(|e| CliError::Runtime(format!("decomposed_load_layer: seek: {}", e)))?;
                    let mut raw = vec![0u8; *byte_size];
                    file.read_exact(&mut raw)
                        .map_err(|e| CliError::Runtime(format!("decomposed_load_layer: read: {}", e)))?;

                    let dequanted: Vec<f32> = match *tensor_type {
                        0 => raw.chunks_exact(4).take(*total_count)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                        1 => raw.chunks_exact(2).take(*total_count)
                            .map(|c| gguf_f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                        12 => {
                            let mut out = Vec::with_capacity(*total_count + 256);
                            for block in raw.chunks(144) {
                                if block.len() < 144 { break; }
                                gguf_dequant_q4k_block(block, &mut out);
                            }
                            out.truncate(*total_count);
                            out
                        }
                        13 => {
                            let mut out = Vec::with_capacity(*total_count + 256);
                            for block in raw.chunks(176) {
                                if block.len() < 176 { break; }
                                gguf_dequant_q5k_block(block, &mut out);
                            }
                            out.truncate(*total_count);
                            out
                        }
                        14 => {
                            let mut out = Vec::with_capacity(*total_count + 256);
                            for block in raw.chunks(210) {
                                if block.len() < 210 { break; }
                                gguf_dequant_q6k_block(block, &mut out);
                            }
                            out.truncate(*total_count);
                            out
                        }
                        _ => return Err(CliError::Runtime(format!("decomposed_load_layer: unsupported type {}", tensor_type))),
                    };

                    // Always keep CPU backup in TENSOR_CACHE (prevents split-brain)
                    TENSOR_CACHE.with(|tc| tc.borrow_mut().insert(cache_key.clone(), dequanted));

                    // Try GPU upload for large tensors (streaming — one at a time, saves RAM)
                    if !*is_small && all_gpu_ok && device_ptr != 0 {
                        let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                        // Re-read from TENSOR_CACHE to upload (data was moved above)
                        let upload_ok = TENSOR_CACHE.with(|tc| {
                            let cache = tc.borrow();
                            if let Some(data) = cache.get(cache_key) {
                                match octoflow_vulkan::upload_buffer(gpu_dev, data) {
                                    Ok(buf) => {
                                        let buf_bytes = buf.len() as u64 * 4;
                                        GPU_BUFFER_CACHE.with(|gc| gc.borrow_mut().insert(cache_key.clone(), buf));
                                        added_bytes += buf_bytes;
                                        true
                                    }
                                    Err(_) => false,
                                }
                            } else { false }
                        });
                        if upload_ok {
                            gpu_keys_uploaded.push(cache_key.clone());
                        } else {
                            all_gpu_ok = false;
                        }
                    }
                }

                // Atomic rollback: if ANY large tensor GPU upload failed, evict ALL from GPU
                if !all_gpu_ok && !gpu_keys_uploaded.is_empty() {
                    for key in &gpu_keys_uploaded {
                        GPU_BUFFER_CACHE.with(|gc| gc.borrow_mut().remove(key));
                    }
                    added_bytes = 0;
                }
                GPU_CACHE_BYTES.with(|c| c.set(c.get() + added_bytes));

                let gpu_resident = all_gpu_ok && !gpu_keys_uploaded.is_empty();
                LAYER_RESIDENCY.with(|lr| lr.borrow_mut().insert(layer_idx, gpu_resident));
                return Ok(Value::Float(if gpu_resident { 1.0 } else { 0.0 }));
            }
            // decomposed_prefetch_layer(bin_path, model, layer_idx, virtual_path)
            // Async version — spawns background thread for file I/O + CPU dequant.
            // Use gguf_prefetch_complete() to join and upload to GPU.
            if name == "decomposed_prefetch_layer" && args.len() == 4 {
                let bin_path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("decomposed_prefetch_layer: bin_path must be string".into()))?.to_string();
                check_read_permission_for(&bin_path)?;
                let model_name = if let ScalarExpr::Ref(name) = &args[1] {
                    name.clone()
                } else {
                    return Err(CliError::Compile("decomposed_prefetch_layer: second arg must be map variable".into()));
                };
                let layer_idx = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;
                let virtual_path = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("decomposed_prefetch_layer: virtual_path must be string".into()))?.to_string();

                let model_map = hashmaps.get(&model_name).ok_or_else(|| {
                    CliError::Compile(format!("decomposed_prefetch_layer: map '{}' not found", model_name))
                })?;

                let li = layer_idx.to_string();

                // Same tensor order as decomposed_load_layer / decompose_gguf.flow
                let mut tensor_list: Vec<(String, bool)> = vec![
                    (format!("blk.{}.attn_norm.weight", li), true),
                    (format!("blk.{}.attn_q.weight", li), false),
                    (format!("blk.{}.attn_k.weight", li), false),
                    (format!("blk.{}.attn_v.weight", li), false),
                    (format!("blk.{}.attn_output.weight", li), false),
                    (format!("blk.{}.ffn_norm.weight", li), true),
                    (format!("blk.{}.ffn_gate.weight", li), false),
                    (format!("blk.{}.ffn_up.weight", li), false),
                    (format!("blk.{}.ffn_down.weight", li), false),
                ];
                let has_bias = model_map.contains_key(&format!("t.blk.{}.attn_q.bias.type", li));
                if has_bias {
                    tensor_list.push((format!("blk.{}.attn_q.bias", li), true));
                    tensor_list.push((format!("blk.{}.attn_k.bias", li), true));
                    tensor_list.push((format!("blk.{}.attn_v.bias", li), true));
                }

                // Compute cumulative offsets and collect metadata (main thread — no GPU access needed)
                let mut file_offset: u64 = 0;
                let mut tensor_metas: Vec<(String, u32, u64, usize, usize, bool)> = Vec::new();

                for (tname, is_small) in &tensor_list {
                    let prefix = format!("t.{}", tname);
                    let total_count = model_map.get(&format!("{}.count", prefix))
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                    if total_count == 0 { continue; }
                    let tensor_type = model_map.get(&format!("{}.type", prefix))
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as u32;
                    let byte_size: usize = match tensor_type {
                        0 => total_count * 4,
                        1 => total_count * 2,
                        12 => (total_count / 256) * 144,
                        13 => (total_count / 256) * 176,
                        14 => (total_count / 256) * 210,
                        _ => total_count * 4,
                    };
                    let cache_key = format!("{}:{}", virtual_path, tname);
                    let in_gpu = GPU_BUFFER_CACHE.with(|gc| gc.borrow().contains_key(&cache_key));
                    let in_tc = TENSOR_CACHE.with(|tc| tc.borrow().contains_key(&cache_key));
                    if in_gpu || in_tc {
                        // Still need to advance file_offset for subsequent tensors
                        file_offset += byte_size as u64;
                        continue;
                    }
                    tensor_metas.push((cache_key, tensor_type, file_offset, total_count, byte_size, *is_small));
                    file_offset += byte_size as u64;
                }

                if tensor_metas.is_empty() {
                    return Ok(Value::Float(1.0)); // All already cached
                }

                let bin_path_clone = bin_path.clone();
                let handle = std::thread::spawn(move || -> Result<Vec<(String, Vec<f32>, bool)>, String> {
                    use std::io::{Read, Seek, SeekFrom};
                    let mut file = std::fs::File::open(&bin_path_clone)
                        .map_err(|e| format!("decomposed_prefetch: open '{}': {}", bin_path_clone, e))?;
                    let mut results = Vec::with_capacity(tensor_metas.len());

                    for (cache_key, tensor_type, offset, total_count, byte_size, is_small) in tensor_metas {
                        file.seek(SeekFrom::Start(offset))
                            .map_err(|e| format!("decomposed_prefetch: seek: {}", e))?;
                        let mut raw = vec![0u8; byte_size];
                        file.read_exact(&mut raw)
                            .map_err(|e| format!("decomposed_prefetch: read: {}", e))?;

                        let dequanted: Vec<f32> = match tensor_type {
                            0 => raw.chunks_exact(4).take(total_count)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                            1 => raw.chunks_exact(2).take(total_count)
                                .map(|c| gguf_f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                            12 => {
                                let mut out = Vec::with_capacity(total_count + 256);
                                for block in raw.chunks(144) {
                                    if block.len() < 144 { break; }
                                    gguf_dequant_q4k_block(block, &mut out);
                                }
                                out.truncate(total_count);
                                out
                            }
                            13 => {
                                let mut out = Vec::with_capacity(total_count + 256);
                                for block in raw.chunks(176) {
                                    if block.len() < 176 { break; }
                                    gguf_dequant_q5k_block(block, &mut out);
                                }
                                out.truncate(total_count);
                                out
                            }
                            14 => {
                                let mut out = Vec::with_capacity(total_count + 256);
                                for block in raw.chunks(210) {
                                    if block.len() < 210 { break; }
                                    gguf_dequant_q6k_block(block, &mut out);
                                }
                                out.truncate(total_count);
                                out
                            }
                            _ => return Err(format!("decomposed_prefetch: unsupported type {}", tensor_type)),
                        };
                        results.push((cache_key, dequanted, is_small));
                    }
                    Ok(results)
                });

                PREFETCH_THREAD.with(|pt| *pt.borrow_mut() = Some(handle));
                return Ok(Value::Float(1.0));
            }
            // gguf_extract_tensor_raw(gguf_path, model, tensor_name, output_path)
            // Copies raw (still-quantized) tensor bytes from GGUF to output file.
            if name == "gguf_extract_tensor_raw" && args.len() == 4 {
                let gguf_path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("gguf_extract_tensor_raw: path must be string".into()))?.to_string();
                check_read_permission_for(&gguf_path)?;
                let model_name = if let ScalarExpr::Ref(name) = &args[1] {
                    name.clone()
                } else {
                    return Err(CliError::Compile("gguf_extract_tensor_raw: second arg must be map variable".into()));
                };
                let tname = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("gguf_extract_tensor_raw: tensor name must be string".into()))?.to_string();
                let out_path = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("gguf_extract_tensor_raw: output path must be string".into()))?.to_string();
                check_write_permission_for(&out_path)?;

                let model_map = hashmaps.get(&model_name).ok_or_else(|| {
                    CliError::Compile(format!("gguf_extract_tensor_raw: map '{}' not found", model_name))
                })?;
                let prefix = format!("t.{}", tname);
                let total_count = model_map.get(&format!("{}.count", prefix))
                    .and_then(|v| v.as_float().ok())
                    .unwrap_or(0.0) as usize;
                if total_count == 0 {
                    return Err(CliError::Runtime(format!("gguf_extract_tensor_raw: tensor '{}' not found or zero count", tname)));
                }
                let tensor_type = model_map.get(&format!("{}.type", prefix))
                    .and_then(|v| v.as_float().ok())
                    .unwrap_or(0.0) as u32;
                let byte_size = match tensor_type {
                    0 => total_count * 4,
                    1 => total_count * 2,
                    12 => (total_count / 256) * 144,
                    14 => (total_count / 256) * 210,
                    _ => total_count * 4,
                };
                let ds_buf = model_map.get("_ds_buf")
                    .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                let hdr_buf = model_map.get("_hdr_buf")
                    .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                let off_pos = model_map.get(&format!("{}.off_pos", prefix))
                    .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                let ds_ptr = mem_table_get_ptr(ds_buf)?;
                let data_start = unsafe { (ds_ptr as *const u64).read_unaligned() };
                let hdr_ptr = mem_table_get_ptr(hdr_buf)?;
                let tensor_offset = unsafe { (hdr_ptr.add(off_pos) as *const u64).read_unaligned() };
                let file_offset = data_start + tensor_offset;

                // Read raw bytes from GGUF file
                use std::io::{Read, Seek, SeekFrom};
                let mut file = std::fs::File::open(&gguf_path)
                    .map_err(|e| CliError::Io(format!("gguf_extract_tensor_raw: open {}: {}", gguf_path, e)))?;
                file.seek(SeekFrom::Start(file_offset))
                    .map_err(|e| CliError::Io(format!("gguf_extract_tensor_raw: seek: {}", e)))?;
                let mut buf = vec![0u8; byte_size];
                file.read_exact(&mut buf)
                    .map_err(|e| CliError::Io(format!("gguf_extract_tensor_raw: read: {}", e)))?;

                // Write to output file (append mode for concatenation)
                use std::io::Write;
                let mut out_file = std::fs::OpenOptions::new()
                    .create(true).append(true).open(&out_path)
                    .map_err(|e| CliError::Io(format!("gguf_extract_tensor_raw: create {}: {}", out_path, e)))?;
                out_file.write_all(&buf)
                    .map_err(|e| CliError::Io(format!("gguf_extract_tensor_raw: write: {}", e)))?;

                return Ok(Value::Float(byte_size as f32));
            }
            // ── Staging buffer builtins for double-buffered streaming ──
            // rt_staging_alloc(size_bytes) → handle
            if name == "rt_staging_alloc" && args.len() == 1 {
                let size = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;
                let handle_id = STAGING_NEXT_ID.with(|id| {
                    let v = id.get();
                    id.set(v + 1);
                    v
                });
                STAGING_HANDLES.with(|sh| {
                    sh.borrow_mut().insert(handle_id, StagingHandle {
                        data: Vec::new(),
                        size_bytes: size,
                        io_thread: None,
                    });
                });
                return Ok(Value::Float(handle_id as f32));
            }
            // rt_staging_load(handle, path, offset, count) → 0.0 (async, returns immediately)
            // Spawns background thread to read raw bytes from file at offset, interpret as f32.
            // Call rt_staging_wait() to block until complete, or rt_staging_ready() to poll.
            if name == "rt_staging_load" && args.len() == 4 {
                let handle_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let path = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("rt_staging_load: path must be string".into()))?.to_string();
                check_read_permission_for(&path)?;
                let offset = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u64;
                let count = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as usize;
                // Spawn background thread for async file I/O
                let thread = std::thread::spawn(move || -> Result<Vec<f32>, String> {
                    use std::io::{Read as _, Seek as _, SeekFrom};
                    let mut file = std::fs::File::open(&path)
                        .map_err(|e| format!("rt_staging_load: {}", e))?;
                    file.seek(SeekFrom::Start(offset))
                        .map_err(|e| format!("rt_staging_load: seek: {}", e))?;
                    let byte_count = count * 4;
                    let mut raw = vec![0u8; byte_count];
                    file.read_exact(&mut raw)
                        .map_err(|e| format!("rt_staging_load: read {} bytes: {}", byte_count, e))?;
                    let floats: Vec<f32> = raw.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Ok(floats)
                });
                STAGING_HANDLES.with(|sh| -> Result<(), CliError> {
                    let mut handles = sh.borrow_mut();
                    let h = handles.get_mut(&handle_id).ok_or_else(||
                        CliError::Runtime(format!("rt_staging_load: invalid handle {}", handle_id)))?;
                    h.io_thread = Some(thread);
                    Ok(())
                })?;
                return Ok(Value::Float(0.0));
            }
            // rt_staging_upload(handle, dest_cache_key) → 1.0
            // Uploads staging CPU data to GPU_BUFFER_CACHE under the given key.
            if name == "rt_staging_upload" && args.len() == 2 {
                let handle_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let dest_key = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("rt_staging_upload: dest must be string".into()))?.to_string();
                let data = STAGING_HANDLES.with(|sh| -> Result<Vec<f32>, CliError> {
                    let handles = sh.borrow();
                    let h = handles.get(&handle_id).ok_or_else(||
                        CliError::Runtime(format!("rt_staging_upload: invalid handle {}", handle_id)))?;
                    Ok(h.data.clone())
                })?;
                let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
                if device_ptr == 0 {
                    return Err(CliError::Runtime("rt_staging_upload: no Vulkan GPU available. Loom VM requires a GPU".into()));
                }
                let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
                let new_buf = octoflow_vulkan::upload_buffer(gpu_dev, &data)
                    .map_err(|e| CliError::Gpu(format!("rt_staging_upload: {}", e)))?;
                GPU_BUFFER_CACHE.with(|gc| gc.borrow_mut().insert(dest_key, new_buf));
                return Ok(Value::Float(1.0));
            }
            // rt_staging_ready(handle) → 1.0 (ready) or 0.0 (I/O still in progress)
            if name == "rt_staging_ready" && args.len() == 1 {
                let handle_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let ready = STAGING_HANDLES.with(|sh| -> Result<bool, CliError> {
                    let handles = sh.borrow();
                    let h = handles.get(&handle_id).ok_or_else(||
                        CliError::Runtime(format!("rt_staging_ready: invalid handle {}", handle_id)))?;
                    Ok(h.io_thread.as_ref().map_or(true, |t| t.is_finished()))
                })?;
                return Ok(Value::Float(if ready { 1.0 } else { 0.0 }));
            }
            // rt_staging_wait(handle) → float_count (blocks until async I/O completes)
            if name == "rt_staging_wait" && args.len() == 1 {
                let handle_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                let count = STAGING_HANDLES.with(|sh| -> Result<f32, CliError> {
                    let mut handles = sh.borrow_mut();
                    let h = handles.get_mut(&handle_id).ok_or_else(||
                        CliError::Runtime(format!("rt_staging_wait: invalid handle {}", handle_id)))?;
                    if let Some(thread) = h.io_thread.take() {
                        let result = thread.join()
                            .map_err(|_| CliError::Runtime("rt_staging_wait: I/O thread panicked".into()))?
                            .map_err(|e| CliError::Io(format!("rt_staging_wait: {}", e)))?;
                        let n = result.len() as f32;
                        h.data = result;
                        Ok(n)
                    } else {
                        // No pending I/O — return current data length
                        Ok(h.data.len() as f32)
                    }
                })?;
                return Ok(Value::Float(count));
            }
            // rt_staging_free(handle) → 1.0
            if name == "rt_staging_free" && args.len() == 1 {
                let handle_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_float()? as u32;
                STAGING_HANDLES.with(|sh| {
                    if let Some(mut h) = sh.borrow_mut().remove(&handle_id) {
                        // Join any pending I/O thread before dropping
                        if let Some(thread) = h.io_thread.take() {
                            let _ = thread.join();
                        }
                    }
                });
                return Ok(Value::Float(1.0));
            }
            // is_directory(path) → 1.0 or 0.0
            if name == "is_directory" && args.len() == 1 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("is_directory() path must be a string".into()))?;
                check_read_permission_for(path)?;
                return Ok(Value::Float(if std::path::Path::new(path).is_dir() { 1.0 } else { 0.0 }));
            }
            // ── Path utility functions (no security check — pure string ops) ──
            // file_ext(path) → extension string (e.g., "csv")
            if name == "file_ext" && args.len() == 1 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("file_ext() path must be a string".into()))?;
                let ext = std::path::Path::new(path)
                    .extension()
                    .map(|e| e.to_string_lossy().into_owned())
                    .unwrap_or_default();
                return Ok(Value::Str(ext));
            }
            // file_name(path) → filename string (e.g., "data.csv")
            if name == "file_name" && args.len() == 1 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("file_name() path must be a string".into()))?;
                let fname = std::path::Path::new(path)
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_default();
                return Ok(Value::Str(fname));
            }
            // file_dir(path) → parent directory string
            if name == "file_dir" && args.len() == 1 {
                let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = path_val.as_str().map_err(|_| CliError::Compile("file_dir() path must be a string".into()))?;
                let dir = std::path::Path::new(path)
                    .parent()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
                return Ok(Value::Str(dir));
            }
            // path_join(dir, file) → joined path string
            if name == "path_join" && args.len() == 2 {
                let dir_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let file_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let dir = dir_val.as_str().map_err(|_| CliError::Compile("path_join() dir must be a string".into()))?;
                let file = file_val.as_str().map_err(|_| CliError::Compile("path_join() file must be a string".into()))?;
                let joined = std::path::Path::new(dir).join(file).to_string_lossy().into_owned();
                return Ok(Value::Str(joined));
            }
            // ── Array scalar functions (Phase 33) ──────────────────────
            // join(arr, delimiter) → string
            if name == "join" && args.len() == 2 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    let delim_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let delim = delim_val.as_str().map_err(|_| CliError::Compile("join() delimiter must be a string".into()))?;
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile(format!("join() requires array, '{}' not found", arr_name)))?;
                    let parts: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                    return Ok(Value::Str(parts.join(delim)));
                }
                return Err(CliError::Compile("join() first argument must be an array name".into()));
            }
            // find(arr, value) → index or -1
            if name == "find" && args.len() == 2 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    let target = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile(format!("find() requires array, '{}' not found", arr_name)))?;
                    for (i, val) in arr.iter().enumerate() {
                        let matches = match (val, &target) {
                            (Value::Float(a), Value::Float(b)) => (a - b).abs() < 1e-6,
                            (Value::Str(a), Value::Str(b)) => a == b,
                            _ => false,
                        };
                        if matches {
                            return Ok(Value::Float(i as f32));
                        }
                    }
                    return Ok(Value::Float(-1.0));
                }
                return Err(CliError::Compile("find() first argument must be an array name".into()));
            }
            // first(arr) → first element (error on empty)
            if name == "first" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile(format!("first() requires array, '{}' not found", arr_name)))?;
                    return arr.first().cloned()
                        .ok_or_else(|| CliError::Compile(format!("first() called on empty array '{}'", arr_name)));
                }
                return Err(CliError::Compile("first() argument must be an array name".into()));
            }
            // last(arr) → last element (error on empty)
            if name == "last" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile(format!("last() requires array, '{}' not found", arr_name)))?;
                    return arr.last().cloned()
                        .ok_or_else(|| CliError::Compile(format!("last() called on empty array '{}'", arr_name)));
                }
                return Err(CliError::Compile("last() argument must be an array name".into()));
            }
            // type_of(val) → "float" or "string"
            if name == "type_of" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                return Ok(Value::Str(match val {
                    Value::Float(_) => "float".to_string(),
                    Value::Int(_) => "int".to_string(),
                    Value::Str(_) => "string".to_string(),
                    Value::Map(_) => "map".to_string(),
                    Value::None => "none".to_string(),
                }));
            }
            // is_none(val) → 1.0 if val is none, 0.0 otherwise
            if name == "is_none" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                return Ok(Value::Float(if val.is_none() { 1.0 } else { 0.0 }));
            }
            // is_nan(val) → 1.0 if val is NaN, 0.0 otherwise
            if name == "is_nan" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                return Ok(Value::Float(match val {
                    Value::Float(f) => if f.is_nan() { 1.0 } else { 0.0 },
                    Value::Int(_) => 0.0, // integers are never NaN
                    _ => 0.0,
                }));
            }
            // is_inf(val) → 1.0 if val is infinite, 0.0 otherwise
            if name == "is_inf" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                return Ok(Value::Float(match val {
                    Value::Float(f) => if f.is_infinite() { 1.0 } else { 0.0 },
                    Value::Int(_) => 0.0, // integers are never infinite
                    _ => 0.0,
                }));
            }
            // min_val(arr) → minimum value in array (floats only)
            if name == "min_val" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile(format!("min_val() requires array, '{}' not found", arr_name)))?;
                    if arr.is_empty() {
                        return Err(CliError::Compile(format!("min_val() called on empty array '{}'", arr_name)));
                    }
                    let mut min = f32::INFINITY;
                    for val in arr {
                        let f = val.as_float().map_err(|_| CliError::Compile("min_val() requires all elements to be floats".into()))?;
                        if f < min { min = f; }
                    }
                    return Ok(Value::Float(min));
                }
                return Err(CliError::Compile("min_val() argument must be an array name".into()));
            }
            // max_val(arr) → maximum value in array (floats only)
            if name == "max_val" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile(format!("max_val() requires array, '{}' not found", arr_name)))?;
                    if arr.is_empty() {
                        return Err(CliError::Compile(format!("max_val() called on empty array '{}'", arr_name)));
                    }
                    let mut max = f32::NEG_INFINITY;
                    for val in arr {
                        let f = val.as_float().map_err(|_| CliError::Compile("max_val() requires all elements to be floats".into()))?;
                        if f > max { max = f; }
                    }
                    return Ok(Value::Float(max));
                }
                return Err(CliError::Compile("max_val() argument must be an array name".into()));
            }
            // ── Type conversion functions (Phase 32) ─────────────────
            // str(val) → convert any value to string
            if name == "str" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                return Ok(match val {
                    Value::Float(f) => {
                        // Format nicely: 3.0 → "3", 3.14 → "3.14"
                        if f == f.trunc() && f.is_finite() {
                            Value::Str(format!("{}", f as i64))
                        } else {
                            Value::Str(format!("{}", f))
                        }
                    }
                    Value::Int(i) => Value::Str(format!("{}", i)),
                    Value::Str(_) => val,
                    Value::Map(m) => Value::Str(format!("{}", Value::Map(m))),
                    Value::None => Value::Str("none".to_string()),
                });
            }
            // float(val) → parse string to float, or pass through float
            if name == "float" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                return Ok(match val {
                    Value::Float(_) => val,
                    Value::Int(i) => Value::Float(i as f32),
                    Value::Str(s) => {
                        let f: f32 = s.trim().parse()
                            .map_err(|_| CliError::Compile(format!("float() cannot parse '{}' as number", s)))?;
                        Value::Float(f)
                    }
                    Value::Map(_) => return Err(CliError::Compile("float() cannot convert map to number".into())),
                    Value::None => return Err(CliError::Compile("float() cannot convert none to number".into())),
                });
            }
            // int(val) → truncate to integer (floor toward zero)
            if name == "int" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                return Ok(match val {
                    Value::Float(f) => Value::Int(f.trunc() as i64),
                    Value::Int(_) => val,
                    Value::Str(s) => {
                        if let Ok(i) = s.trim().parse::<i64>() {
                            Value::Int(i)
                        } else {
                            let f: f32 = s.trim().parse()
                                .map_err(|_| CliError::Compile(format!("int() cannot parse '{}' as number", s)))?;
                            Value::Int(f.trunc() as i64)
                        }
                    }
                    Value::Map(_) => return Err(CliError::Compile("int() cannot convert map to number".into())),
                    Value::None => return Err(CliError::Compile("int() cannot convert none to number".into())),
                });
            }
            // ── String operations (Phase 32) ──────────────────────────
            // substr(s, start, len) → extract substring
            if name == "substr" && args.len() == 3 {
                let s_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let start_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let len_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = s_val.as_str().map_err(|_| CliError::Compile("substr() first argument must be a string".into()))?;
                let start = start_val.as_float()? as usize;
                let length = len_val.as_float()? as usize;
                let chars: Vec<char> = s.chars().collect();
                let end = (start + length).min(chars.len());
                let start = start.min(chars.len());
                return Ok(Value::Str(chars[start..end].iter().collect()));
            }
            // replace(s, old, new) → replace all occurrences
            if name == "replace" && args.len() == 3 {
                let s_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let old_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let new_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = s_val.as_str().map_err(|_| CliError::Compile("replace() first argument must be a string".into()))?;
                let old = old_val.as_str().map_err(|_| CliError::Compile("replace() second argument must be a string".into()))?;
                let new = new_val.as_str().map_err(|_| CliError::Compile("replace() third argument must be a string".into()))?;
                return Ok(Value::Str(s.replace(old, new)));
            }
            // trim(s) → strip leading/trailing whitespace
            if name == "trim" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = val.as_str().map_err(|_| CliError::Compile("trim() argument must be a string".into()))?;
                return Ok(Value::Str(s.trim().to_string()));
            }
            // to_upper(s) → uppercase
            if name == "to_upper" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = val.as_str().map_err(|_| CliError::Compile("to_upper() argument must be a string".into()))?;
                return Ok(Value::Str(s.to_uppercase()));
            }
            // to_lower(s) → lowercase
            if name == "to_lower" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = val.as_str().map_err(|_| CliError::Compile("to_lower() argument must be a string".into()))?;
                return Ok(Value::Str(s.to_lowercase()));
            }
            // tokenize(text) → count (tokens via RETURNED_ARRAY)
            // split_whitespace → to_lowercase → trim punctuation → filter len > 1
            if name == "tokenize" && args.len() == 1 {
                let text = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("tokenize() argument must be a string".into()))?.to_string();
                let mut tokens = Vec::new();
                for word in text.split_whitespace() {
                    let lower = word.to_lowercase();
                    let trimmed = lower.trim_matches(|c: char| c.is_ascii_punctuation());
                    if trimmed.len() > 1 {
                        tokens.push(Value::Str(trimmed.to_string()));
                    }
                }
                let count = tokens.len();
                if count > 0 {
                    RETURNED_ARRAY.with(|r| {
                        *r.borrow_mut() = Some(tokens);
                    });
                }
                return Ok(Value::Float(count as f32));
            }

            // starts_with(s, prefix) → 1.0 or 0.0
            if name == "starts_with" && args.len() == 2 {
                let s_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let prefix_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = s_val.as_str().map_err(|_| CliError::Compile("starts_with() first argument must be a string".into()))?;
                let prefix = prefix_val.as_str().map_err(|_| CliError::Compile("starts_with() second argument must be a string".into()))?;
                return Ok(Value::Float(if s.starts_with(prefix) { 1.0 } else { 0.0 }));
            }
            // ends_with(s, suffix) → 1.0 or 0.0
            if name == "ends_with" && args.len() == 2 {
                let s_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let suffix_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = s_val.as_str().map_err(|_| CliError::Compile("ends_with() first argument must be a string".into()))?;
                let suffix = suffix_val.as_str().map_err(|_| CliError::Compile("ends_with() second argument must be a string".into()))?;
                return Ok(Value::Float(if s.ends_with(suffix) { 1.0 } else { 0.0 }));
            }
            // float_byte(f, index) → byte at position index (0-3) of IEEE 754 bits
            // Avoids f32 precision loss for large u32 values (> 2^24)
            // float_byte(1.0, 3) == 63 (0x3F), float_byte(1.0, 0) == 0
            if name == "float_byte" && args.len() == 2 {
                let f_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let idx_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let f = f_val.as_float()?;
                let idx = idx_val.as_float()? as usize;
                let bits = f.to_bits();
                let byte_val = ((bits >> (idx * 8)) & 0xFF) as f32;
                return Ok(Value::Float(byte_val));
            }
            // index_of(s, needle) → position or -1
            if name == "index_of" && args.len() == 2 {
                let s_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let needle_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = s_val.as_str().map_err(|_| CliError::Compile("index_of() first argument must be a string".into()))?;
                let needle = needle_val.as_str().map_err(|_| CliError::Compile("index_of() second argument must be a string".into()))?;
                return Ok(Value::Float(match s.find(needle) {
                    Some(pos) => pos as f32,
                    None => -1.0,
                }));
            }
            // char_at(s, index) → single character string
            if name == "char_at" && args.len() == 2 {
                let s_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let idx_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = s_val.as_str().map_err(|_| CliError::Compile("char_at() first argument must be a string".into()))?;
                let idx = idx_val.as_float()? as usize;
                let chars: Vec<char> = s.chars().collect();
                if idx >= chars.len() {
                    return Err(CliError::Compile(format!("char_at() index {} out of bounds (string length {})", idx, chars.len())));
                }
                return Ok(Value::Str(chars[idx].to_string()));
            }
            // ord(c) → Unicode code point of first character (f32)
            // Enables character comparison without string escape sequences.
            // ord("A") == 65.0, ord(" ") == 32.0, ord("\n") == 10.0
            if name == "ord" && args.len() == 1 {
                let s_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = s_val.as_str().map_err(|_| CliError::Compile("ord() argument must be a string".into()))?;
                let ch = s.chars().next().ok_or_else(|| CliError::Compile("ord() called on empty string".into()))?;
                return Ok(Value::Float(ch as u32 as f32));
            }
            // chr(n) → string containing the character with Unicode code point n
            // chr(65.0) == "A", chr(10.0) == newline, chr(34.0) == double-quote
            if name == "chr" && args.len() == 1 {
                let n_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let n = n_val.as_float()? as u32;
                let ch = char::from_u32(n).ok_or_else(|| CliError::Compile(format!("chr() invalid code point: {}", n)))?;
                return Ok(Value::Str(ch.to_string()));
            }
            // float_to_bits(f) → IEEE 754 bit representation of f32 as f32
            // Enables SPIR-V binary emission from .flow (self-hosting GPU pipeline)
            // float_to_bits(1.0) == 1065353216.0 (0x3F800000)
            if name == "float_to_bits" && args.len() == 1 {
                let f_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let f = f_val.as_float()?;
                let bits = f.to_bits();
                return Ok(Value::Float(bits as f32));
            }
            // bits_to_float(n) → reinterpret u32 bits as IEEE 754 f32
            // bits_to_float(1065353216.0) == 1.0
            if name == "bits_to_float" && args.len() == 1 {
                let n_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let n = n_val.as_float()? as u32;
                return Ok(Value::Float(f32::from_bits(n)));
            }
            // ── mem_* builtins (Phase 72a) — raw memory for FFI struct construction ──
            // All require --allow-ffi. Handle-based: f32 index into MEM_TABLE.
            if name == "mem_alloc" && args.len() == 1 {
                check_ffi_permission()?;
                let size_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let size = size_val.as_float()? as usize;
                return Ok(Value::Float(mem_table_alloc(size)?));
            }
            if name == "mem_free" && args.len() == 1 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                mem_table_free(h)?;
                return Ok(Value::Float(0.0));
            }
            if name == "mem_size" && args.len() == 1 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                return Ok(Value::Float(mem_table_get_size(h)? as f32));
            }
            if name == "mem_set_u32" && args.len() == 3 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let off_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let val_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                let off = off_val.as_float()? as usize;
                let val = val_val.as_float()? as u32;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                if size > 0 && off + 4 > size {
                    return Err(CliError::Runtime(format!("mem_set_u32: offset {} + 4 > size {}", off, size)));
                }
                unsafe { (ptr.add(off) as *mut u32).write_unaligned(val); }
                return Ok(Value::Float(0.0));
            }
            if name == "mem_set_f32" && args.len() == 3 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let off_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let val_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                let off = off_val.as_float()? as usize;
                let val = val_val.as_float()?;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                if size > 0 && off + 4 > size {
                    return Err(CliError::Runtime(format!("mem_set_f32: offset {} + 4 > size {}", off, size)));
                }
                unsafe { (ptr.add(off) as *mut f32).write_unaligned(val); }
                return Ok(Value::Float(0.0));
            }
            if name == "mem_set_ptr" && args.len() == 3 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let off_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let src_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                let off = off_val.as_float()? as usize;
                let src_h = src_val.as_float()? as isize;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                let ptr_size = std::mem::size_of::<usize>();
                if size > 0 && off + ptr_size > size {
                    return Err(CliError::Runtime(format!("mem_set_ptr: offset {} + {} > size {}", off, ptr_size, size)));
                }
                let src_ptr = if src_h <= 0 { std::ptr::null::<u8>() } else { mem_table_get_ptr(src_h as usize)? };
                unsafe { (ptr.add(off) as *mut usize).write_unaligned(src_ptr as usize); }
                return Ok(Value::Float(0.0));
            }
            if name == "mem_get_u32" && args.len() == 2 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let off_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                let off = off_val.as_float()? as usize;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                if size > 0 && off + 4 > size {
                    return Err(CliError::Runtime(format!("mem_get_u32: offset {} + 4 > size {}", off, size)));
                }
                let val = unsafe { (ptr.add(off) as *const u32).read_unaligned() };
                return Ok(Value::Float(val as f32));
            }
            if name == "mem_get_f32" && args.len() == 2 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let off_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                let off = off_val.as_float()? as usize;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                if size > 0 && off + 4 > size {
                    return Err(CliError::Runtime(format!("mem_get_f32: offset {} + 4 > size {}", off, size)));
                }
                let val = unsafe { (ptr.add(off) as *const f32).read_unaligned() };
                return Ok(Value::Float(val));
            }
            if name == "mem_get_ptr" && args.len() == 2 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let off_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                let off = off_val.as_float()? as usize;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                let ptr_size = std::mem::size_of::<usize>();
                if size > 0 && off + ptr_size > size {
                    return Err(CliError::Runtime(format!("mem_get_ptr: offset {} + {} > size {}", off, ptr_size, size)));
                }
                let raw = unsafe { (ptr.add(off) as *const usize).read_unaligned() };
                return Ok(Value::Float(mem_table_store_external(raw as *mut u8)));
            }
            if name == "mem_copy" && args.len() == 5 {
                check_ffi_permission()?;
                let sh_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let so_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let dh_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let do_val = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let n_val = eval_scalar(&args[4], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let sh = sh_val.as_float()? as usize;
                let so = so_val.as_float()? as usize;
                let dh = dh_val.as_float()? as usize;
                let d_off = do_val.as_float()? as usize;
                let n = n_val.as_float()? as usize;
                let src_ptr = mem_table_get_ptr(sh)?;
                let src_size = mem_table_get_size(sh)?;
                let dst_ptr = mem_table_get_ptr(dh)?;
                let dst_size = mem_table_get_size(dh)?;
                if src_size > 0 && so + n > src_size {
                    return Err(CliError::Runtime(format!("mem_copy: src offset {} + {} > size {}", so, n, src_size)));
                }
                if dst_size > 0 && d_off + n > dst_size {
                    return Err(CliError::Runtime(format!("mem_copy: dst offset {} + {} > size {}", d_off, n, dst_size)));
                }
                unsafe { std::ptr::copy_nonoverlapping(src_ptr.add(so), dst_ptr.add(d_off), n); }
                return Ok(Value::Float(0.0));
            }

            // ── Phase 72b: additional mem_* builtins ──
            if name == "mem_set_u8" && args.len() == 3 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let off_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let val_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                let off = off_val.as_float()? as usize;
                let val = val_val.as_float()? as u8;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                if size > 0 && off >= size {
                    return Err(CliError::Runtime(format!("mem_set_u8: offset {} >= size {}", off, size)));
                }
                unsafe { *ptr.add(off) = val; }
                return Ok(Value::Float(0.0));
            }
            if name == "mem_get_u8" && args.len() == 2 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let off_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                let off = off_val.as_float()? as usize;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                if size > 0 && off >= size {
                    return Err(CliError::Runtime(format!("mem_get_u8: offset {} >= size {}", off, size)));
                }
                let val = unsafe { *ptr.add(off) };
                return Ok(Value::Float(val as f32));
            }
            if name == "mem_set_u64" && args.len() == 3 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let off_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let val_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                let off = off_val.as_float()? as usize;
                let val = val_val.as_float()? as u64;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                if size > 0 && off + 8 > size {
                    return Err(CliError::Runtime(format!("mem_set_u64: offset {} + 8 > size {}", off, size)));
                }
                unsafe { (ptr.add(off) as *mut u64).write_unaligned(val); }
                return Ok(Value::Float(0.0));
            }
            if name == "mem_get_u64" && args.len() == 2 {
                check_ffi_permission()?;
                let h_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let off_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h = h_val.as_float()? as usize;
                let off = off_val.as_float()? as usize;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                if size > 0 && off + 8 > size {
                    return Err(CliError::Runtime(format!("mem_get_u64: offset {} + 8 > size {}", off, size)));
                }
                let val = unsafe { (ptr.add(off) as *const u64).read_unaligned() };
                return Ok(Value::Float(val as f32));
            }
            if name == "mem_from_str" && args.len() == 1 {
                check_ffi_permission()?;
                let s_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = s_val.as_str().map_err(|_| CliError::Compile("mem_from_str() argument must be a string".into()))?;
                let bytes = s.as_bytes();
                let h = mem_table_alloc(bytes.len() + 1)?; // +1 for null terminator
                let ptr = mem_table_get_ptr(h as usize)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
                    *ptr.add(bytes.len()) = 0; // null terminate
                }
                return Ok(Value::Float(h));
            }
            if name == "mem_to_str" && args.len() == 2 {
                check_ffi_permission()?;
                let h = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let len = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let s = MEM_TABLE.with(|t| {
                    let table = t.borrow();
                    match table.get(h).and_then(|o| o.as_ref()) {
                        Some(b) => {
                            if len > b.size {
                                Err(CliError::Runtime(format!("mem_to_str: length {} exceeds buffer size {}", len, b.size)))
                            } else {
                                let slice = unsafe { std::slice::from_raw_parts(b.ptr, len) };
                                Ok(String::from_utf8_lossy(slice).into_owned())
                            }
                        }
                        None => Err(CliError::Runtime(format!("mem_to_str: invalid handle {}", h))),
                    }
                })?;
                return Ok(Value::Str(s));
            }

            // mem_to_str_at(handle, offset, length) — read UTF-8 string at arbitrary offset
            if name == "mem_to_str_at" && args.len() == 3 {
                check_ffi_permission()?;
                let h = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let off = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let len = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let ptr = mem_table_get_ptr(h)?;
                let size = mem_table_get_size(h)?;
                if size > 0 && off + len > size {
                    return Err(CliError::Runtime(format!("mem_to_str_at: offset {} + {} > size {}", off, len, size)));
                }
                let slice = unsafe { std::slice::from_raw_parts(ptr.add(off), len) };
                let s = String::from_utf8_lossy(slice).into_owned();
                return Ok(Value::Str(s));
            }
            // mem_u64_add(dest_handle, dest_pos, src_handle, src_pos) — u64 addition in MEM_TABLE
            if name == "mem_u64_add" && args.len() == 4 {
                check_ffi_permission()?;
                let dh = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let dp = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let sh = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let sp = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let d_ptr = mem_table_get_ptr(dh)?;
                let s_ptr = mem_table_get_ptr(sh)?;
                let d_size = mem_table_get_size(dh)?;
                let s_size = mem_table_get_size(sh)?;
                if d_size > 0 && dp + 8 > d_size {
                    return Err(CliError::Runtime(format!("mem_u64_add: dest offset {} + 8 > size {}", dp, d_size)));
                }
                if s_size > 0 && sp + 8 > s_size {
                    return Err(CliError::Runtime(format!("mem_u64_add: src offset {} + 8 > size {}", sp, s_size)));
                }
                unsafe {
                    let d_val = (d_ptr.add(dp) as *const u64).read_unaligned();
                    let s_val = (s_ptr.add(sp) as *const u64).read_unaligned();
                    (d_ptr.add(dp) as *mut u64).write_unaligned(d_val.wrapping_add(s_val));
                }
                return Ok(Value::Float(0.0));
            }
            // file_read_into_mem(path, dest_handle, file_offset, byte_count) — read file region into MEM_TABLE
            if name == "file_read_into_mem" && args.len() == 4 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("file_read_into_mem: path must be a string".into()))?.to_string();
                check_read_permission_for(&path)?;
                let dh = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let offset = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u64;
                let count = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let ptr = mem_table_get_ptr(dh)?;
                let size = mem_table_get_size(dh)?;
                if size > 0 && count > size {
                    return Err(CliError::Runtime(format!("file_read_into_mem: count {} > buffer size {}", count, size)));
                }
                use std::io::{Read, Seek, SeekFrom};
                let mut file = std::fs::File::open(&path)
                    .map_err(|e| CliError::Io(format!("file_read_into_mem(\"{}\"): {}", path, e)))?;
                file.seek(SeekFrom::Start(offset))
                    .map_err(|e| CliError::Io(format!("file_read_into_mem: seek to {}: {}", offset, e)))?;
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr, count) };
                file.read_exact(slice)
                    .map_err(|e| CliError::Io(format!("file_read_into_mem: read {} bytes: {}", count, e)))?;
                return Ok(Value::Float(count as f32));
            }
            // file_read_into_mem_u64(path, dest_handle, offset_buf, offset_pos, byte_count)
            // Like file_read_into_mem but reads the u64 file offset from a MEM_TABLE buffer (no f32 precision loss)
            if name == "file_read_into_mem_u64" && args.len() == 5 {
                let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                    .as_str().map_err(|_| CliError::Compile("file_read_into_mem_u64: path must be a string".into()))?.to_string();
                check_read_permission_for(&path)?;
                let dh = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let ob = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let op = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                let count = eval_scalar(&args[4], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
                // Read u64 offset from offset buffer
                let o_ptr = mem_table_get_ptr(ob)?;
                let o_size = mem_table_get_size(ob)?;
                if o_size > 0 && op + 8 > o_size {
                    return Err(CliError::Runtime(format!("file_read_into_mem_u64: offset_pos {} + 8 > buffer size {}", op, o_size)));
                }
                let offset = unsafe { (o_ptr.add(op) as *const u64).read_unaligned() };
                let d_ptr = mem_table_get_ptr(dh)?;
                let d_size = mem_table_get_size(dh)?;
                if d_size > 0 && count > d_size {
                    return Err(CliError::Runtime(format!("file_read_into_mem_u64: count {} > dest size {}", count, d_size)));
                }
                use std::io::{Read, Seek, SeekFrom};
                let mut file = std::fs::File::open(&path)
                    .map_err(|e| CliError::Io(format!("file_read_into_mem_u64(\"{}\"): {}", path, e)))?;
                file.seek(SeekFrom::Start(offset))
                    .map_err(|e| CliError::Io(format!("file_read_into_mem_u64: seek to {}: {}", offset, e)))?;
                let slice = unsafe { std::slice::from_raw_parts_mut(d_ptr, count) };
                file.read_exact(slice)
                    .map_err(|e| CliError::Io(format!("file_read_into_mem_u64: read {} bytes: {}", count, e)))?;
                return Ok(Value::Float(count as f32));
            }

            // Bitwise operations (Phase 72b — needed for FFI flag checking)
            if name == "bit_and" && args.len() == 2 {
                let a = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                let b = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                return Ok(Value::Float((a & b) as f32));
            }
            if name == "bit_or" && args.len() == 2 {
                let a = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                let b = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                return Ok(Value::Float((a | b) as f32));
            }
            if name == "bit_test" && args.len() == 2 {
                let n = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                let bit = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                return Ok(Value::Float(if bit < 32 && (n & (1 << bit)) != 0 { 1.0 } else { 0.0 }));
            }
            if name == "bit_shl" && args.len() == 2 {
                let a = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                let b = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                return Ok(Value::Float(if b < 32 { (a << b) as f32 } else { 0.0 }));
            }
            if name == "bit_shr" && args.len() == 2 {
                let a = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                let b = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                return Ok(Value::Float(if b < 32 { (a >> b) as f32 } else { 0.0 }));
            }
            if name == "bit_xor" && args.len() == 2 {
                let a = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                let b = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
                return Ok(Value::Float((a ^ b) as f32));
            }

            // repeat(s, count) → string repeated N times
            if name == "repeat" && args.len() == 2 {
                let s_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let count_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = s_val.as_str().map_err(|_| CliError::Compile("repeat() first argument must be a string".into()))?;
                let count = count_val.as_float()? as usize;
                return Ok(Value::Str(s.repeat(count)));
            }

            // Statistics functions (Phase 41)
            if name == "mean" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    gpu_array_materialize(arr_name, arrays);
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile("mean() requires array variable".into()))?;
                    let values: Vec<f32> = arr.iter().filter_map(|v| v.as_float().ok()).collect();
                    if values.is_empty() {
                        return Err(CliError::Compile("mean() requires non-empty numeric array".into()));
                    }
                    let sum: f32 = values.iter().sum();
                    return Ok(Value::Float(sum / values.len() as f32));
                } else {
                    return Err(CliError::Compile("mean() requires array variable (not expression)".into()));
                }
            }

            if name == "median" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    gpu_array_materialize(arr_name, arrays);
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile("median() requires array variable".into()))?;
                    let mut values: Vec<f32> = arr.iter().filter_map(|v| v.as_float().ok()).collect();
                    if values.is_empty() {
                        return Err(CliError::Compile("median() requires non-empty numeric array".into()));
                    }
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let mid = values.len() / 2;
                    let median = if values.len() % 2 == 0 {
                        (values[mid - 1] + values[mid]) / 2.0
                    } else {
                        values[mid]
                    };
                    return Ok(Value::Float(median));
                } else {
                    return Err(CliError::Compile("median() requires array variable".into()));
                }
            }

            if name == "stddev" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    gpu_array_materialize(arr_name, arrays);
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile("stddev() requires array variable".into()))?;
                    let values: Vec<f32> = arr.iter().filter_map(|v| v.as_float().ok()).collect();
                    if values.is_empty() {
                        return Err(CliError::Compile("stddev() requires non-empty numeric array".into()));
                    }
                    if values.len() < 2 {
                        return Ok(Value::Float(0.0));
                    }
                    let mean = values.iter().sum::<f32>() / values.len() as f32;
                    let variance = values.iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f32>() / (values.len() - 1) as f32;
                    return Ok(Value::Float(variance.sqrt()));
                } else {
                    return Err(CliError::Compile("stddev() requires array variable".into()));
                }
            }

            if name == "variance" && args.len() == 1 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    gpu_array_materialize(arr_name, arrays);
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile("variance() requires array variable".into()))?;
                    let values: Vec<f32> = arr.iter().filter_map(|v| v.as_float().ok()).collect();
                    if values.is_empty() {
                        return Err(CliError::Compile("variance() requires non-empty numeric array".into()));
                    }
                    if values.len() < 2 {
                        return Ok(Value::Float(0.0));
                    }
                    let mean = values.iter().sum::<f32>() / values.len() as f32;
                    let variance = values.iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f32>() / (values.len() - 1) as f32;
                    return Ok(Value::Float(variance));
                } else {
                    return Err(CliError::Compile("variance() requires array variable".into()));
                }
            }

            if name == "quantile" && args.len() == 2 {
                if let ScalarExpr::Ref(arr_name) = &args[0] {
                    gpu_array_materialize(arr_name, arrays);
                    let arr = arrays.get(arr_name)
                        .ok_or_else(|| CliError::Compile("quantile() requires array variable".into()))?;
                    let mut values: Vec<f32> = arr.iter().filter_map(|v| v.as_float().ok()).collect();
                    if values.is_empty() {
                        return Err(CliError::Compile("quantile() requires non-empty numeric array".into()));
                    }
                    if values.len() == 1 {
                        return Ok(Value::Float(values[0]));
                    }
                    let p_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let p = p_val.as_float()?;
                    if p < 0.0 || p > 1.0 {
                        return Err(CliError::Compile("quantile() requires p in [0.0, 1.0]".into()));
                    }
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let index = (p * (values.len() - 1) as f32).round() as usize;
                    return Ok(Value::Float(values[index]));
                } else {
                    return Err(CliError::Compile("quantile() first arg must be array variable".into()));
                }
            }

            if name == "correlation" && args.len() == 2 {
                if let ScalarExpr::Ref(arr1_name) = &args[0] {
                    if let ScalarExpr::Ref(arr2_name) = &args[1] {
                        gpu_array_materialize(arr1_name, arrays);
                        gpu_array_materialize(arr2_name, arrays);
                        let arr1 = arrays.get(arr1_name)
                            .ok_or_else(|| CliError::Compile("correlation() first arg requires array variable".into()))?;
                        let arr2 = arrays.get(arr2_name)
                            .ok_or_else(|| CliError::Compile("correlation() second arg requires array variable".into()))?;
                        let vals1: Vec<f32> = arr1.iter().filter_map(|v| v.as_float().ok()).collect();
                        let vals2: Vec<f32> = arr2.iter().filter_map(|v| v.as_float().ok()).collect();
                        if vals1.len() != vals2.len() {
                            return Err(CliError::Compile("correlation() requires arrays of same length".into()));
                        }
                        if vals1.is_empty() {
                            return Err(CliError::Compile("correlation() requires non-empty numeric arrays".into()));
                        }
                        let mean1 = vals1.iter().sum::<f32>() / vals1.len() as f32;
                        let mean2 = vals2.iter().sum::<f32>() / vals2.len() as f32;
                        let mut cov = 0.0_f32;
                        let mut var1 = 0.0_f32;
                        let mut var2 = 0.0_f32;
                        for i in 0..vals1.len() {
                            let d1 = vals1[i] - mean1;
                            let d2 = vals2[i] - mean2;
                            cov += d1 * d2;
                            var1 += d1 * d1;
                            var2 += d2 * d2;
                        }
                        let correlation = cov / (var1 * var2).sqrt();
                        return Ok(Value::Float(correlation));
                    } else {
                        return Err(CliError::Compile("correlation() second arg must be array variable".into()));
                    }
                } else {
                    return Err(CliError::Compile("correlation() first arg must be array variable".into()));
                }
            }

            // Path operations (Phase 41)
            if name == "join_path" {
                if args.is_empty() {
                    return Err(CliError::Compile("join_path() requires at least 1 argument".into()));
                }
                let mut parts = Vec::new();
                for arg in args {
                    let val = eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    let s = val.as_str().map_err(|_| CliError::Compile("join_path() arguments must be strings".into()))?;
                    parts.push(s.to_string());
                }
                // Security: validate components
                for (i, part) in parts.iter().enumerate() {
                    if part.contains('\0') {
                        return Err(CliError::Security("path component contains null byte".into()));
                    }
                    if i > 0 && (part.starts_with('/') || part.starts_with('\\')) {
                        return Err(CliError::Security("path component cannot be absolute (use first arg for root)".into()));
                    }
                }
                use std::path::PathBuf;
                let mut path = PathBuf::from(&parts[0]);
                for part in &parts[1..] {
                    path.push(part);
                }
                return Ok(Value::Str(path.to_string_lossy().to_string()));
            }

            if name == "dirname" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = val.as_str().map_err(|_| CliError::Compile("dirname() requires string".into()))?;
                use std::path::Path;
                let p = Path::new(path);
                let parent = p.parent().unwrap_or(Path::new(""));
                return Ok(Value::Str(parent.to_string_lossy().to_string()));
            }

            if name == "basename" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = val.as_str().map_err(|_| CliError::Compile("basename() requires string".into()))?;
                use std::path::Path;
                let p = Path::new(path);
                let filename = p.file_name().unwrap_or_else(|| std::ffi::OsStr::new(""));
                return Ok(Value::Str(filename.to_string_lossy().to_string()));
            }

            if name == "canonicalize_path" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = val.as_str().map_err(|_| CliError::Compile("canonicalize_path() requires string".into()))?;
                check_read_permission_for(path)?;
                use std::fs;
                let canonical = fs::canonicalize(path)
                    .map_err(|e| CliError::Io(format!("canonicalize_path(\"{}\"): {}", path, e)))?;
                return Ok(Value::Str(canonical.to_string_lossy().to_string()));
            }

            if name == "is_file" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = val.as_str().map_err(|_| CliError::Compile("is_file() requires string".into()))?;
                check_read_permission_for(path)?;
                use std::fs;
                let is_file = fs::metadata(path)
                    .map(|m| m.is_file())
                    .unwrap_or(false);
                return Ok(Value::Float(if is_file { 1.0 } else { 0.0 }));
            }

            if name == "is_dir" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = val.as_str().map_err(|_| CliError::Compile("is_dir() requires string".into()))?;
                check_read_permission_for(path)?;
                use std::fs;
                let is_dir = fs::metadata(path)
                    .map(|m| m.is_dir())
                    .unwrap_or(false);
                return Ok(Value::Float(if is_dir { 1.0 } else { 0.0 }));
            }

            if name == "file_mtime" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = val.as_str().map_err(|_| CliError::Compile("file_mtime() requires string".into()))?;
                check_read_permission_for(path)?;
                match std::fs::metadata(path) {
                    Ok(meta) => {
                        match meta.modified() {
                            Ok(time) => {
                                let duration = time.duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default();
                                return Ok(Value::Float(duration.as_secs_f64() as f32));
                            }
                            Err(_) => return Ok(Value::Float(0.0)),
                        }
                    }
                    Err(_) => return Ok(Value::Float(0.0)),
                }
            }

            if name == "is_symlink" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let path = val.as_str().map_err(|_| CliError::Compile("is_symlink() requires string".into()))?;
                check_read_permission_for(path)?;
                use std::fs;
                let is_symlink = fs::symlink_metadata(path)
                    .map(|m| m.is_symlink())
                    .unwrap_or(false);
                return Ok(Value::Float(if is_symlink { 1.0 } else { 0.0 }));
            }

            // Base64/hex encoding (Phase 41)
            if name == "base64_encode" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = val.as_str().map_err(|_| CliError::Compile("base64_encode() requires string".into()))?;
                let encoded = base64_encode_pure(s.as_bytes());
                return Ok(Value::Str(encoded));
            }

            if name == "base64_decode" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = val.as_str().map_err(|_| CliError::Compile("base64_decode() requires string".into()))?;
                let decoded_bytes = base64_decode_pure(s)
                    .map_err(|e| CliError::Compile(format!("base64_decode: {}", e)))?;
                let decoded_str = String::from_utf8(decoded_bytes)
                    .map_err(|e| CliError::Compile(format!("base64_decode: not valid UTF-8: {}", e)))?;
                return Ok(Value::Str(decoded_str));
            }

            if name == "hex_encode" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = val.as_str().map_err(|_| CliError::Compile("hex_encode() requires string".into()))?;
                let encoded = s.as_bytes()
                    .iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<String>();
                return Ok(Value::Str(encoded));
            }

            if name == "hex_decode" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let s = val.as_str().map_err(|_| CliError::Compile("hex_decode() requires string".into()))?;
                if s.len() % 2 != 0 {
                    return Err(CliError::Compile("hex_decode: odd length hex string".into()));
                }
                let mut bytes = Vec::new();
                for i in (0..s.len()).step_by(2) {
                    let byte_str = &s[i..i+2];
                    let byte = u8::from_str_radix(byte_str, 16)
                        .map_err(|e| CliError::Compile(format!("hex_decode: invalid hex: {}", e)))?;
                    bytes.push(byte);
                }
                let decoded_str = String::from_utf8(bytes)
                    .map_err(|e| CliError::Compile(format!("hex_decode: not valid UTF-8: {}", e)))?;
                return Ok(Value::Str(decoded_str));
            }

            // Date/Time operations (Phase 42) — self-hosting friendly (timestamps as floats)
            if name == "now" && args.is_empty() {
                use std::time::{SystemTime, UNIX_EPOCH};
                let duration = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(|e| CliError::Compile(format!("now(): {}", e)))?;
                return Ok(Value::Int(duration.as_secs() as i64));
            }

            // now_ms() — milliseconds since process start; for benchmarking and timing.
            // Returns f32 in [0, ~16M) ms (wraps after ~4.5h), suitable for (t1-t0) elapsed.
            if (name == "now_ms" || name == "time_ms") && args.is_empty() {
                use std::time::Instant;
                use std::sync::OnceLock;
                static START: OnceLock<Instant> = OnceLock::new();
                let start = START.get_or_init(Instant::now);
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(Value::Float(elapsed_ms as f32));
            }
            // R-22: Microsecond timer for GPU profiling
            if name == "now_us" && args.is_empty() {
                use std::time::Instant;
                use std::sync::OnceLock;
                static START_US: OnceLock<Instant> = OnceLock::new();
                let start = START_US.get_or_init(Instant::now);
                let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;
                return Ok(Value::Float(elapsed_us as f32));
            }
            // R-23: GPU timer start/end (CPU-side dispatch timing)
            if name == "gpu_timer_start" && args.is_empty() {
                use std::time::Instant;
                GPU_TIMER_START.with(|t| t.set(Some(Instant::now())));
                return Ok(Value::Float(0.0));
            }
            if name == "gpu_timer_end" && args.is_empty() {
                let elapsed = GPU_TIMER_START.with(|t| {
                    t.get().map(|start| start.elapsed().as_secs_f64() * 1000.0)
                });
                return Ok(Value::Float(elapsed.unwrap_or(0.0) as f32));
            }

            if name == "timestamp" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let iso_str = val.as_str().map_err(|_| CliError::Compile("timestamp() requires ISO8601 string".into()))?;
                let unix = parse_iso8601(iso_str)
                    .map_err(|e| CliError::Compile(format!("timestamp(): {}", e)))?;
                return Ok(Value::Float(unix as f32));
            }

            if name == "timestamp_from_unix" && args.len() == 1 {
                let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let unix_seconds = val.as_float()?;
                return Ok(Value::Float(unix_seconds));  // Identity, but validates it's a number
            }

            if name == "add_seconds" && args.len() == 2 {
                let ts_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let seconds_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ts = ts_val.as_float()?;
                let seconds = seconds_val.as_float()?;
                return Ok(Value::Float(ts + seconds));
            }

            if name == "add_minutes" && args.len() == 2 {
                let ts_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let minutes_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ts = ts_val.as_float()?;
                let minutes = minutes_val.as_float()?;
                return Ok(Value::Float(ts + (minutes * 60.0)));
            }

            if name == "add_hours" && args.len() == 2 {
                let ts_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let hours_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ts = ts_val.as_float()?;
                let hours = hours_val.as_float()?;
                return Ok(Value::Float(ts + (hours * 3600.0)));
            }

            if name == "add_days" && args.len() == 2 {
                let ts_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let days_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ts = ts_val.as_float()?;
                let days = days_val.as_float()?;
                return Ok(Value::Float(ts + (days * 86400.0)));
            }

            if name == "diff_seconds" && args.len() == 2 {
                let ts1_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ts2_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ts1 = ts1_val.as_float()?;
                let ts2 = ts2_val.as_float()?;
                return Ok(Value::Float(ts1 - ts2));
            }

            if name == "diff_days" && args.len() == 2 {
                let ts1_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ts2_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ts1 = ts1_val.as_float()?;
                let ts2 = ts2_val.as_float()?;
                return Ok(Value::Float((ts1 - ts2) / 86400.0));
            }

            if name == "diff_hours" && args.len() == 2 {
                let ts1_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ts2_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let ts1 = ts1_val.as_float()?;
                let ts2 = ts2_val.as_float()?;
                return Ok(Value::Float((ts1 - ts2) / 3600.0));
            }

            if name == "format_datetime" && args.len() == 2 {
                let ts_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fmt_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let unix_ts = ts_val.as_float()?;
                let fmt_str = fmt_val.as_str().map_err(|_| CliError::Compile("format_datetime() format must be string".into()))?;
                let formatted = format_unix_timestamp(unix_ts as i64, fmt_str)
                    .map_err(|e| CliError::Compile(format!("format_datetime(): {}", e)))?;
                return Ok(Value::Str(formatted));
            }

            // Regex operations (Phase 43)
            if name == "regex_match" && args.len() == 2 {
                let text_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let pattern_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let text = text_val.as_str().map_err(|_| CliError::Compile("regex_match() first arg must be string".into()))?;
                let pattern = pattern_val.as_str().map_err(|_| CliError::Compile("regex_match() second arg must be string (pattern)".into()))?;

                use crate::regex_io::Regex;
                let re = Regex::new(pattern)
                    .map_err(|e| CliError::Compile(format!("regex_match(): invalid pattern: {}", e)))?;
                let is_match = re.is_match(text);
                return Ok(Value::Float(if is_match { 1.0 } else { 0.0 }));
            }

            if name == "is_match" && args.len() == 2 {
                // Alias for regex_match
                let text_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let pattern_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let text = text_val.as_str()?;
                let pattern = pattern_val.as_str()?;

                use crate::regex_io::Regex;
                let re = Regex::new(pattern)
                    .map_err(|e| CliError::Compile(format!("is_match(): invalid pattern: {}", e)))?;
                return Ok(Value::Float(if re.is_match(text) { 1.0 } else { 0.0 }));
            }

            if name == "regex_find" && args.len() == 2 {
                let text_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let pattern_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let text = text_val.as_str()?;
                let pattern = pattern_val.as_str()?;

                use crate::regex_io::Regex;
                let re = Regex::new(pattern)
                    .map_err(|e| CliError::Compile(format!("regex_find(): invalid pattern: {}", e)))?;
                let found = re.find(text).unwrap_or_default();
                return Ok(Value::Str(found));
            }

            if name == "regex_replace" && args.len() == 3 {
                let text_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let pattern_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let replacement_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let text = text_val.as_str()?;
                let pattern = pattern_val.as_str()?;
                let replacement = replacement_val.as_str()?;

                use crate::regex_io::Regex;
                let re = Regex::new(pattern)
                    .map_err(|e| CliError::Compile(format!("regex_replace(): invalid pattern: {}", e)))?;
                let result = re.replace_all(text, replacement);
                return Ok(Value::Str(result));
            }

            // ── TCP/UDP sockets (Phase 45) ────────────────────────────
            // tcp_connect(host, port) → float (fd or -1)
            if name == "tcp_connect" && args.len() == 2 {
                check_net_permission()?;
                let host_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let port_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let host = host_val.as_str().map_err(|_| CliError::Compile("tcp_connect() host must be string".into()))?;
                let port = port_val.as_float()? as u16;
                return Ok(Value::Float(crate::net_io::tcp_connect(host, port) as f32));
            }

            // tcp_send(fd, data) → float (bytes sent or -1)
            if name == "tcp_send" && args.len() == 2 {
                check_net_permission()?;
                let fd_val  = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let data_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd   = fd_val.as_float()? as i64;
                let data = data_val.as_str().map_err(|_| CliError::Compile("tcp_send() data must be string".into()))?;
                return Ok(Value::Float(crate::net_io::tcp_send(fd, data) as f32));
            }

            // tcp_recv(fd, max_bytes) → string
            if name == "tcp_recv" && args.len() == 2 {
                check_net_permission()?;
                let fd_val  = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let max_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd  = fd_val.as_float()? as i64;
                let max = max_val.as_float()? as usize;
                let data = crate::net_io::tcp_recv(fd, max)
                    .unwrap_or_else(|_| String::new());
                return Ok(Value::Str(data));
            }

            // tcp_close(fd) → 0.0
            if name == "tcp_close" && args.len() == 1 {
                let fd_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                crate::net_io::socket_close(fd_val.as_float()? as i64);
                return Ok(Value::Float(0.0));
            }

            // tcp_listen(port) → float (fd or -1), requires --allow-net
            if name == "tcp_listen" && args.len() == 1 {
                check_net_permission()?;
                let port_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let port = port_val.as_float()? as u16;
                return Ok(Value::Float(crate::net_io::tcp_listen(port) as f32));
            }

            // tcp_accept(listener_fd) → float (client fd or -1)
            if name == "tcp_accept" && args.len() == 1 {
                check_net_permission()?;
                let fd_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                return Ok(Value::Float(crate::net_io::tcp_accept(fd_val.as_float()? as i64) as f32));
            }

            // udp_socket() → float (fd or -1)
            if name == "udp_socket" && args.is_empty() {
                check_net_permission()?;
                return Ok(Value::Float(crate::net_io::udp_socket() as f32));
            }

            // udp_send_to(fd, host, port, data) → float (bytes or -1)
            if name == "udp_send_to" && args.len() == 4 {
                check_net_permission()?;
                let fd_val   = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let host_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let port_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let data_val = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd   = fd_val.as_float()? as i64;
                let host = host_val.as_str().map_err(|_| CliError::Compile("udp_send_to() host must be string".into()))?;
                let port = port_val.as_float()? as u16;
                let data = data_val.as_str().map_err(|_| CliError::Compile("udp_send_to() data must be string".into()))?;
                return Ok(Value::Float(crate::net_io::udp_send_to(fd, host, port, data) as f32));
            }

            // udp_recv_from(fd, max_bytes) → string (data; use udp_from_addr to get source)
            if name == "udp_recv_from" && args.len() == 2 {
                check_net_permission()?;
                let fd_val  = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let max_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd  = fd_val.as_float()? as i64;
                let max = max_val.as_float()? as usize;
                let data = crate::net_io::udp_recv_from(fd, max)
                    .map(|(d, _)| d)
                    .unwrap_or_else(|_| String::new());
                return Ok(Value::Str(data));
            }

            // ── HTTP server builtins (Phase 46) ──────────────────────
            // http_listen(port) → fd (alias for tcp_listen)
            if name == "http_listen" && args.len() == 1 {
                check_net_permission()?;
                let port_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let port = port_val.as_float()? as u16;
                return Ok(Value::Float(crate::net_io::tcp_listen(port) as f32));
            }

            // http_accept(srv_fd) → client fd (blocks until request, parses it, stores in HTTP_REQUESTS)
            if name == "http_accept" && args.len() == 1 {
                check_net_permission()?;
                let fd_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let srv_fd = fd_val.as_float()? as i64;
                let client_fd = crate::net_io::tcp_accept(srv_fd);
                if client_fd < 0 { return Ok(Value::Float(-1.0)); }
                match crate::http_io::read_request(client_fd) {
                    Ok(req) => {
                        HTTP_REQUESTS.with(|map| map.borrow_mut().insert(client_fd, req));
                        return Ok(Value::Float(client_fd as f32));
                    }
                    Err(_) => {
                        crate::net_io::socket_close(client_fd);
                        return Ok(Value::Float(-1.0));
                    }
                }
            }

            // http_method(fd) → string ("GET", "POST", etc.)
            if name == "http_method" && args.len() == 1 {
                let fd_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd = fd_val.as_float()? as i64;
                let method = HTTP_REQUESTS.with(|map| {
                    map.borrow().get(&fd).map(|r| r.method.clone()).unwrap_or_default()
                });
                return Ok(Value::Str(method));
            }

            // http_path(fd) → string ("/api/hello")
            if name == "http_path" && args.len() == 1 {
                let fd_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd = fd_val.as_float()? as i64;
                let path = HTTP_REQUESTS.with(|map| {
                    map.borrow().get(&fd).map(|r| r.path.clone()).unwrap_or_default()
                });
                return Ok(Value::Str(path));
            }

            // http_query(fd) → string ("key=value&key2=val2")
            if name == "http_query" && args.len() == 1 {
                let fd_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd = fd_val.as_float()? as i64;
                let query = HTTP_REQUESTS.with(|map| {
                    map.borrow().get(&fd).map(|r| r.query.clone()).unwrap_or_default()
                });
                return Ok(Value::Str(query));
            }

            // http_body(fd) → string (request body)
            if name == "http_body" && args.len() == 1 {
                let fd_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd = fd_val.as_float()? as i64;
                let body = HTTP_REQUESTS.with(|map| {
                    map.borrow().get(&fd).map(|r| r.body.clone()).unwrap_or_default()
                });
                return Ok(Value::Str(body));
            }

            // http_header(fd, name) → string (header value or "")
            if name == "http_header" && args.len() == 2 {
                let fd_val   = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let name_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd = fd_val.as_float()? as i64;
                let hname = name_val.as_str().map_err(|_| CliError::Compile("http_header() name must be string".into()))?.to_lowercase();
                let val = HTTP_REQUESTS.with(|map| {
                    map.borrow().get(&fd)
                        .and_then(|r| r.headers.iter().find(|(k, _)| k == &hname).map(|(_, v)| v.clone()))
                        .unwrap_or_default()
                });
                return Ok(Value::Str(val));
            }

            // http_respond(fd, status, body) → 0.0
            // Sends HTTP response and closes the connection.
            if name == "http_respond" && args.len() == 3 {
                check_net_permission()?;
                let fd_val     = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let status_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let body_val   = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd     = fd_val.as_float()? as i64;
                let status = status_val.as_float()? as u16;
                let body   = body_val.as_str().map_err(|_| CliError::Compile("http_respond() body must be string".into()))?;
                let ct = "text/plain; charset=utf-8";
                crate::http_io::send_response(fd, status, ct, body);
                HTTP_REQUESTS.with(|map| { map.borrow_mut().remove(&fd); });
                return Ok(Value::Float(0.0));
            }

            // http_respond_json(fd, status, json_body) → 0.0
            if name == "http_respond_json" && args.len() == 3 {
                check_net_permission()?;
                let fd_val     = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let status_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let body_val   = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd     = fd_val.as_float()? as i64;
                let status = status_val.as_float()? as u16;
                let body   = body_val.as_str().map_err(|_| CliError::Compile("http_respond_json() body must be string".into()))?;
                crate::http_io::send_response(fd, status, "application/json; charset=utf-8", body);
                HTTP_REQUESTS.with(|map| { map.borrow_mut().remove(&fd); });
                return Ok(Value::Float(0.0));
            }

            // ── Phase 83c: HTTP Live View builtins ───────────────────
            // http_accept_nonblock(srv_fd) → client fd or -1.0 immediately
            if name == "http_accept_nonblock" && args.len() == 1 {
                check_net_permission()?;
                let fd_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let srv_fd = fd_val.as_float()? as i64;
                let client_fd = crate::net_io::tcp_accept_nonblock(srv_fd);
                if client_fd < 0 { return Ok(Value::Float(-1.0)); }
                match crate::http_io::read_request(client_fd) {
                    Ok(req) => {
                        HTTP_REQUESTS.with(|map| map.borrow_mut().insert(client_fd, req));
                        return Ok(Value::Float(client_fd as f32));
                    }
                    Err(_) => {
                        crate::net_io::socket_close(client_fd);
                        return Ok(Value::Float(-1.0));
                    }
                }
            }

            // http_respond_html(fd, status, body) → 0.0
            if name == "http_respond_html" && args.len() == 3 {
                check_net_permission()?;
                let fd_val     = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let status_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let body_val   = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd     = fd_val.as_float()? as i64;
                let status = status_val.as_float()? as u16;
                let body   = body_val.as_str().map_err(|_| CliError::Compile("http_respond_html() body must be string".into()))?;
                crate::http_io::send_response(fd, status, "text/html; charset=utf-8", body);
                HTTP_REQUESTS.with(|map| { map.borrow_mut().remove(&fd); });
                return Ok(Value::Float(0.0));
            }

            // http_respond_with_headers(fd, status, headers_map, body) → 0.0
            // headers_map is a hashmap: {"Content-Type": "text/html", "Location": "/new"}
            if name == "http_respond_with_headers" && args.len() == 4 {
                check_net_permission()?;
                let fd_val     = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let status_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                // args[2] must be a hashmap variable name
                let headers_name = match &args[2] {
                    ScalarExpr::Ref(name) => name.clone(),
                    _ => return Err(CliError::Compile(
                        "http_respond_with_headers() arg 3 must be a hashmap variable".into()
                    )),
                };
                let body_val = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;

                let fd     = fd_val.as_float()? as i64;
                let status = status_val.as_float()? as u16;
                let body   = body_val.as_str().map_err(|_|
                    CliError::Compile("http_respond_with_headers() body must be string".into())
                )?;

                let header_pairs: Vec<(String, String)> = hashmaps
                    .get(&headers_name)
                    .map(|m| m.iter().map(|(k, v)| {
                        let val = match v {
                            Value::Str(s) => s.clone(),
                            Value::Float(f) => format!("{}", f),
                            Value::Int(i) => format!("{}", i),
                            _ => format!("{:?}", v),
                        };
                        (k.clone(), val)
                    }).collect())
                    .unwrap_or_default();

                let refs: Vec<(&str, &str)> = header_pairs.iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();

                crate::http_io::send_response_with_headers(fd, status, &refs, body);
                HTTP_REQUESTS.with(|map| { map.borrow_mut().remove(&fd); });
                return Ok(Value::Float(0.0));
            }

            // http_respond_image(fd, status, w, h, r_arr, g_arr, b_arr) → 0.0
            // Encodes GPU pixel arrays as PNG and sends as HTTP response
            if name == "http_respond_image" && args.len() == 7 {
                check_net_permission()?;
                let fd_val     = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let status_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let w_val      = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let h_val      = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let fd     = fd_val.as_float()? as i64;
                let status = status_val.as_float()? as u16;
                let w      = w_val.as_float()? as usize;
                let h      = h_val.as_float()? as usize;
                let r_arr = extract_array_arg("http_respond_image", &args[4], arrays)?;
                let g_arr = extract_array_arg("http_respond_image", &args[5], arrays)?;
                let b_arr = extract_array_arg("http_respond_image", &args[6], arrays)?;
                let total = w * h;
                if r_arr.len() < total || g_arr.len() < total || b_arr.len() < total {
                    return Err(CliError::Compile(format!(
                        "http_respond_image(): arrays must have at least {}*{}={} elements",
                        w, h, total)));
                }
                let mut rgb = Vec::with_capacity(total * 3);
                for i in 0..total {
                    rgb.push(r_arr[i].clamp(0.0, 255.0) as u8);
                    rgb.push(g_arr[i].clamp(0.0, 255.0) as u8);
                    rgb.push(b_arr[i].clamp(0.0, 255.0) as u8);
                }
                let png_bytes = crate::image_io::png::encode(&rgb, w as u32, h as u32);
                crate::http_io::send_response_bytes(fd, status, "image/png", &png_bytes);
                HTTP_REQUESTS.with(|map| { map.borrow_mut().remove(&fd); });
                return Ok(Value::Float(0.0));
            }

            // Try user-defined scalar functions — detect array args
            if let Some(fn_def) = scalar_fns.get(name).cloned() {
                if args.len() != fn_def.params.len() {
                    return Err(CliError::Compile(format!(
                        "function '{}' expects {} argument{}, got {}",
                        name, fn_def.params.len(),
                        if fn_def.params.len() == 1 { "" } else { "s" },
                        args.len()
                    )));
                }
                // Separate array/hashmap args from scalar args
                let mut scalar_args: Vec<(usize, Value)> = Vec::new();
                let mut array_bindings: Vec<(String, String)> = Vec::new(); // (param_name, original_name)
                let mut hashmap_bindings: Vec<(String, String)> = Vec::new(); // (param_name, original_name)
                for (i, arg) in args.iter().enumerate() {
                    if let ScalarExpr::Ref(ref_name) = arg {
                        if arrays.contains_key(ref_name.as_str()) || gpu_array_has(ref_name) {
                            if gpu_array_has(ref_name) {
                                gpu_array_materialize(ref_name, arrays);
                            }
                            array_bindings.push((fn_def.params[i].clone(), ref_name.clone()));
                            continue;
                        }
                        if hashmaps.contains_key(ref_name.as_str()) {
                            hashmap_bindings.push((fn_def.params[i].clone(), ref_name.clone()));
                            continue;
                        }
                    }
                    // R-03: Handle inline array literals → create temporary array, bind as array param
                    if let ScalarExpr::ArrayLiteral(elements) = arg {
                        let mut vals = Vec::new();
                        for elem in elements {
                            vals.push(eval_scalar(elem, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?);
                        }
                        let temp_name = format!("__array_lit_{}_{}", name, i);
                        arrays.insert(temp_name.clone(), vals);
                        array_bindings.push((fn_def.params[i].clone(), temp_name));
                        continue;
                    }
                    let val = eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                    scalar_args.push((i, val));
                }
                // R-03: Clean up temporary arrays after function call
                let result = execute_user_fn(
                    &fn_def, &scalar_args, &array_bindings, &hashmap_bindings,
                    scalars, arrays, hashmaps, mutable_scalars,
                    streams, gpu, &scalar_fns, struct_defs, rng,
                );
                for i in 0..args.len() {
                    let temp_name = format!("__array_lit_{}_{}", name, i);
                    arrays.remove(&temp_name);
                }
                return result;
            }
            let evaluated: Vec<Value> = args.iter()
                .map(|a| eval_scalar(a, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars))
                .collect::<Result<_, _>>()?;
            // Built-in scalar fns require float args
            let float_args: Vec<f32> = evaluated.iter()
                .map(|v| v.as_float())
                .collect::<Result<_, _>>()?;
            Ok(Value::Float(eval_scalar_fn(name, &float_args)?))
        }
        ScalarExpr::Index { array, index } => {
            // Evaluate index first to avoid borrow conflict
            let idx_val = eval_scalar(index, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            // Check GPU arrays first (zero-copy single element)
            if gpu_array_has(array) {
                let idx = idx_val.as_float()? as usize;
                return gpu_array_index(array, idx)
                    .map(Value::Float)
                    .ok_or_else(|| CliError::Compile(format!(
                        "Array index out of bounds: {}[{}]", array, idx
                    )));
            }
            // Try regular array
            if let Some(arr) = arrays.get(array) {
                let idx = idx_val.as_float()? as usize;
                return arr.get(idx)
                    .cloned()
                    .ok_or_else(|| CliError::Compile(format!(
                        "Array index out of bounds: {}[{}] (length {})", array, idx, arr.len()
                    )));
            }
            // Fall back to hashmap bracket access: map["key"]
            if let Some(hm) = hashmaps.get(array) {
                let key = match &idx_val {
                    Value::Str(s) => s.clone(),
                    Value::Float(f) => {
                        if *f == f.trunc() && f.is_finite() {
                            format!("{}", *f as i64)
                        } else {
                            format!("{}", f)
                        }
                    }
                    Value::Int(i) => format!("{}", i),
                    Value::Map(_) => return Err(CliError::Compile("map index key cannot be a map".into())),
                    Value::None => return Err(CliError::Compile("map index key cannot be none".into())),
                };
                return hm.get(&key)
                    .cloned()
                    .ok_or_else(|| CliError::Compile(format!("key '{}' not found in map '{}'", key, array)));
            }
            // Check if name is a scalar holding Value::Map (e.g., from for-each over read_csv result)
            if let Some(Value::Map(map)) = scalars.get(array.as_str()) {
                let key = idx_val.as_str().map_err(|_| CliError::Compile(
                    format!("map index on '{}' must be a string", array)))?;
                return map.get(key).cloned().ok_or_else(|| CliError::Compile(
                    format!("key '{}' not found in map '{}'", key, array)));
            }
            Err(CliError::Compile(format!("Undefined array or map '{}'", array)))
        }
        ScalarExpr::Lambda { .. } => {
            Err(CliError::Compile("lambda can only be used as argument to filter/map_each/reduce/sort_by".into()))
        }
        ScalarExpr::ArrayLiteral(_) => {
            Err(CliError::Compile("array literals can only be used as function arguments (e.g., sum([1,2,3]))".into()))
        }
    }
}

/// xorshift64* RNG — advance state and return f32 in [0.0, 1.0).
fn next_random(rng: &Cell<u64>) -> f32 {
    let mut s = rng.get();
    // Ensure seed is never zero (xorshift requirement)
    if s == 0 { s = 0xdeadbeef; }
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    rng.set(s);
    // xorshift64* mixing step + normalize to [0, 1)
    let mixed = s.wrapping_mul(0x2545F4914F6CDD1D);
    // Take upper 23 bits for f32 mantissa precision
    (mixed >> 40) as f32 / (1u64 << 24) as f32
}

/// Extract lambda params and body from a ScalarExpr, or return an error.
fn extract_lambda(expr: &ScalarExpr) -> Result<(&[String], &ScalarExpr), CliError> {
    match expr {
        ScalarExpr::Lambda { params, body } => Ok((params, body)),
        _ => Err(CliError::Compile("expected lambda: fn(x) expr end".into())),
    }
}

/// Invoke a lambda expression with the given argument bindings.
/// Clones outer scalars into a local scope, binds params, evaluates the body.
#[allow(clippy::too_many_arguments)]
fn invoke_lambda(
    params: &[String],
    body: &ScalarExpr,
    arg_values: &[Value],
    outer_scalars: &HashMap<String, Value>,
    streams: &HashMap<String, Vec<f32>>,
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    arrays: &mut HashMap<String, Vec<Value>>,
    hashmaps: &mut HashMap<String, HashMap<String, Value>>,
    scalar_fns: &HashMap<String, ScalarFnDef>,
    struct_defs: &HashMap<String, Vec<String>>,
    rng: &Cell<u64>,
) -> Result<Value, CliError> {
    let mut local_scalars = outer_scalars.clone();
    for (p, v) in params.iter().zip(arg_values.iter()) {
        local_scalars.insert(p.clone(), v.clone());
    }
    let empty_mutable = std::collections::HashSet::new();
    eval_scalar(body, streams, &local_scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, &empty_mutable)
}

// ── GGUF dequant helpers (v1.22) ─────────────────────────────────────────
// Compiled Rust replaces interpreted .flow for ~1000x speedup.

fn gguf_f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;
    if exp == 0 {
        if mantissa == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        let val = (mantissa as f32 / 1024.0) * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 { return 0.0; } // OctoFlow: inf/nan → 0
    let val = (1.0 + mantissa as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15);
    if sign == 1 { -val } else { val }
}

fn gguf_q4k_scale_min(j: usize, scales: &[u8]) -> (f32, f32) {
    if j < 4 {
        let d = (scales[j] & 63) as f32;
        let m = (scales[j + 4] & 63) as f32;
        (d, m)
    } else {
        let low_d = scales[j + 4] & 0x0F;
        let high_d = scales[j - 4] >> 6;
        let d = (low_d | (high_d << 4)) as f32;
        let low_m = scales[j + 4] >> 4;
        let high_m = scales[j] >> 6;
        let m = (low_m | (high_m << 4)) as f32;
        (d, m)
    }
}

fn gguf_dequant_q4k_block(data: &[u8], out: &mut Vec<f32>) {
    let d = gguf_f16_to_f32(u16::from_le_bytes([data[0], data[1]]));
    let dmin = gguf_f16_to_f32(u16::from_le_bytes([data[2], data[3]]));
    let scales = &data[4..16];
    let qs = &data[16..144];
    let mut is_idx = 0usize;
    let mut q_ptr = 0usize;
    for _ in 0..4 {
        let (sc1, mn1) = gguf_q4k_scale_min(is_idx, scales);
        let d1 = d * sc1;
        let m1 = dmin * mn1;
        let (sc2, mn2) = gguf_q4k_scale_min(is_idx + 1, scales);
        let d2 = d * sc2;
        let m2 = dmin * mn2;
        for l in 0..32 {
            out.push(d1 * (qs[q_ptr + l] & 0x0F) as f32 - m1);
        }
        for l in 0..32 {
            out.push(d2 * (qs[q_ptr + l] >> 4) as f32 - m2);
        }
        q_ptr += 32;
        is_idx += 2;
    }
}

fn gguf_dequant_q5k_block(data: &[u8], out: &mut Vec<f32>) {
    // Q5_K layout: d(f16, 2B) + dmin(f16, 2B) + scales(12B) + qh(32B) + qs(128B) = 176B
    // 256 elements per block, 5 bits per weight (4 from qs nibbles + 1 from qh bits)
    let d = gguf_f16_to_f32(u16::from_le_bytes([data[0], data[1]]));
    let dmin = gguf_f16_to_f32(u16::from_le_bytes([data[2], data[3]]));
    let scales = &data[4..16];
    let qh = &data[16..48];
    let qs = &data[48..176];
    let mut ql_off = 0usize;
    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for _ in 0..4 {
        let (sc1, mn1) = gguf_q4k_scale_min(is, scales);
        let d1 = d * sc1;
        let m1 = dmin * mn1;
        let (sc2, mn2) = gguf_q4k_scale_min(is + 1, scales);
        let d2 = d * sc2;
        let m2 = dmin * mn2;
        for l in 0..32 {
            let lo = (qs[ql_off + l] & 0x0F) as f32;
            let hi = if (qh[l] & u1) != 0 { 16.0 } else { 0.0 };
            out.push(d1 * (lo + hi) - m1);
        }
        for l in 0..32 {
            let lo = (qs[ql_off + l] >> 4) as f32;
            let hi = if (qh[l] & u2) != 0 { 16.0 } else { 0.0 };
            out.push(d2 * (lo + hi) - m2);
        }
        ql_off += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn gguf_dequant_q6k_block(data: &[u8], out: &mut Vec<f32>) {
    // Q6_K layout: ql(128B) + qh(64B) + scales(16B, signed i8) + d(f16, 2B) = 210B
    let ql = &data[0..128];
    let qh = &data[128..192];
    let sc = &data[192..208];
    let d = gguf_f16_to_f32(u16::from_le_bytes([data[208], data[209]]));
    let mut block = [0.0f32; 256];
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    let mut y_off = 0usize;
    for _half in 0..2usize {
        for l in 0..32usize {
            let is = l / 16;
            let ql_a = ql[ql_off + l];
            let ql_b = ql[ql_off + l + 32];
            let qh_v = qh[qh_off + l];
            let q1 = ((ql_a & 0x0F) as i32 | (((qh_v as i32) & 3) << 4)) - 32;
            let q2 = ((ql_b & 0x0F) as i32 | ((((qh_v >> 2) as i32) & 3) << 4)) - 32;
            let q3 = ((ql_a >> 4) as i32 | ((((qh_v >> 4) as i32) & 3) << 4)) - 32;
            let q4 = ((ql_b >> 4) as i32 | ((((qh_v >> 6) as i32) & 3) << 4)) - 32;
            block[y_off + l]      = d * (sc[sc_off + is] as i8 as f32) * q1 as f32;
            block[y_off + l + 32] = d * (sc[sc_off + is + 2] as i8 as f32) * q2 as f32;
            block[y_off + l + 64] = d * (sc[sc_off + is + 4] as i8 as f32) * q3 as f32;
            block[y_off + l + 96] = d * (sc[sc_off + is + 6] as i8 as f32) * q4 as f32;
        }
        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
    out.extend_from_slice(&block);
}

// Embedded dequant SPIR-V kernels for GPU-accelerated tensor dequantization
static DEQUANT_Q4K_SPV: &[u8] = include_bytes!("../../../stdlib/llm/kernels/dequant_q4k.spv");
static DEQUANT_Q6K_SPV: &[u8] = include_bytes!("../../../stdlib/llm/kernels/dequant_q6k.spv");

/// GPU batch Q4_K dequantization. CPU pre-decodes d/dmin/scales into param buffer,
/// GPU handles nibble extraction + final multiply. Falls back to Err if no GPU or too small.
/// Upper bound: tensors > 4M elements stay on CPU (transfer cost dominates, avoids OOM).
fn gpu_dequant_q4k_batch(raw: &[u8], total_count: usize) -> Result<Vec<f32>, CliError> {
    let device_ptr = GPU_DEVICE_PTR.with(|p| p.get());
    if device_ptr == 0 || total_count < 10_000 || total_count > 4_000_000 {
        return Err(CliError::Runtime("no GPU, tensor too small, or too large for GPU dequant".into()));
    }
    let device = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
    let n_blocks = total_count / 256;

    // CPU: extract qs bytes (128/block as f32) + compute 16 params/block
    let mut qs_floats = Vec::with_capacity(n_blocks * 128);
    let mut params = Vec::with_capacity(n_blocks * 16);
    for block in raw.chunks(144) {
        if block.len() < 144 { break; }
        let d = gguf_f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = gguf_f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        // qs bytes at offset 16..144
        for i in 16..144 { qs_floats.push(block[i] as f32); }
        // 8 sub-blocks → 16 params: [d*sc0, dmin*mn0, d*sc1, dmin*mn1, ...]
        for j in 0..8 {
            let (sc, mn) = gguf_q4k_scale_min(j, scales);
            params.push(d * sc);
            params.push(dmin * mn);
        }
    }

    // Dispatch GPU kernel: 2 inputs (qs, params) + 1 output
    let inputs = vec![qs_floats, params];
    let wg_x = ((total_count + 255) / 256) as u32;
    octoflow_vulkan::gpu_run_dispatch(device, DEQUANT_Q4K_SPV, &inputs, &[], total_count, Some((wg_x, 1, 1)))
        .map_err(|e| CliError::Gpu(format!("gpu_dequant_q4k_batch: {}", e)))
}

/// GPU batch Q6_K dequantization. CPU pre-decodes d*scale into param buffer,
/// GPU handles bit extraction + final multiply. Falls back to Err if no GPU or too small.
/// Upper bound: tensors > 4M elements stay on CPU (transfer cost dominates, avoids OOM).
fn gpu_dequant_q6k_batch(raw: &[u8], total_count: usize) -> Result<Vec<f32>, CliError> {
    let device_ptr = GPU_DEVICE_PTR.with(|p| p.get());
    if device_ptr == 0 || total_count < 10_000 || total_count > 4_000_000 {
        return Err(CliError::Runtime("no GPU, tensor too small, or too large for GPU dequant".into()));
    }
    let device = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
    let n_blocks = total_count / 256;

    // CPU: extract raw bytes (128 ql + 64 qh = 192/block as f32) + 16 d*scale params
    let mut raw_floats = Vec::with_capacity(n_blocks * 192);
    let mut params = Vec::with_capacity(n_blocks * 16);
    for block in raw.chunks(210) {
        if block.len() < 210 { break; }
        let d = gguf_f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        // 128 ql bytes + 64 qh bytes = 192
        for i in 0..192 { raw_floats.push(block[i] as f32); }
        // 16 signed scale params: d * sc[i]
        for i in 0..16 {
            let sv = block[192 + i] as i8 as f32;
            params.push(d * sv);
        }
    }

    let inputs = vec![raw_floats, params];
    let wg_x = ((total_count + 255) / 256) as u32;
    octoflow_vulkan::gpu_run_dispatch(device, DEQUANT_Q6K_SPV, &inputs, &[], total_count, Some((wg_x, 1, 1)))
        .map_err(|e| CliError::Gpu(format!("gpu_dequant_q6k_batch: {}", e)))
}

/// Evaluate array-returning functions: read_lines, list_dir, split, GPU ops.
/// Returns Some(ArrayResult) if this is an array-returning function, None otherwise.
/// GPU-producing functions return ArrayResult::GpuFloats for zero-copy storage.
#[allow(clippy::too_many_arguments)]
fn eval_array_fn(
    fn_name: &str,
    args: &[ScalarExpr],
    streams: &HashMap<String, Vec<f32>>,
    scalars: &HashMap<String, Value>,
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    arrays: &mut HashMap<String, Vec<Value>>,
    hashmaps: &mut HashMap<String, HashMap<String, Value>>,
    scalar_fns: &HashMap<String, ScalarFnDef>,
    struct_defs: &HashMap<String, Vec<String>>,
    rng: &Cell<u64>,
    mutable_scalars: &std::collections::HashSet<String>,
) -> Result<Option<ArrayResult>, CliError> {
    match fn_name {
        "read_bytes" if args.len() == 1 => {
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let path = path_val.as_str().map_err(|_| CliError::Compile("read_bytes() path must be a string".into()))?;
            check_read_permission_for(path)?;
            let bytes = std::fs::read(path)
                .map_err(|e| CliError::Io(format!("read_bytes(\"{}\"): {}", path, e)))?;
            let arr: Vec<Value> = bytes.into_iter().map(|b| Value::Float(b as f32)).collect();
            Ok(Some(ArrayResult::Values(arr)))
        }
        // gdi_text_atlas() — pack all batch entries into flat alpha array for GPU
        #[cfg(target_os = "windows")]
        "gdi_text_atlas" if args.is_empty() => {
            let atlas = crate::text_render::text_atlas();
            Ok(Some(ArrayResult::GpuFloats(atlas)))
        }
        // read_f32_le(path) — read raw little-endian f32 binary file into array
        "read_f32_le" if args.len() == 1 => {
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let path = path_val.as_str().map_err(|_| CliError::Compile("read_f32_le() path must be a string".into()))?;
            check_read_permission_for(path)?;
            let bytes = std::fs::read(path)
                .map_err(|e| CliError::Io(format!("read_f32_le(\"{}\"): {}", path, e)))?;
            if bytes.len() % 4 != 0 {
                return Err(CliError::Compile(format!("read_f32_le(\"{}\"): file size {} not divisible by 4", path, bytes.len())));
            }
            let arr: Vec<Value> = bytes.chunks_exact(4)
                .map(|c| Value::Float(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect();
            Ok(Some(ArrayResult::Values(arr)))
        }
        "read_lines" if args.len() == 1 => {
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let path = path_val.as_str().map_err(|_| CliError::Compile("read_lines() path must be a string".into()))?;
            check_read_permission_for(path)?;
            let contents = std::fs::read_to_string(path)
                .map_err(|e| CliError::Io(format!("read_lines(\"{}\"): {}", path, e)))?;
            let lines: Vec<Value> = contents.lines().map(|l| Value::Str(l.to_string())).collect();
            Ok(Some(ArrayResult::Values(lines)))
        }
        "list_dir" if args.len() == 1 => {
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let path = path_val.as_str().map_err(|_| CliError::Compile("list_dir() path must be a string".into()))?;
            check_read_permission_for(path)?;
            let mut entries: Vec<Value> = Vec::new();
            let dir = std::fs::read_dir(path)
                .map_err(|e| CliError::Io(format!("list_dir(\"{}\"): {}", path, e)))?;
            for entry in dir {
                let entry = entry.map_err(|e| CliError::Io(format!("list_dir: {}", e)))?;
                entries.push(Value::Str(entry.file_name().to_string_lossy().into_owned()));
            }
            entries.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
            Ok(Some(ArrayResult::Values(entries)))
        }
        "walk_dir" if args.len() == 1 => {
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let path = path_val.as_str().map_err(|_| CliError::Compile("walk_dir() path must be a string".into()))?;
            check_read_permission_for(path)?;

            fn walk_recursive(dir: &std::path::Path, out: &mut Vec<Value>) -> Result<(), std::io::Error> {
                for entry in std::fs::read_dir(dir)? {
                    let entry = entry?;
                    if entry.file_type()?.is_dir() {
                        walk_recursive(&entry.path(), out)?;
                    } else {
                        out.push(Value::Str(entry.path().to_string_lossy().replace('\\', "/").to_string()));
                    }
                }
                Ok(())
            }

            let mut files = Vec::new();
            walk_recursive(std::path::Path::new(path), &mut files)
                .map_err(|e| CliError::Io(format!("walk_dir(\"{}\"): {}", path, e)))?;
            files.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
            Ok(Some(ArrayResult::Values(files)))
        }
        "split" if args.len() == 2 => {
            let str_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let delim_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let s = str_val.as_str().map_err(|_| CliError::Compile("split() first arg must be a string".into()))?;
            let delim = delim_val.as_str().map_err(|_| CliError::Compile("split() second arg must be a string".into()))?;
            let parts: Vec<Value> = s.split(delim).map(|p| Value::Str(p.to_string())).collect();
            Ok(Some(ArrayResult::Values(parts)))
        }
        // ── Regex (Phase 43) ──────────────────────────────────────
        "regex_split" if args.len() == 2 => {
            let text_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let pattern_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let text = text_val.as_str()?;
            let pattern = pattern_val.as_str()?;

            use crate::regex_io::Regex;
            let re = Regex::new(pattern)
                .map_err(|e| CliError::Compile(format!("regex_split(): invalid pattern: {}", e)))?;
            let parts: Vec<Value> = re.split(text).into_iter().map(Value::Str).collect();
            Ok(Some(ArrayResult::Values(parts)))
        }
        "capture_groups" if args.len() == 2 => {
            let text_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let pattern_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let text = text_val.as_str()?;
            let pattern = pattern_val.as_str()?;

            use crate::regex_io::Regex;
            let re = Regex::new(pattern)
                .map_err(|e| CliError::Compile(format!("capture_groups(): invalid pattern: {}", e)))?;

            let captures: Vec<Value> = re.captures(text)
                .unwrap_or_default()
                .into_iter()
                .skip(1)
                .filter_map(|c| c.map(Value::Str))
                .collect();
            Ok(Some(ArrayResult::Values(captures)))
        }
        "regex_find_all" if args.len() == 2 => {
            let text_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let pattern_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let text = text_val.as_str()?;
            let pattern = pattern_val.as_str()?;

            use crate::regex_io::Regex;
            let re = Regex::new(pattern)
                .map_err(|e| CliError::Compile(format!("regex_find_all(): invalid pattern: {}", e)))?;
            let matches: Vec<Value> = re.find_all(text).into_iter().map(Value::Str).collect();
            Ok(Some(ArrayResult::Values(matches)))
        }
        // ── Structured CSV (Phase 39) ─────────────────────────────
        "read_csv" if args.len() == 1 => {
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let path = path_val.as_str().map_err(|_| CliError::Compile("read_csv() path must be a string".into()))?.to_string();
            check_read_permission_for(&path)?;
            let rows = csv_read_structured(&path)?;
            let values: Vec<Value> = rows.into_iter().map(Value::Map).collect();
            Ok(Some(ArrayResult::Values(values)))
        }
        // ── Web search (v1.2) ────────────────────────────────────
        "web_search" if args.len() == 1 => {
            check_net_permission()?;
            let query_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let query = query_val.as_str().map_err(|_| CliError::Compile("web_search() argument must be a string".into()))?.to_string();
            let results = crate::io::web::web_search(&query)?;
            Ok(Some(ArrayResult::Values(results)))
        }
        // ── Array operations (Phase 33) ──────────────────────────
        // reverse(arr) → new reversed array
        "reverse" if args.len() == 1 => {
            if let ScalarExpr::Ref(arr_name) = &args[0] {
                gpu_array_materialize(arr_name, arrays);
                let arr = arrays.get(arr_name)
                    .ok_or_else(|| CliError::Compile(format!("reverse() requires array, '{}' not found", arr_name)))?;
                let mut reversed = arr.clone();
                reversed.reverse();
                return Ok(Some(ArrayResult::Values(reversed)));
            }
            Err(CliError::Compile("reverse() argument must be an array name".into()))
        }
        // slice(arr, start, end) → subarray
        "slice" if args.len() == 3 => {
            if let ScalarExpr::Ref(arr_name) = &args[0] {
                gpu_array_materialize(arr_name, arrays);
                let start_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let end_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                let arr = arrays.get(arr_name)
                    .ok_or_else(|| CliError::Compile(format!("slice() requires array, '{}' not found", arr_name)))?;
                let start = (start_val.as_float()? as usize).min(arr.len());
                let end = (end_val.as_float()? as usize).min(arr.len());
                let start = start.min(end);
                return Ok(Some(ArrayResult::Values(arr[start..end].to_vec())));
            }
            Err(CliError::Compile("slice() first argument must be an array name".into()))
        }
        // sort_array(arr) → new sorted array (float ascending, strings alphabetical)
        "sort_array" if args.len() == 1 => {
            if let ScalarExpr::Ref(arr_name) = &args[0] {
                gpu_array_materialize(arr_name, arrays);
                let arr = arrays.get(arr_name)
                    .ok_or_else(|| CliError::Compile(format!("sort_array() requires array, '{}' not found", arr_name)))?;
                let mut sorted = arr.clone();
                sorted.sort_by(|a, b| {
                    match (a, b) {
                        (Value::Float(fa), Value::Float(fb)) => fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Str(sa), Value::Str(sb)) => sa.cmp(sb),
                        (Value::Float(_), Value::Str(_)) => std::cmp::Ordering::Less,
                        (Value::Str(_), Value::Float(_)) => std::cmp::Ordering::Greater,
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                return Ok(Some(ArrayResult::Values(sorted)));
            }
            Err(CliError::Compile("sort_array() argument must be an array name".into()))
        }
        // unique(arr) → new array with duplicates removed (preserves order)
        "unique" if args.len() == 1 => {
            if let ScalarExpr::Ref(arr_name) = &args[0] {
                gpu_array_materialize(arr_name, arrays);
                let arr = arrays.get(arr_name)
                    .ok_or_else(|| CliError::Compile(format!("unique() requires array, '{}' not found", arr_name)))?;
                let mut seen = Vec::new();
                let mut result = Vec::new();
                for val in arr {
                    let key = val.to_string();
                    if !seen.contains(&key) {
                        seen.push(key);
                        result.push(val.clone());
                    }
                }
                return Ok(Some(ArrayResult::Values(result)));
            }
            Err(CliError::Compile("unique() argument must be an array name".into()))
        }
        // range_array(start, end) → array of floats [start, start+1, ..., end-1]
        "range_array" if args.len() == 2 => {
            let start_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let end_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let start = start_val.as_float()? as i64;
            let end = end_val.as_float()? as i64;
            let result: Vec<Value> = (start..end).map(|i| Value::Float(i as f32)).collect();
            Ok(Some(ArrayResult::Values(result)))
        }
        "json_parse_array" if args.len() == 1 => {
            let json_str = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let s = json_str.as_str().map_err(|_| CliError::Compile("json_parse_array() argument must be a string".into()))?.to_string();
            let result = crate::json_io::parse_array(&s)?;
            Ok(Some(ArrayResult::Values(result)))
        }
        // ── Higher-order array functions (Phase 38) ─────────────────
        // filter(arr, fn(x) cond end) → keep elements where cond is truthy
        "filter" if args.len() == 2 => {
            if let ScalarExpr::Ref(arr_name) = &args[0] {
                gpu_array_materialize(arr_name, arrays);
                let (params, body) = extract_lambda(&args[1])?;
                if params.len() != 1 {
                    return Err(CliError::Compile("filter() lambda must take exactly 1 parameter".into()));
                }
                let source = arrays.get(arr_name)
                    .ok_or_else(|| CliError::Compile(format!("filter() requires array, '{}' not found", arr_name)))?
                    .clone();
                let mut result = Vec::new();
                for elem in &source {
                    let val = invoke_lambda(params, body, &[elem.clone()], scalars, streams, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng)?;
                    let keep = match &val {
                        Value::Float(f) => *f != 0.0,
                        Value::Int(i) => *i != 0,
                        Value::Str(s) => !s.is_empty(),
                        Value::Map(m) => !m.is_empty(),
                        Value::None => false,
                    };
                    if keep { result.push(elem.clone()); }
                }
                return Ok(Some(ArrayResult::Values(result)));
            }
            Err(CliError::Compile("filter() first argument must be an array name".into()))
        }
        // map_each(arr, fn(x) expr end) → transform each element
        "map_each" if args.len() == 2 => {
            if let ScalarExpr::Ref(arr_name) = &args[0] {
                gpu_array_materialize(arr_name, arrays);
                let (params, body) = extract_lambda(&args[1])?;
                if params.len() != 1 {
                    return Err(CliError::Compile("map_each() lambda must take exactly 1 parameter".into()));
                }
                let source = arrays.get(arr_name)
                    .ok_or_else(|| CliError::Compile(format!("map_each() requires array, '{}' not found", arr_name)))?
                    .clone();
                let mut result = Vec::new();
                for elem in &source {
                    let val = invoke_lambda(params, body, &[elem.clone()], scalars, streams, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng)?;
                    result.push(val);
                }
                return Ok(Some(ArrayResult::Values(result)));
            }
            Err(CliError::Compile("map_each() first argument must be an array name".into()))
        }
        // sort_by(arr, fn(x) key end) → sort by key function
        "sort_by" if args.len() == 2 => {
            if let ScalarExpr::Ref(arr_name) = &args[0] {
                gpu_array_materialize(arr_name, arrays);
                let (params, body) = extract_lambda(&args[1])?;
                if params.len() != 1 {
                    return Err(CliError::Compile("sort_by() lambda must take exactly 1 parameter".into()));
                }
                let source = arrays.get(arr_name)
                    .ok_or_else(|| CliError::Compile(format!("sort_by() requires array, '{}' not found", arr_name)))?
                    .clone();
                // Compute sort keys for each element
                let mut keyed: Vec<(Value, Value)> = Vec::new();
                for elem in &source {
                    let key = invoke_lambda(params, body, &[elem.clone()], scalars, streams, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng)?;
                    keyed.push((elem.clone(), key));
                }
                keyed.sort_by(|(_, ka), (_, kb)| {
                    match (ka, kb) {
                        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Str(a), Value::Str(b)) => a.cmp(b),
                        (Value::Float(_), Value::Str(_)) => std::cmp::Ordering::Less,
                        (Value::Str(_), Value::Float(_)) => std::cmp::Ordering::Greater,
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                let result: Vec<Value> = keyed.into_iter().map(|(v, _)| v).collect();
                return Ok(Some(ArrayResult::Values(result)));
            }
            Err(CliError::Compile("sort_by() first argument must be an array name".into()))
        }
        // ── GPU compute from .flow SPIR-V (Phase 67) ──────────────────
        // gpu_compute(spv_path, input_array_name) → output array
        // Loads a SPIR-V binary and dispatches it on the GPU with the given input.
        "gpu_compute" if args.len() == 2 => {
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let spv_path = path_val.as_str().map_err(|_| CliError::Compile("gpu_compute() first arg must be a path string".into()))?;
            check_read_permission_for(spv_path)?;

            // Second arg: array name (string literal) or inline array
            let input_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let input_name = input_val.as_str().map_err(|_| CliError::Compile("gpu_compute() second arg must be an array name string".into()))?;
            let input_f32: Vec<f32> = if let Some(ga) = gpu_array_get(input_name) {
                ga
            } else {
                let input_arr = arrays.get(input_name).ok_or_else(|| {
                    CliError::Runtime(format!("gpu_compute(): array '{}' not found", input_name))
                })?;
                input_arr.iter().map(|v| v.as_float().unwrap_or(0.0)).collect()
            };

            let gpu_ref = gpu.as_ref().ok_or_else(|| {
                CliError::Runtime("gpu_compute(): no GPU device available".into())
            })?;

            let spv_key = spv_path.to_string();
            let spirv = SPIRV_FILE_CACHE.with(|cache| {
                let c = cache.borrow();
                if let Some(data) = c.get(&spv_key) {
                    return Ok(data.clone());
                }
                drop(c);
                let data = std::fs::read(spv_path)
                    .map_err(|e| CliError::Io(format!("gpu_compute(\"{}\"): {}", spv_path, e)))?;
                cache.borrow_mut().insert(spv_key, data.clone());
                Ok(data)
            })?;

            let output = octoflow_vulkan::dispatch_compute(gpu_ref, &spirv, &input_f32, 256)
                .map_err(|e| CliError::Runtime(format!("gpu_compute(): GPU dispatch failed: {:?}", e)))?;

            Ok(Some(ArrayResult::GpuFloats(output)))
        }
        // ── Matrix multiply (Phase 75) ─────────────────────────────────
        "mat_mul" if args.len() == 5 => {
            let a_f32 = if let ScalarExpr::Ref(name) = &args[0] {
                if let Some(ga) = gpu_array_get(name) { ga }
                else if let Some(arr) = arrays.get(name) { arr.iter().map(|v| v.as_float().unwrap_or(0.0)).collect() }
                else { return Err(CliError::Compile(format!("mat_mul(): array '{}' not found", name))); }
            } else {
                return Err(CliError::Compile("mat_mul() first arg must be an array name".into()));
            };
            let b_f32 = if let ScalarExpr::Ref(name) = &args[1] {
                if let Some(ga) = gpu_array_get(name) { ga }
                else if let Some(arr) = arrays.get(name) { arr.iter().map(|v| v.as_float().unwrap_or(0.0)).collect() }
                else { return Err(CliError::Compile(format!("mat_mul(): array '{}' not found", name))); }
            } else {
                return Err(CliError::Compile("mat_mul() second arg must be an array name".into()));
            };
            let m_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
            let n_val = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
            let k_val = eval_scalar(&args[4], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u32;
            let result = if let Some(ref gpu_dev) = gpu {
                octoflow_vulkan::dispatch_matmul(gpu_dev, &a_f32, &b_f32, m_val, n_val, k_val)
                    .map_err(|e| CliError::Gpu(format!("mat_mul(): {}", e)))?
            } else {
                cpu_matmul(&a_f32, &b_f32, m_val as usize, n_val as usize, k_val as usize)
            };
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── GPU element-wise unary ops (Phase 79: resident fast-path) ────
        "gpu_abs" | "gpu_sqrt" | "gpu_exp" | "gpu_log" | "gpu_negate"
        | "gpu_floor" | "gpu_ceil" | "gpu_round" | "gpu_sin" | "gpu_cos"
        if args.len() == 1 => {
            let op = match fn_name {
                "gpu_abs" => MapOp::Abs, "gpu_sqrt" => MapOp::Sqrt,
                "gpu_exp" => MapOp::Exp, "gpu_log" => MapOp::Log,
                "gpu_negate" => MapOp::Negate, "gpu_floor" => MapOp::Floor,
                "gpu_ceil" => MapOp::Ceil, "gpu_round" => MapOp::Round,
                "gpu_sin" => MapOp::Sin, _ => MapOp::Cos,
            };
            // Resident fast-path
            if let ScalarExpr::Ref(name) = &args[0] {
                if let Some(r) = gpu_array_get_resident(name) {
                    if let Some(ref gpu_dev) = gpu {
                        let result = octoflow_vulkan::dispatch_map_deferred(gpu_dev, op, r)
                            .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                        return Ok(Some(ArrayResult::Resident(result)));
                    }
                }
            }
            let arr = extract_array_arg(fn_name, &args[0], arrays)?;
            let result = dispatch_gpu_map(gpu, op, &arr)?;
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── GPU element-wise with scalar param (Phase 79: resident fast-path) ──
        "gpu_scale" if args.len() == 2 => {
            let s = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let op = MapOp::Multiply(s);
            if let ScalarExpr::Ref(name) = &args[0] {
                if let Some(r) = gpu_array_get_resident(name) {
                    if let Some(ref gpu_dev) = gpu {
                        let result = octoflow_vulkan::dispatch_map_deferred(gpu_dev, op, r)
                            .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                        return Ok(Some(ArrayResult::Resident(result)));
                    }
                }
            }
            let arr = extract_array_arg("gpu_scale", &args[0], arrays)?;
            let result = dispatch_gpu_map(gpu, op, &arr)?;
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        "gpu_clamp" if args.len() == 3 => {
            let lo = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let hi = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let op = MapOp::Clamp(lo, hi);
            if let ScalarExpr::Ref(name) = &args[0] {
                if let Some(r) = gpu_array_get_resident(name) {
                    if let Some(ref gpu_dev) = gpu {
                        let result = octoflow_vulkan::dispatch_map_deferred(gpu_dev, op, r)
                            .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                        return Ok(Some(ArrayResult::Resident(result)));
                    }
                }
            }
            let arr = extract_array_arg("gpu_clamp", &args[0], arrays)?;
            let result = dispatch_gpu_map(gpu, op, &arr)?;
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        "gpu_pow" if args.len() == 2 => {
            let exp = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let op = MapOp::Pow(exp);
            if let ScalarExpr::Ref(name) = &args[0] {
                if let Some(r) = gpu_array_get_resident(name) {
                    if let Some(ref gpu_dev) = gpu {
                        let result = octoflow_vulkan::dispatch_map_deferred(gpu_dev, op, r)
                            .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                        return Ok(Some(ArrayResult::Resident(result)));
                    }
                }
            }
            let arr = extract_array_arg("gpu_pow", &args[0], arrays)?;
            let result = dispatch_gpu_map(gpu, op, &arr)?;
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── GPU element-wise binary ops (Phase 79: resident fast-path) ─────
        "gpu_add" | "gpu_sub" | "gpu_mul" | "gpu_div" if args.len() == 2 => {
            let op = match fn_name {
                "gpu_add" => BinaryOp::Add,
                "gpu_sub" => BinaryOp::Sub,
                "gpu_mul" => BinaryOp::Mul,
                _ => BinaryOp::Div,
            };
            // Resident fast-path: both inputs in VRAM → zero PCIe chaining
            if let (ScalarExpr::Ref(a_name), ScalarExpr::Ref(b_name)) = (&args[0], &args[1]) {
                let a_ref = gpu_array_get_resident(a_name);
                let b_ref = gpu_array_get_resident(b_name);
                if let (Some(ar), Some(br)) = (a_ref, b_ref) {
                    if let Some(ref gpu_dev) = gpu {
                        let result = octoflow_vulkan::dispatch_binop_deferred(gpu_dev, op, ar, br)
                            .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                        return Ok(Some(ArrayResult::Resident(result)));
                    }
                }
            }
            // Fallback: download + regular dispatch
            let a = extract_array_arg(fn_name, &args[0], arrays)?;
            let b = extract_array_arg(fn_name, &args[1], arrays)?;
            let result = dispatch_gpu_binop(gpu, op, &a, &b)?;
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── GPU conditional select (Phase 79: resident fast-path) ────────
        "gpu_where" if args.len() == 3 => {
            // Resident fast-path: all 3 inputs in VRAM → zero PCIe
            if let (ScalarExpr::Ref(c_name), ScalarExpr::Ref(a_name), ScalarExpr::Ref(b_name)) = (&args[0], &args[1], &args[2]) {
                let c_ref = gpu_array_get_resident(c_name);
                let a_ref = gpu_array_get_resident(a_name);
                let b_ref = gpu_array_get_resident(b_name);
                if let (Some(cr), Some(ar), Some(br)) = (c_ref, a_ref, b_ref) {
                    if let Some(ref gpu_dev) = gpu {
                        let result = octoflow_vulkan::dispatch_select_deferred(gpu_dev, cr, ar, br)
                            .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                        return Ok(Some(ArrayResult::Resident(result)));
                    }
                }
            }
            // Fallback
            let cond = extract_array_arg("gpu_where", &args[0], arrays)?;
            let a = extract_array_arg("gpu_where", &args[1], arrays)?;
            let b = extract_array_arg("gpu_where", &args[2], arrays)?;
            let result = dispatch_gpu_select(gpu, &cond, &a, &b)?;
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── GPU prefix scan ─────────────────────────────────────────────
        "gpu_cumsum" if args.len() == 1 => {
            let arr = extract_array_arg("gpu_cumsum", &args[0], arrays)?;
            let result = cpu_prefix_sum(&arr);
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── Matrix transpose ────────────────────────────────────────────
        "mat_transpose" if args.len() == 3 => {
            let arr = extract_array_arg("mat_transpose", &args[0], arrays)?;
            let rows = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
            let cols = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
            let result = cpu_transpose(&arr, rows, cols);
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── normalize(arr) — unit vector ────────────────────────────────
        "normalize" if args.len() == 1 => {
            let arr = extract_array_arg("normalize", &args[0], arrays)?;
            let sq_sum: f32 = arr.iter().map(|x| x * x).sum();
            let n = sq_sum.sqrt();
            let result: Vec<f32> = if n > 0.0 { arr.iter().map(|x| x / n).collect() } else { arr };
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── GPU utility: fill, range, reverse (Phase 79: auto-upload) ────
        "gpu_fill" if args.len() == 2 => {
            let val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let size = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
            let data = vec![val; size];
            if let Some(ref gpu_dev) = gpu {
                let buf = octoflow_vulkan::upload_buffer(gpu_dev, &data)
                    .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                Ok(Some(ArrayResult::Resident(buf)))
            } else {
                Ok(Some(ArrayResult::GpuFloats(data)))
            }
        }
        "gpu_range" if args.len() == 3 => {
            let start = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let end = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let step = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            if step == 0.0 { return Err(CliError::Compile("gpu_range(): step cannot be 0".into())); }
            let mut result = Vec::new();
            let mut v = start;
            if step > 0.0 {
                while v < end { result.push(v); v += step; }
            } else {
                while v > end { result.push(v); v += step; }
            }
            if let Some(ref gpu_dev) = gpu {
                let buf = octoflow_vulkan::upload_buffer(gpu_dev, &result)
                    .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                Ok(Some(ArrayResult::Resident(buf)))
            } else {
                Ok(Some(ArrayResult::GpuFloats(result)))
            }
        }
        // ── GPU filesystem: load data directly to VRAM (Phase 80) ─────────
        "gpu_load_csv" if args.len() == 1 => {
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let path = path_val.as_str().map_err(|_| CliError::Compile("gpu_load_csv() path must be a string".into()))?.to_string();
            check_read_permission_for(&path)?;
            let data = csv_read_floats(&path)?;
            if let Some(ref gpu_dev) = gpu {
                let buf = octoflow_vulkan::upload_buffer(gpu_dev, &data)
                    .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                Ok(Some(ArrayResult::Resident(buf)))
            } else {
                Ok(Some(ArrayResult::GpuFloats(data)))
            }
        }
        "gpu_load_binary" if args.len() == 1 => {
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let path = path_val.as_str().map_err(|_| CliError::Compile("gpu_load_binary() path must be a string".into()))?.to_string();
            check_read_permission_for(&path)?;
            let bytes = std::fs::read(&path)
                .map_err(|e| CliError::Io(format!("gpu_load_binary(\"{}\"): {}", path, e)))?;
            if bytes.len() % 4 != 0 {
                return Err(CliError::Compile(format!(
                    "gpu_load_binary(\"{}\"): file size {} not a multiple of 4 bytes (f32)", path, bytes.len())));
            }
            let data: Vec<f32> = bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if let Some(ref gpu_dev) = gpu {
                let buf = octoflow_vulkan::upload_buffer(gpu_dev, &data)
                    .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                Ok(Some(ArrayResult::Resident(buf)))
            } else {
                Ok(Some(ArrayResult::GpuFloats(data)))
            }
        }
        "gpu_reverse" if args.len() == 1 => {
            let arr = extract_array_arg("gpu_reverse", &args[0], arrays)?;
            let result: Vec<f32> = arr.into_iter().rev().collect();
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── GPU random, matmul, EMA (Phase 81) ──────────────────────────
        "gpu_random" if args.len() == 3 => {
            let n = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as usize;
            let lo = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let hi = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let seed = rng.get() as u32;
            rng.set(rng.get().wrapping_mul(6364136223846793005).wrapping_add(1));
            // CPU xorshift PRNG — GPU kernel removed with SPIR-V crate removal
            let range = hi - lo;
            let mut rng_state = seed as u64;
            if rng_state == 0 { rng_state = 0xDEADBEEF; }
            let result: Vec<f32> = (0..n).map(|_| {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let t = (rng_state as f32) / (u64::MAX as f32);
                lo + t * range
            }).collect();
            if let Some(ref gpu_dev) = gpu {
                let buf = octoflow_vulkan::upload_buffer(gpu_dev, &result)
                    .map_err(|e| CliError::Gpu(format!("{}", e)))?;
                Ok(Some(ArrayResult::Resident(buf)))
            } else {
                Ok(Some(ArrayResult::GpuFloats(result)))
            }
        }
        "gpu_matmul" if args.len() == 5 => {
            let a = extract_array_arg("gpu_matmul", &args[0], arrays)?;
            let b = extract_array_arg("gpu_matmul", &args[1], arrays)?;
            let m_f = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let n_f = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let k_f = eval_scalar(&args[4], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            // Validate dimensions are positive integers within safe limits
            for (name, val) in &[("m", m_f), ("n", n_f), ("k", k_f)] {
                if *val < 0.0 {
                    return Err(CliError::Gpu(format!(
                        "gpu_matmul(): {} = {} is negative — dimensions must be positive integers",
                        name, val
                    )));
                }
                if *val != val.trunc() {
                    return Err(CliError::Gpu(format!(
                        "gpu_matmul(): {} = {} is not an integer — dimensions must be whole numbers",
                        name, val
                    )));
                }
            }
            let m = m_f as u32;
            let n = n_f as u32;
            let k = k_f as u32;
            if (m as u64) * (k as u64) > 100_000_000 {
                return Err(CliError::Gpu(format!(
                    "gpu_matmul(): matrix A would be {}×{} = {} elements — max 100,000,000",
                    m, k, (m as u64) * (k as u64)
                )));
            }
            if (k as u64) * (n as u64) > 100_000_000 {
                return Err(CliError::Gpu(format!(
                    "gpu_matmul(): matrix B would be {}×{} = {} elements — max 100,000,000",
                    k, n, (k as u64) * (n as u64)
                )));
            }
            // Validate array lengths match declared dimensions
            let a_expected = (m as usize) * (k as usize);
            let b_expected = (k as usize) * (n as usize);
            if a.len() != a_expected {
                return Err(CliError::Gpu(format!(
                    "gpu_matmul(): matrix A has {} elements but m×k = {}×{} = {}",
                    a.len(), m, k, a_expected
                )));
            }
            if b.len() != b_expected {
                return Err(CliError::Gpu(format!(
                    "gpu_matmul(): matrix B has {} elements but k×n = {}×{} = {}",
                    b.len(), k, n, b_expected
                )));
            }
            let result = if let Some(ref gpu_dev) = gpu {
                octoflow_vulkan::dispatch_matmul(gpu_dev, &a, &b, m, n, k)
                    .unwrap_or_else(|_| cpu_matmul(&a, &b, m as usize, n as usize, k as usize))
            } else {
                cpu_matmul(&a, &b, m as usize, n as usize, k as usize)
            };
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── word_encode_hash: Rust-native character hash encoding ─────────
        // Replaces the .flow version that calls char_at per character (O(n²) allocs).
        // Single-pass iteration, zero intermediate allocations.
        "word_encode_hash" if args.len() == 2 => {
            let word_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let word = word_val.as_str().map_err(|_| CliError::Compile("word_encode_hash(): first arg must be a string".into()))?.to_string();
            let dim_f = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let embed_dim = dim_f as usize;
            if embed_dim == 0 || embed_dim > 4096 {
                return Err(CliError::Compile(format!("word_encode_hash(): embed_dim {} out of range (1..4096)", embed_dim)));
            }
            let mut vec = vec![0.0f32; embed_dim];
            for ch in word.chars() {
                let code = ch as u32 as f32;
                let raw_idx = code * 7.13;
                let dim_idx = (raw_idx - (raw_idx / embed_dim as f32).floor() * embed_dim as f32) as usize;
                if dim_idx < embed_dim {
                    vec[dim_idx] += 1.0;
                }
            }
            Ok(Some(ArrayResult::GpuFloats(vec)))
        }
        "gpu_ema" if args.len() == 2 => {
            let arr = extract_array_arg("gpu_ema", &args[0], arrays)?;
            let alpha = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()?;
            let result = cpu_temporal(TemporalOp::Ema(alpha), &arr);
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── GPU Tier 2: concat + gather (Phase 75b) ──────────────────────
        "gpu_concat" if args.len() == 2 => {
            let a = extract_array_arg("gpu_concat", &args[0], arrays)?;
            let b = extract_array_arg("gpu_concat", &args[1], arrays)?;
            let mut result = Vec::with_capacity(a.len() + b.len());
            result.extend_from_slice(&a);
            result.extend_from_slice(&b);
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        "gpu_gather" if args.len() == 2 => {
            let data = extract_array_arg("gpu_gather", &args[0], arrays)?;
            let indices = extract_array_arg("gpu_gather", &args[1], arrays)?;
            let n = data.len();
            let result: Vec<f32> = indices.iter().map(|&idx| {
                let i = idx as usize;
                if i < n { data[i] } else { 0.0 }
            }).collect();
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        "gpu_scatter" if args.len() == 3 => {
            let values = extract_array_arg("gpu_scatter", &args[0], arrays)?;
            let indices = extract_array_arg("gpu_scatter", &args[1], arrays)?;
            let dest_size_val = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let dest_size = dest_size_val.as_float()? as usize;
            let mut result = vec![0.0f32; dest_size];
            for (val, &idx) in values.iter().zip(indices.iter()) {
                let i = idx as usize;
                if i < dest_size { result[i] = *val; }
            }
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── sort — CPU array sort (S5: CPU fallback) ────────────────────
        "sort" | "gpu_sort" if args.len() == 1 => {
            let mut data = extract_array_arg("sort", &args[0], arrays)?;
            data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Ok(Some(ArrayResult::GpuFloats(data)))
        }
        // ── gpu_topk(arr, k) — top-K largest values (descending) ────────
        "gpu_topk" if args.len() == 2 => {
            let data = extract_array_arg("gpu_topk", &args[0], arrays)?;
            let k_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let k = k_val.as_float()? as usize;
            let k = k.min(data.len());
            if k == 0 || data.is_empty() {
                return Ok(Some(ArrayResult::GpuFloats(vec![])));
            }
            // Partial sort: O(n*k) for small k, faster than full sort
            let mut result = Vec::with_capacity(k);
            let mut used = vec![false; data.len()];
            for _ in 0..k {
                let mut best_idx = None;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in data.iter().enumerate() {
                    if !used[i] && (v > best_val || (v == best_val && best_idx.is_none())) {
                        best_val = v;
                        best_idx = Some(i);
                    }
                }
                if let Some(idx) = best_idx {
                    used[idx] = true;
                    result.push(best_val);
                }
            }
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── gpu_topk_indices(arr, k) — indices of top-K largest values ──
        "gpu_topk_indices" if args.len() == 2 => {
            let data = extract_array_arg("gpu_topk_indices", &args[0], arrays)?;
            let k_val = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let k = k_val.as_float()? as usize;
            let k = k.min(data.len());
            if k == 0 || data.is_empty() {
                return Ok(Some(ArrayResult::GpuFloats(vec![])));
            }
            let mut result = Vec::with_capacity(k);
            let mut used = vec![false; data.len()];
            for _ in 0..k {
                let mut best_idx = None;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in data.iter().enumerate() {
                    if !used[i] && (v > best_val || (v == best_val && best_idx.is_none())) {
                        best_val = v;
                        best_idx = Some(i);
                    }
                }
                if let Some(idx) = best_idx {
                    used[idx] = true;
                    result.push(idx as f32);
                }
            }
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── gpu_run — universal GPU dispatch (Phase 76) ─────────────────
        "gpu_run" if args.len() >= 2 => {
            // gpu_run(spv_path, arr1, arr2, ..., scalar_param1, scalar_param2, ...)
            // First arg: .spv file path (string)
            // Remaining args: arrays become input buffers, scalars become push constants
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let spv_path = path_val.as_str().map_err(|_| CliError::Compile("gpu_run() first arg must be a .spv file path string".into()))?;

            // Read the .spv file (cached)
            check_read_permission_for(spv_path)?;
            let spv_key = spv_path.to_string();
            let spirv = SPIRV_FILE_CACHE.with(|cache| {
                let c = cache.borrow();
                if let Some(data) = c.get(&spv_key) {
                    return Ok(data.clone());
                }
                drop(c);
                let data = std::fs::read(spv_path)
                    .map_err(|e| CliError::Io(format!("gpu_run(): cannot read \"{}\": {}", spv_path, e)))?;
                cache.borrow_mut().insert(spv_key, data.clone());
                Ok(data)
            })?;
            if spirv.len() % 4 != 0 {
                return Err(CliError::Compile(format!("gpu_run(): invalid .spv file \"{}\" (size not multiple of 4)", spv_path)));
            }

            // Classify remaining args: arrays → input buffers, scalars → push constants
            let mut input_arrays: Vec<Vec<f32>> = Vec::new();
            let mut push_constants: Vec<f32> = Vec::new();
            for arg in &args[1..] {
                if let ScalarExpr::Ref(name) = arg {
                    // Check GPU arrays first (zero-copy)
                    if let Some(gpu_arr) = gpu_array_get(name) {
                        input_arrays.push(gpu_arr);
                        continue;
                    }
                    if let Some(arr) = arrays.get(name) {
                        input_arrays.push(arr.iter().map(|v| v.as_float().unwrap_or(0.0)).collect());
                        continue;
                    }
                }
                // Not an array ref — evaluate as scalar push constant
                let val = eval_scalar(arg, streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
                push_constants.push(val.as_float().map_err(|_| {
                    CliError::Compile("gpu_run(): non-array args must be numeric (push constants)".into())
                })?);
            }

            if input_arrays.is_empty() {
                return Err(CliError::Compile("gpu_run() requires at least one input array".into()));
            }

            // Infer output size: default to first input size, or use push constants for mat_mul-style ops
            let output_size = input_arrays[0].len();

            let gpu_dev = gpu.as_ref().ok_or_else(|| {
                CliError::Gpu("gpu_run(): no GPU available (Vulkan not initialized)".into())
            })?;

            let result = octoflow_vulkan::gpu_run_dispatch(
                gpu_dev, &spirv, &input_arrays, &push_constants, output_size, None,
            ).map_err(|e| CliError::Gpu(format!("gpu_run(): {}", e)))?;

            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── grammar_mask — constrained decoding logit masking ──────
        // grammar_mask(logits, vocab) → masked logits array
        // Uses thread-local grammar state from grammar_load().
        // Invalid tokens get -inf, safety net unmarks argmax if all masked.
        "grammar_mask" if args.len() == 2 => {
            let logits = extract_array_arg("grammar_mask", &args[0], arrays)?;
            let vocab_arr = if let ScalarExpr::Ref(name) = &args[1] {
                arrays.get(name)
                    .ok_or_else(|| CliError::Compile(format!("grammar_mask: array '{}' not found", name)))?
                    .clone()
            } else {
                return Err(CliError::Compile("grammar_mask: second arg must be vocab array".into()));
            };
            let vocab: Vec<String> = vocab_arr.iter().map(|v| match v {
                Value::Str(s) => s.clone(),
                Value::Float(f) => format!("{}", f),
                Value::Int(i) => format!("{}", i),
                _ => String::new(),
            }).collect();
            let masked = grammar::GRAMMAR_STATE.with(|gs| {
                let state = gs.borrow();
                match state.as_ref() {
                    Some(s) => grammar::grammar_mask(s, &logits, &vocab),
                    None => logits.clone(), // no grammar loaded, pass through
                }
            });
            Ok(Some(ArrayResult::GpuFloats(masked)))
        }

        // ── gguf_matvec — fused tensor-cache + GPU matvec dispatch ──────
        // gguf_matvec(path, model, tensor_name, input_vec)
        // Loads weight from tensor cache (dequants on first call only),
        // dispatches matvec kernel on GPU, returns result array.
        // Weight tensor NEVER enters .flow array system — stays in Rust Vec<f32>.
        "gguf_matvec" if args.len() == 4 => {
            let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_str().map_err(|_| CliError::Compile("gguf_matvec: path must be string".into()))?.to_string();
            check_read_permission_for(&path)?;
            let model_name = if let ScalarExpr::Ref(name) = &args[1] {
                name.clone()
            } else {
                return Err(CliError::Compile("gguf_matvec: second arg must be map variable".into()));
            };
            let tname = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_str().map_err(|_| CliError::Compile("gguf_matvec: tensor name must be string".into()))?.to_string();
            let input = extract_array_arg("gguf_matvec", &args[3], arrays)?;
            let input_dim = input.len();

            // Get weight from tensor cache (or trigger dequant on first access)
            let cache_key = format!("{}:{}", path, tname);
            // Try fast path: use GPU-cached matvec with weight in VRAM
            let cached_result = TENSOR_CACHE.with(|tc| {
                let cache = tc.borrow();
                if let Some(weight) = cache.get(&cache_key) {
                    let output_dim = weight.len() / input_dim;
                    if output_dim == 0 || weight.len() != output_dim * input_dim {
                        return Some(Err(CliError::Runtime(format!(
                            "gguf_matvec: dimension mismatch: weight {} elements, input {} elements",
                            weight.len(), input_dim
                        ))));
                    }
                    let result = gpu_cached_matvec(&cache_key, weight, &input, output_dim, input_dim);
                    return Some(Ok(result));
                }
                None
            });
            if let Some(res) = cached_result {
                return Ok(Some(ArrayResult::GpuFloats(res?)));
            }
            // Cache miss — dequant and populate cache
            let weight = {
                    // First access: need to dequant. Use the full gguf_load_tensor path.
                    // Parse metadata from model map to get dimensions.
                    let model_map = hashmaps.get(&model_name).ok_or_else(|| {
                        CliError::Compile(format!("gguf_matvec: map '{}' not found", model_name))
                    })?;
                    let prefix = format!("t.{}", tname);
                    let total_count = model_map.get(&format!("{}.count", prefix))
                        .and_then(|v| v.as_float().ok())
                        .unwrap_or(0.0) as usize;
                    let tensor_type = model_map.get(&format!("{}.type", prefix))
                        .and_then(|v| v.as_float().ok())
                        .ok_or_else(|| CliError::Runtime(format!("gguf_matvec: tensor '{}' not found", tname)))? as u32;
                    let byte_size = match tensor_type {
                        0 => total_count * 4,
                        1 => total_count * 2,
                        12 => (total_count / 256) * 144,
                        13 => (total_count / 256) * 176,
                        14 => (total_count / 256) * 210,
                        _ => total_count * 4,
                    };
                    let ds_buf = model_map.get("_ds_buf")
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                    let hdr_buf = model_map.get("_hdr_buf")
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                    let off_pos = model_map.get(&format!("{}.off_pos", prefix))
                        .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as usize;
                    let ds_ptr = mem_table_get_ptr(ds_buf)?;
                    let data_start = unsafe { (ds_ptr as *const u64).read_unaligned() };
                    let hdr_ptr = mem_table_get_ptr(hdr_buf)?;
                    let tensor_offset = unsafe { (hdr_ptr.add(off_pos) as *const u64).read_unaligned() };
                    let file_offset = data_start + tensor_offset;

                    let raw: Vec<u8> = FILE_CACHE.with(|cache| -> Result<Vec<u8>, CliError> {
                        let c = cache.borrow();
                        if let Some(cached_bytes) = c.get(&path) {
                            let start = file_offset as usize;
                            let end = start + byte_size;
                            if end <= cached_bytes.len() {
                                return Ok(cached_bytes[start..end].to_vec());
                            }
                            return Err(CliError::Runtime(format!(
                                "gguf_matvec: offset {} + size {} exceeds cached file size {}",
                                start, byte_size, cached_bytes.len()
                            )));
                        }
                        drop(c);
                        use std::io::{Read, Seek, SeekFrom};
                        let mut file = std::fs::File::open(&path)
                            .map_err(|e| CliError::Io(format!("gguf_matvec: {}", e)))?;
                        file.seek(SeekFrom::Start(file_offset))
                            .map_err(|e| CliError::Io(format!("gguf_matvec: seek: {}", e)))?;
                        let mut buf = vec![0u8; byte_size];
                        file.read_exact(&mut buf)
                            .map_err(|e| CliError::Io(format!("gguf_matvec: read: {}", e)))?;
                        Ok(buf)
                    })?;

                    let dequanted: Vec<f32> = match tensor_type {
                        0 => raw.chunks_exact(4).take(total_count).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                        1 => raw.chunks_exact(2).take(total_count).map(|c| gguf_f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                        12 => {
                            let mut out = Vec::with_capacity(total_count + 256);
                            for block in raw.chunks(144) {
                                if block.len() < 144 { break; }
                                gguf_dequant_q4k_block(block, &mut out);
                            }
                            out.truncate(total_count);
                            out
                        }
                        13 => {
                            let mut out = Vec::with_capacity(total_count + 256);
                            for block in raw.chunks(176) {
                                if block.len() < 176 { break; }
                                gguf_dequant_q5k_block(block, &mut out);
                            }
                            out.truncate(total_count);
                            out
                        }
                        14 => {
                            let mut out = Vec::with_capacity(total_count + 256);
                            for block in raw.chunks(210) {
                                if block.len() < 210 { break; }
                                gguf_dequant_q6k_block(block, &mut out);
                            }
                            out.truncate(total_count);
                            out
                        }
                        _ => return Err(CliError::Runtime(format!("gguf_matvec: unsupported type {}", tensor_type))),
                    };
                    TENSOR_CACHE.with(|tc| tc.borrow_mut().insert(cache_key.clone(), dequanted.clone()));
                    dequanted
            };

            // First call: compute matvec from freshly dequanted weight via GPU
            let output_dim = weight.len() / input_dim;
            if output_dim == 0 || weight.len() != output_dim * input_dim {
                return Err(CliError::Runtime(format!(
                    "gguf_matvec: dimension mismatch: weight {} elements, input {} elements",
                    weight.len(), input_dim
                )));
            }
            let result = gpu_cached_matvec(&cache_key, &weight, &input, output_dim, input_dim);
            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── gguf_infer_layer — full transformer layer in Rust (v1.23) ──────
        // gguf_infer_layer(path, model, hidden, layer_idx, seq_pos, max_seq)
        // Runs complete transformer layer: RMSNorm → Q/K/V → bias → RoPE →
        // KV cache → multi-head attention → O proj → residual →
        // RMSNorm → gate/up → SiLU*up → down → residual.
        // KV cache stored in Rust-side thread-local (zero .flow overhead).
        // Weights loaded from TENSOR_CACHE (dequanted once on first call).
        "gguf_infer_layer" if args.len() == 6 => {
            let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_str().map_err(|_| CliError::Compile("gguf_infer_layer: path must be string".into()))?.to_string();
            check_read_permission_for(&path)?;
            let model_name = if let ScalarExpr::Ref(name) = &args[1] {
                name.clone()
            } else {
                return Err(CliError::Compile("gguf_infer_layer: second arg must be map variable".into()));
            };
            let hidden_in = extract_array_arg("gguf_infer_layer", &args[2], arrays)?;
            let layer_idx = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as usize;
            let seq_pos = eval_scalar(&args[4], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as usize;
            let max_seq = eval_scalar(&args[5], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as usize;

            // Read model dimensions from model map
            let model_map = hashmaps.get(&model_name).ok_or_else(|| {
                CliError::Compile(format!("gguf_infer_layer: map '{}' not found", model_name))
            })?;
            let get_f = |key: &str| -> f32 {
                model_map.get(key).and_then(|v| v.as_float().ok()).unwrap_or(0.0) as f32
            };
            let n_embd = get_f("n_embd") as usize;
            let n_head = get_f("n_head") as usize;
            let n_kv_head = {
                let v = get_f("n_kv_head") as usize;
                if v == 0 { n_head } else { v }
            };
            let n_ff = get_f("n_ff") as usize;
            // Read eps from kv.{arch}.attention.layer_norm_rms_epsilon
            let arch = model_map.get("arch")
                .and_then(|v| v.as_str().ok())
                .unwrap_or("llama".into()).to_string();
            let eps = {
                let key1 = format!("kv.{}.attention.layer_norm_rms_epsilon", arch);
                let key2 = "kv.llama.attention.layer_norm_rms_epsilon";
                let v = model_map.get(&key1).or_else(|| model_map.get(key2))
                    .and_then(|v| v.as_float().ok()).unwrap_or(0.0) as f32;
                if v == 0.0 { 0.00001f32 } else { v }
            };
            let rope_theta = {
                let key1 = format!("kv.{}.rope.freq_base", arch);
                let key2 = "kv.llama.rope.freq_base";
                let v = model_map.get(&key1).or_else(|| model_map.get(key2))
                    .and_then(|v| v.as_float().ok()).unwrap_or(0.0);
                if v == 0.0 { 10000.0f64 } else { v as f64 }
            };

            let head_dim = n_embd / n_head;
            let kv_dim = n_kv_head * head_dim;
            let heads_per_kv = if n_kv_head > 0 { n_head / n_kv_head } else { 1 };
            let inv_sqrt_dh = 1.0f32 / (head_dim as f32).sqrt();

            // Initialize KV cache on first call
            KV_CACHE.with(|kv| {
                let mut kv = kv.borrow_mut();
                if kv.is_none() {
                    let n_layer = get_f("n_layer") as usize;
                    let cache_size = n_layer * max_seq * kv_dim;
                    *kv = Some(InferKvCache {
                        k: vec![0.0f32; cache_size],
                        v: vec![0.0f32; cache_size],
                        n_layer,
                        max_seq,
                        kv_dim,
                    });
                }
            });

            // Load weight tensors: large matrices → GPU_BUFFER_CACHE only
            // Small tensors (norms, biases) → TENSOR_CACHE for CPU access
            let li = layer_idx.to_string();
            let weight_names: Vec<String> = vec![
                format!("blk.{}.attn_q.weight", li),     // 0
                format!("blk.{}.attn_k.weight", li),     // 1
                format!("blk.{}.attn_v.weight", li),     // 2
                format!("blk.{}.attn_output.weight", li), // 3
                format!("blk.{}.ffn_gate.weight", li),   // 4
                format!("blk.{}.ffn_up.weight", li),     // 5
                format!("blk.{}.ffn_down.weight", li),   // 6
            ];
            let small_names: Vec<String> = vec![
                format!("blk.{}.attn_norm.weight", li),  // 0
                format!("blk.{}.ffn_norm.weight", li),   // 1
            ];
            let bias_names: Vec<String> = vec![
                format!("blk.{}.attn_q.bias", li),
                format!("blk.{}.attn_k.bias", li),
                format!("blk.{}.attn_v.bias", li),
            ];
            let has_qbias = model_map.contains_key(&format!("t.blk.{}.attn_q.bias.type", li));

            // Build GPU cache keys early for fast-path check
            let gpu_keys: Vec<String> = weight_names.iter()
                .map(|wn| format!("{}:{}", path, wn))
                .collect();

            // Fast path: if ALL weights already in GPU_BUFFER_CACHE (decomposed pre-load),
            // skip ensure_gpu_buffer_cached entirely — avoids redundant function calls,
            // format! allocations, and HashMap lookups per weight per layer per token.
            let all_gpu_cached = GPU_BUFFER_CACHE.with(|gc| {
                let cache = gc.borrow();
                gpu_keys.iter().all(|k| cache.contains_key(k))
            });
            let all_small_cached = if all_gpu_cached {
                TENSOR_CACHE.with(|tc| {
                    let cache = tc.borrow();
                    let smalls_ok = small_names.iter().all(|sn| cache.contains_key(&format!("{}:{}", path, sn)));
                    let biases_ok = if has_qbias {
                        bias_names.iter().all(|bn| cache.contains_key(&format!("{}:{}", path, bn)))
                    } else { true };
                    smalls_ok && biases_ok
                })
            } else { false };

            if !all_gpu_cached || !all_small_cached {
                // Slow path: ensure weights are loaded (from GGUF or decomposed files)
                let verbose = VERBOSE_INFER.with(|c| c.get());
                for (wi, wn) in weight_names.iter().enumerate() {
                    if verbose { eprintln!("[infer L{} w{}/{}] loading {}", layer_idx, wi, weight_names.len(), wn); }
                    ensure_gpu_buffer_cached(&path, model_map, wn)?;
                    if verbose { eprintln!("[infer L{} w{}/{}] OK", layer_idx, wi, weight_names.len()); }
                }
                for sn in &small_names {
                    ensure_tensor_cached(&path, model_map, sn)?;
                }
                if has_qbias {
                    for bn in &bias_names {
                        ensure_tensor_cached(&path, model_map, bn)?;
                    }
                }
            }

            // ── Compute layer: GPU matvec + CPU norm/attention/activation ──
            // Small tensors borrowed from TENSOR_CACHE
            let (attn_norm_w, ffn_norm_w, bias_data) = TENSOR_CACHE.with(|tc| {
                let cache = tc.borrow();
                let get = |name: &str| -> Vec<f32> {
                    cache.get(&format!("{}:{}", path, name)).unwrap().clone()
                };
                let an = get(&small_names[0]);
                let fn_n = get(&small_names[1]);
                let biases = if has_qbias {
                    Some((get(&bias_names[0]), get(&bias_names[1]), get(&bias_names[2])))
                } else {
                    None
                };
                (an, fn_n, biases)
            });

            let mut h = hidden_in;

            // Diagnostic disabled for speed
            let diag = false;
            let l2_norm = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };

            // ── 1. Attention sublayer ──────────────────────────────
            let normed = fast_rmsnorm(&h, &attn_norm_w, eps);

            // Q, K, V projections — dispatched from GPU_BUFFER_CACHE
            let mut q = gpu_matvec_from_cache(&gpu_keys[0], &normed, n_embd, n_embd)
                .unwrap_or_else(|| fast_matvec_fallback(&path, &weight_names[0], &normed, n_embd, n_embd));
            let mut k = gpu_matvec_from_cache(&gpu_keys[1], &normed, kv_dim, n_embd)
                .unwrap_or_else(|| fast_matvec_fallback(&path, &weight_names[1], &normed, kv_dim, n_embd));
            let mut v = gpu_matvec_from_cache(&gpu_keys[2], &normed, kv_dim, n_embd)
                .unwrap_or_else(|| fast_matvec_fallback(&path, &weight_names[2], &normed, kv_dim, n_embd));

            let _q_pre = l2_norm(&q);
            let _k_pre = l2_norm(&k);

            // Add biases
            if let Some((ref qb, ref kb, ref vb)) = bias_data {
                for i in 0..n_embd { q[i] += qb[i]; }
                for i in 0..kv_dim { k[i] += kb[i]; }
                for i in 0..kv_dim { v[i] += vb[i]; }
            }

            // RoPE
            fast_rope(&mut q, seq_pos, head_dim, n_head, rope_theta);
            fast_rope(&mut k, seq_pos, head_dim, n_kv_head, rope_theta);

            if diag { eprintln!("  [L{} pos{}] after rope: Q norm={:.4} K norm={:.4} (theta={:.0})", layer_idx, seq_pos, l2_norm(&q), l2_norm(&k), rope_theta); }

            // Store K, V in KV cache + compute attention
            let attn_concat = KV_CACHE.with(|kv_cell| -> Vec<f32> {
                let mut kv = kv_cell.borrow_mut();
                let kv = kv.as_mut().unwrap();
                let cache_base = layer_idx * kv.max_seq * kv.kv_dim + seq_pos * kv.kv_dim;
                for d in 0..kv_dim {
                    kv.k[cache_base + d] = k[d];
                    kv.v[cache_base + d] = v[d];
                }
                let seq_len = seq_pos + 1;
                let mut concat = Vec::with_capacity(n_embd);
                for qh in 0..n_head {
                    let kvh = qh / heads_per_kv;
                    let q_offset = qh * head_dim;
                    let kv_head_offset = kvh * head_dim;
                    let mut score_max = f32::NEG_INFINITY;
                    let mut scores = Vec::with_capacity(seq_len);
                    for t in 0..seq_len {
                        let k_base = layer_idx * kv.max_seq * kv.kv_dim + t * kv.kv_dim + kv_head_offset;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_offset + d] * kv.k[k_base + d];
                        }
                        let scaled = dot * inv_sqrt_dh;
                        if scaled > score_max { score_max = scaled; }
                        scores.push(scaled);
                    }
                    let mut exp_sum = 0.0f32;
                    let mut weights = Vec::with_capacity(seq_len);
                    for t in 0..seq_len {
                        let e = (scores[t] - score_max).exp();
                        weights.push(e);
                        exp_sum += e;
                    }
                    let inv_exp_sum = 1.0 / exp_sum;
                    for d in 0..head_dim {
                        let mut weighted_val = 0.0f32;
                        for t in 0..seq_len {
                            let v_base = layer_idx * kv.max_seq * kv.kv_dim + t * kv.kv_dim + kv_head_offset;
                            weighted_val += weights[t] * inv_exp_sum * kv.v[v_base + d];
                        }
                        concat.push(weighted_val);
                    }
                }
                concat
            });

            if diag { eprintln!("  [L{} pos{}] attn_concat norm={:.4}", layer_idx, seq_pos, l2_norm(&attn_concat)); }

            // Output projection — dispatched from GPU_BUFFER_CACHE
            let attn_out = gpu_matvec_from_cache(&gpu_keys[3], &attn_concat, n_embd, n_embd)
                .unwrap_or_else(|| fast_matvec_fallback(&path, &weight_names[3], &attn_concat, n_embd, n_embd));
            for i in 0..n_embd {
                h[i] += attn_out[i];
            }

            if diag { eprintln!("  [L{} pos{}] h_post_attn norm={:.4}", layer_idx, seq_pos, l2_norm(&h)); }

            // ── 2. FFN sublayer ───────────────────────────────────
            let fn_normed = fast_rmsnorm(&h, &ffn_norm_w, eps);

            let gate = gpu_matvec_from_cache(&gpu_keys[4], &fn_normed, n_ff, n_embd)
                .unwrap_or_else(|| fast_matvec_fallback(&path, &weight_names[4], &fn_normed, n_ff, n_embd));
            let up = gpu_matvec_from_cache(&gpu_keys[5], &fn_normed, n_ff, n_embd)
                .unwrap_or_else(|| fast_matvec_fallback(&path, &weight_names[5], &fn_normed, n_ff, n_embd));

            // SiLU(gate) * up
            let mut mid = vec![0.0f32; n_ff];
            for i in 0..n_ff {
                let x = gate[i];
                let silu = x / (1.0 + (-x).exp());
                mid[i] = silu * up[i];
            }

            let down = gpu_matvec_from_cache(&gpu_keys[6], &mid, n_embd, n_ff)
                .unwrap_or_else(|| fast_matvec_fallback(&path, &weight_names[6], &mid, n_embd, n_ff));
            for i in 0..n_embd {
                h[i] += down[i];
            }

            if diag { eprintln!("  [L{} pos{}] h_out norm={:.4} min={:.6} max={:.6}", layer_idx, seq_pos, l2_norm(&h), h.iter().cloned().fold(f32::INFINITY, f32::min), h.iter().cloned().fold(f32::NEG_INFINITY, f32::max)); }

            let result = h;

            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── gguf_load_tensor — OS-boundary GGUF dequant (v1.22) ──────────
        // gguf_load_tensor(file_path, model_map, tensor_name [, row_id])
        // 3 args: load full tensor. 4 args: load single row (row_id * n_cols).
        // Reads tensor metadata from model map, reads file, dequants to f32 array.
        "gguf_load_tensor" if args.len() == 3 || args.len() == 4 => {
            // Evaluate all scalar args BEFORE borrowing hashmaps
            let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_str().map_err(|_| CliError::Compile("gguf_load_tensor: path must be a string".into()))?.to_string();
            check_read_permission_for(&path)?;
            let model_name = if let ScalarExpr::Ref(name) = &args[1] {
                name.clone()
            } else {
                return Err(CliError::Compile("gguf_load_tensor: second arg must be a map variable".into()));
            };
            let tname = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_str().map_err(|_| CliError::Compile("gguf_load_tensor: tensor name must be a string".into()))?.to_string();
            let row_id: Option<u64> = if args.len() == 4 {
                Some(eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?.as_float()? as u64)
            } else { None };

            // NOW borrow the model map (no more eval_scalar calls after this)
            let model_map = hashmaps.get(&model_name).ok_or_else(|| {
                CliError::Compile(format!("gguf_load_tensor: map '{}' not found", model_name))
            })?;

            // Read tensor metadata from model map (flat key pattern: t.NAME.type, t.NAME.count, etc.)
            let prefix = format!("t.{}", tname);
            let tensor_type = model_map.get(&format!("{}.type", prefix))
                .and_then(|v| v.as_float().ok())
                .ok_or_else(|| CliError::Runtime(format!("gguf_load_tensor: tensor '{}' not found in model", tname)))? as u32;
            let total_count = model_map.get(&format!("{}.count", prefix))
                .and_then(|v| v.as_float().ok())
                .unwrap_or(0.0) as usize;
            // dim0 = innermost dimension (row width for 2D tensors)
            let dim0 = model_map.get(&format!("{}.dim0", prefix))
                .and_then(|v| v.as_float().ok())
                .unwrap_or(total_count as f32) as usize;

            // For row extraction: compute what slice of data to load
            let (count, row_byte_offset) = if let Some(rid) = row_id {
                // Load one row: dim0 elements starting at row rid
                let row_elements = dim0;
                let row_byte_off: u64 = match tensor_type {
                    0 => rid * (dim0 as u64) * 4,    // F32: 4 bytes per element
                    1 => rid * (dim0 as u64) * 2,    // F16: 2 bytes per element
                    12 => {
                        // Q4_K: 256 elements per 144-byte block, rows must be block-aligned
                        let row_start_elem = rid * (dim0 as u64);
                        let block_idx = row_start_elem / 256;
                        block_idx * 144
                    }
                    14 => {
                        let row_start_elem = rid * (dim0 as u64);
                        let block_idx = row_start_elem / 256;
                        block_idx * 210
                    }
                    _ => rid * (dim0 as u64) * 4,
                };
                (row_elements, row_byte_off)
            } else {
                (total_count, 0u64)
            };

            // Check tensor cache (full tensor loads only, not row extractions)
            if row_id.is_none() {
                let cache_key = format!("{}:{}", path, tname);
                let cached = TENSOR_CACHE.with(|tc| {
                    tc.borrow().get(&cache_key).cloned()
                });
                if let Some(cached_floats) = cached {
                    return Ok(Some(ArrayResult::GpuFloats(cached_floats)));
                }
            }

            // Compute byte size from type + count
            let byte_size = match tensor_type {
                0 => count * 4,           // F32
                1 => count * 2,           // F16
                12 => {
                    // Q4_K: might need extra blocks for row extraction
                    if row_id.is_some() {
                        let rid = row_id.unwrap();
                        let row_start = rid * (dim0 as u64);
                        let row_end = row_start + (dim0 as u64);
                        let block_start = row_start / 256;
                        let block_end = (row_end + 255) / 256;
                        ((block_end - block_start) as usize) * 144
                    } else {
                        (count / 256) * 144
                    }
                }
                14 => {
                    if row_id.is_some() {
                        let rid = row_id.unwrap();
                        let row_start = rid * (dim0 as u64);
                        let row_end = row_start + (dim0 as u64);
                        let block_start = row_start / 256;
                        let block_end = (row_end + 255) / 256;
                        ((block_end - block_start) as usize) * 210
                    } else {
                        (count / 256) * 210
                    }
                }
                _ => count * 4,
            };

            // Compute u64 file offset: data_start + tensor_offset + row_byte_offset
            let ds_buf = model_map.get("_ds_buf")
                .and_then(|v| v.as_float().ok())
                .ok_or_else(|| CliError::Runtime("gguf_load_tensor: _ds_buf not in model".into()))? as usize;
            let hdr_buf = model_map.get("_hdr_buf")
                .and_then(|v| v.as_float().ok())
                .ok_or_else(|| CliError::Runtime("gguf_load_tensor: _hdr_buf not in model".into()))? as usize;
            let off_pos = model_map.get(&format!("{}.off_pos", prefix))
                .and_then(|v| v.as_float().ok())
                .unwrap_or(0.0) as usize;

            let ds_ptr = mem_table_get_ptr(ds_buf)?;
            let data_start = unsafe { (ds_ptr as *const u64).read_unaligned() };
            let hdr_ptr = mem_table_get_ptr(hdr_buf)?;
            let tensor_offset = unsafe { (hdr_ptr.add(off_pos) as *const u64).read_unaligned() };
            let file_offset = data_start + tensor_offset + row_byte_offset;

            // Read raw bytes — from file cache if available, else from disk
            let raw: Vec<u8> = FILE_CACHE.with(|cache| -> Result<Vec<u8>, CliError> {
                let c = cache.borrow();
                if let Some(cached_bytes) = c.get(&path) {
                    // Fast path: slice from cached file bytes
                    let start = file_offset as usize;
                    let end = start + byte_size;
                    if end <= cached_bytes.len() {
                        return Ok(cached_bytes[start..end].to_vec());
                    }
                    return Err(CliError::Runtime(format!(
                        "gguf_load_tensor: offset {} + size {} exceeds cached file size {}",
                        start, byte_size, cached_bytes.len()
                    )));
                }
                // Slow path: read from disk
                drop(c);
                use std::io::{Read, Seek, SeekFrom};
                let mut file = std::fs::File::open(&path)
                    .map_err(|e| CliError::Io(format!("gguf_load_tensor(\"{}\"): {}", path, e)))?;
                file.seek(SeekFrom::Start(file_offset))
                    .map_err(|e| CliError::Io(format!("gguf_load_tensor: seek to {}: {}", file_offset, e)))?;
                let mut buf = vec![0u8; byte_size];
                file.read_exact(&mut buf)
                    .map_err(|e| CliError::Io(format!("gguf_load_tensor: read {} bytes: {}", byte_size, e)))?;
                Ok(buf)
            })?;

            let result: Vec<f32> = match tensor_type {
                0 => {
                    // F32: reinterpret bytes as f32
                    raw.chunks_exact(4).take(count).map(|c| {
                        f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    }).collect()
                }
                1 => {
                    // F16: decode each 2-byte value
                    raw.chunks_exact(2).take(count).map(|c| {
                        gguf_f16_to_f32(u16::from_le_bytes([c[0], c[1]]))
                    }).collect()
                }
                12 => {
                    // Q4_K: 256 elements per 144-byte block
                    let mut out = Vec::with_capacity(count + 256);
                    for block in raw.chunks(144) {
                        if block.len() < 144 { break; }
                        gguf_dequant_q4k_block(block, &mut out);
                    }
                    // Row mode: skip elements before row start within first block
                    if let Some(rid) = row_id {
                        let row_start_elem = rid * (dim0 as u64);
                        let block_start_elem = (row_start_elem / 256) * 256;
                        let skip = (row_start_elem - block_start_elem) as usize;
                        if skip > 0 { out.drain(..skip); }
                    }
                    out.truncate(count);
                    out
                }
                14 => {
                    // Q6_K: 256 elements per 210-byte block
                    let mut out = Vec::with_capacity(count + 256);
                    for block in raw.chunks(210) {
                        if block.len() < 210 { break; }
                        gguf_dequant_q6k_block(block, &mut out);
                    }
                    if let Some(rid) = row_id {
                        let row_start_elem = rid * (dim0 as u64);
                        let block_start_elem = (row_start_elem / 256) * 256;
                        let skip = (row_start_elem - block_start_elem) as usize;
                        if skip > 0 { out.drain(..skip); }
                    }
                    out.truncate(count);
                    out
                }
                _ => {
                    return Err(CliError::Runtime(format!(
                        "gguf_load_tensor: unsupported tensor type {}", tensor_type
                    )));
                }
            };

            // Cache full tensor loads for reuse across tokens
            if row_id.is_none() {
                let cache_key = format!("{}:{}", path, tname);
                TENSOR_CACHE.with(|tc| {
                    tc.borrow_mut().insert(cache_key, result.clone());
                });
            }

            Ok(Some(ArrayResult::GpuFloats(result)))
        }
        // ── rt_load_file_to_buffer — read raw bytes as f32 array (v1.32) ──
        // rt_load_file_to_buffer(path, byte_offset, byte_count) → array of floats (0-255)
        // Each byte becomes one f32 element. Returns GpuFloats for zero-copy GPU upload.
        // Uses FILE_CACHE fast path if file is already cached.
        "rt_load_file_to_buffer" if args.len() == 3 => {
            let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_str().map_err(|_| CliError::Compile("rt_load_file_to_buffer: path must be a string".into()))?.to_string();
            check_read_permission_for(&path)?;
            let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float().map_err(|_| CliError::Compile("rt_load_file_to_buffer: offset must be numeric".into()))? as u64;
            let count = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float().map_err(|_| CliError::Compile("rt_load_file_to_buffer: count must be numeric".into()))? as usize;

            if count == 0 {
                return Ok(Some(ArrayResult::GpuFloats(Vec::new())));
            }

            // Read raw bytes — from file cache if available, else from disk
            let raw: Vec<u8> = FILE_CACHE.with(|cache| -> Result<Vec<u8>, CliError> {
                let c = cache.borrow();
                if let Some(cached_bytes) = c.get(&path) {
                    let start = offset as usize;
                    let end = start + count;
                    if end <= cached_bytes.len() {
                        return Ok(cached_bytes[start..end].to_vec());
                    }
                    return Err(CliError::Runtime(format!(
                        "rt_load_file_to_buffer: offset {} + count {} exceeds file size {}",
                        start, count, cached_bytes.len()
                    )));
                }
                drop(c);
                use std::io::{Read, Seek, SeekFrom};
                let mut file = std::fs::File::open(&path)
                    .map_err(|e| CliError::Io(format!("rt_load_file_to_buffer(\"{}\"): {}", path, e)))?;
                file.seek(SeekFrom::Start(offset))
                    .map_err(|e| CliError::Io(format!("rt_load_file_to_buffer: seek to {}: {}", offset, e)))?;
                let mut buf = vec![0u8; count];
                file.read_exact(&mut buf)
                    .map_err(|e| CliError::Io(format!("rt_load_file_to_buffer: read {} bytes: {}", count, e)))?;
                Ok(buf)
            })?;

            // Convert each byte to f32 (0.0-255.0)
            let floats: Vec<f32> = raw.iter().map(|&b| b as f32).collect();
            Ok(Some(ArrayResult::GpuFloats(floats)))
        }
        // ── gguf_load_vocab — OS-boundary vocab string array loader (v1.23) ──
        // gguf_load_vocab(file_path) → array of strings (the vocabulary)
        // Reads the tokenizer.ggml.tokens string array from a GGUF file.
        // Returns Vec<Value::Str> — one string per token ID.
        "gguf_load_vocab" if args.len() == 1 => {
            let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_str().map_err(|_| CliError::Compile("gguf_load_vocab: path must be a string".into()))?.to_string();
            check_read_permission_for(&path)?;

            use std::io::{Read, Seek, SeekFrom};
            let mut file = std::fs::File::open(&path)
                .map_err(|e| CliError::Io(format!("gguf_load_vocab(\"{}\"): {}", path, e)))?;

            // Read GGUF header (24 bytes): magic(4) + version(4) + tensor_count(8) + kv_count(8)
            let mut hdr = [0u8; 24];
            file.read_exact(&mut hdr)
                .map_err(|e| CliError::Io(format!("gguf_load_vocab: read header: {}", e)))?;
            let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
            if magic != 0x46554747 {
                return Err(CliError::Runtime("gguf_load_vocab: invalid GGUF magic".into()));
            }
            let kv_count = u64::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19], hdr[20], hdr[21], hdr[22], hdr[23]]);

            // Walk KV pairs to find tokenizer.ggml.tokens
            let target_key = "tokenizer.ggml.tokens";
            let mut found = false;
            let mut result: Vec<Value> = Vec::new();

            for _kv_idx in 0..kv_count {
                // Read key: u64 length + bytes
                let mut len_buf = [0u8; 8];
                file.read_exact(&mut len_buf)
                    .map_err(|e| CliError::Io(format!("gguf_load_vocab: read key len: {}", e)))?;
                let key_len = u64::from_le_bytes(len_buf) as usize;
                let mut key_buf = vec![0u8; key_len];
                file.read_exact(&mut key_buf)
                    .map_err(|e| CliError::Io(format!("gguf_load_vocab: read key: {}", e)))?;
                let key = String::from_utf8_lossy(&key_buf).to_string();

                // Read value type (u32)
                let mut type_buf = [0u8; 4];
                file.read_exact(&mut type_buf)
                    .map_err(|e| CliError::Io(format!("gguf_load_vocab: read type: {}", e)))?;
                let val_type = u32::from_le_bytes(type_buf);

                if key == target_key && val_type == 9 {
                    // Array type! Read element type + count
                    let mut arr_hdr = [0u8; 12];
                    file.read_exact(&mut arr_hdr)
                        .map_err(|e| CliError::Io(format!("gguf_load_vocab: read array header: {}", e)))?;
                    let elem_type = u32::from_le_bytes([arr_hdr[0], arr_hdr[1], arr_hdr[2], arr_hdr[3]]);
                    let arr_count = u64::from_le_bytes([arr_hdr[4], arr_hdr[5], arr_hdr[6], arr_hdr[7], arr_hdr[8], arr_hdr[9], arr_hdr[10], arr_hdr[11]]);

                    if elem_type != 8 {
                        return Err(CliError::Runtime(format!(
                            "gguf_load_vocab: expected string array (type 8), got type {}", elem_type
                        )));
                    }

                    // Read strings
                    result.reserve(arr_count as usize);
                    for _si in 0..arr_count {
                        let mut slen_buf = [0u8; 8];
                        file.read_exact(&mut slen_buf)
                            .map_err(|e| CliError::Io(format!("gguf_load_vocab: read str len: {}", e)))?;
                        let slen = u64::from_le_bytes(slen_buf) as usize;
                        let mut sbuf = vec![0u8; slen];
                        file.read_exact(&mut sbuf)
                            .map_err(|e| CliError::Io(format!("gguf_load_vocab: read str: {}", e)))?;
                        result.push(Value::Str(String::from_utf8_lossy(&sbuf).to_string()));
                    }
                    found = true;
                    break;
                }

                // Skip this KV value (macro-like helper to wrap io errors)
                let io_err = |e: std::io::Error| CliError::Io(format!("gguf_load_vocab: {}", e));
                match val_type {
                    0 | 1 | 7 => { file.seek(SeekFrom::Current(1)).map_err(&io_err)?; }
                    2 | 3 => { file.seek(SeekFrom::Current(2)).map_err(&io_err)?; }
                    4 | 5 | 6 => { file.seek(SeekFrom::Current(4)).map_err(&io_err)?; }
                    10 | 11 | 12 => { file.seek(SeekFrom::Current(8)).map_err(&io_err)?; }
                    8 => {
                        let mut sl = [0u8; 8];
                        file.read_exact(&mut sl).map_err(&io_err)?;
                        let slen = u64::from_le_bytes(sl);
                        file.seek(SeekFrom::Current(slen as i64)).map_err(&io_err)?;
                    }
                    9 => {
                        let mut ah = [0u8; 12];
                        file.read_exact(&mut ah).map_err(&io_err)?;
                        let et = u32::from_le_bytes([ah[0], ah[1], ah[2], ah[3]]);
                        let ac = u64::from_le_bytes([ah[4], ah[5], ah[6], ah[7], ah[8], ah[9], ah[10], ah[11]]);
                        match et {
                            0 | 1 | 7 => { file.seek(SeekFrom::Current(ac as i64)).map_err(&io_err)?; }
                            2 | 3 => { file.seek(SeekFrom::Current(ac as i64 * 2)).map_err(&io_err)?; }
                            4 | 5 | 6 => { file.seek(SeekFrom::Current(ac as i64 * 4)).map_err(&io_err)?; }
                            10 | 11 | 12 => { file.seek(SeekFrom::Current(ac as i64 * 8)).map_err(&io_err)?; }
                            8 => {
                                for _ in 0..ac {
                                    let mut sl2 = [0u8; 8];
                                    file.read_exact(&mut sl2).map_err(&io_err)?;
                                    let slen2 = u64::from_le_bytes(sl2);
                                    file.seek(SeekFrom::Current(slen2 as i64)).map_err(&io_err)?;
                                }
                            }
                            _ => { return Err(CliError::Runtime(format!("gguf_load_vocab: unsupported array element type {}", et))); }
                        }
                    }
                    _ => { return Err(CliError::Runtime(format!("gguf_load_vocab: unsupported KV type {}", val_type))); }
                }
            }

            if !found {
                return Err(CliError::Runtime("gguf_load_vocab: tokenizer.ggml.tokens not found in GGUF file".into()));
            }

            Ok(Some(ArrayResult::Values(result)))
        }
        // ── gguf_tokenize — GPT-2 byte-level BPE tokenizer (v1.24) ──
        // gguf_tokenize(file_path, text) → array of float token IDs
        // Reads vocab + merges from GGUF, applies byte-level BPE encoding.
        // Returns token IDs as f32 array. Does NOT auto-prepend BOS.
        "gguf_tokenize" if args.len() == 2 => {
            let path = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_str().map_err(|_| CliError::Compile("gguf_tokenize: path must be a string".into()))?.to_string();
            check_read_permission_for(&path)?;
            let text = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_str().map_err(|_| CliError::Compile("gguf_tokenize: text must be a string".into()))?.to_string();

            if text.is_empty() {
                return Ok(Some(ArrayResult::GpuFloats(Vec::new())));
            }

            use std::io::{Read, Seek, SeekFrom};

            let io_err = |e: std::io::Error| CliError::Io(format!("gguf_tokenize: {}", e));
            let mut file = std::fs::File::open(&path)
                .map_err(|e| CliError::Io(format!("gguf_tokenize(\"{}\"): {}", path, e)))?;

            // Read GGUF header (24 bytes)
            let mut hdr = [0u8; 24];
            file.read_exact(&mut hdr).map_err(&io_err)?;
            let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
            if magic != 0x46554747 {
                return Err(CliError::Runtime("gguf_tokenize: invalid GGUF magic".into()));
            }
            let kv_count = u64::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19], hdr[20], hdr[21], hdr[22], hdr[23]]);

            // Walk KV pairs — collect tokens and merges in single pass
            let mut vocab: Vec<String> = Vec::new();
            let mut merges_raw: Vec<String> = Vec::new();
            let mut found_tokens = false;
            let mut found_merges = false;

            for _ in 0..kv_count {
                if found_tokens && found_merges { break; }

                // Read key
                let mut len_buf = [0u8; 8];
                file.read_exact(&mut len_buf).map_err(&io_err)?;
                let key_len = u64::from_le_bytes(len_buf) as usize;
                let mut key_buf = vec![0u8; key_len];
                file.read_exact(&mut key_buf).map_err(&io_err)?;
                let key = String::from_utf8_lossy(&key_buf).to_string();

                // Read value type
                let mut type_buf = [0u8; 4];
                file.read_exact(&mut type_buf).map_err(&io_err)?;
                let val_type = u32::from_le_bytes(type_buf);

                // Check target keys (both are string arrays = type 9)
                let is_tokens = key == "tokenizer.ggml.tokens" && val_type == 9;
                let is_merges = key == "tokenizer.ggml.merges" && val_type == 9;

                if is_tokens || is_merges {
                    let mut arr_hdr = [0u8; 12];
                    file.read_exact(&mut arr_hdr).map_err(&io_err)?;
                    let elem_type = u32::from_le_bytes([arr_hdr[0], arr_hdr[1], arr_hdr[2], arr_hdr[3]]);
                    let arr_count = u64::from_le_bytes([arr_hdr[4], arr_hdr[5], arr_hdr[6], arr_hdr[7], arr_hdr[8], arr_hdr[9], arr_hdr[10], arr_hdr[11]]);
                    if elem_type != 8 {
                        return Err(CliError::Runtime(format!("gguf_tokenize: {} expected string array, got type {}", key, elem_type)));
                    }
                    let mut strings = Vec::with_capacity(arr_count as usize);
                    for _ in 0..arr_count {
                        let mut sl = [0u8; 8];
                        file.read_exact(&mut sl).map_err(&io_err)?;
                        let slen = u64::from_le_bytes(sl) as usize;
                        let mut sb = vec![0u8; slen];
                        file.read_exact(&mut sb).map_err(&io_err)?;
                        strings.push(String::from_utf8_lossy(&sb).to_string());
                    }
                    if is_tokens { vocab = strings; found_tokens = true; }
                    else { merges_raw = strings; found_merges = true; }
                    continue;
                }

                // Skip this KV value
                match val_type {
                    0 | 1 | 7 => { file.seek(SeekFrom::Current(1)).map_err(&io_err)?; }
                    2 | 3 => { file.seek(SeekFrom::Current(2)).map_err(&io_err)?; }
                    4 | 5 | 6 => { file.seek(SeekFrom::Current(4)).map_err(&io_err)?; }
                    10 | 11 | 12 => { file.seek(SeekFrom::Current(8)).map_err(&io_err)?; }
                    8 => {
                        let mut sl = [0u8; 8];
                        file.read_exact(&mut sl).map_err(&io_err)?;
                        let slen = u64::from_le_bytes(sl);
                        file.seek(SeekFrom::Current(slen as i64)).map_err(&io_err)?;
                    }
                    9 => {
                        let mut ah = [0u8; 12];
                        file.read_exact(&mut ah).map_err(&io_err)?;
                        let et = u32::from_le_bytes([ah[0], ah[1], ah[2], ah[3]]);
                        let ac = u64::from_le_bytes([ah[4], ah[5], ah[6], ah[7], ah[8], ah[9], ah[10], ah[11]]);
                        match et {
                            0 | 1 | 7 => { file.seek(SeekFrom::Current(ac as i64)).map_err(&io_err)?; }
                            2 | 3 => { file.seek(SeekFrom::Current(ac as i64 * 2)).map_err(&io_err)?; }
                            4 | 5 | 6 => { file.seek(SeekFrom::Current(ac as i64 * 4)).map_err(&io_err)?; }
                            10 | 11 | 12 => { file.seek(SeekFrom::Current(ac as i64 * 8)).map_err(&io_err)?; }
                            8 => {
                                for _ in 0..ac {
                                    let mut sl2 = [0u8; 8];
                                    file.read_exact(&mut sl2).map_err(&io_err)?;
                                    let slen2 = u64::from_le_bytes(sl2);
                                    file.seek(SeekFrom::Current(slen2 as i64)).map_err(&io_err)?;
                                }
                            }
                            _ => { return Err(CliError::Runtime(format!("gguf_tokenize: unsupported array elem type {}", et))); }
                        }
                    }
                    _ => { return Err(CliError::Runtime(format!("gguf_tokenize: unsupported KV type {}", val_type))); }
                }
            }

            if !found_tokens {
                return Err(CliError::Runtime("gguf_tokenize: tokenizer.ggml.tokens not found".into()));
            }
            if !found_merges {
                return Err(CliError::Runtime("gguf_tokenize: tokenizer.ggml.merges not found".into()));
            }

            // Build token→ID lookup
            let mut token_to_id: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::with_capacity(vocab.len());
            for (i, tok) in vocab.iter().enumerate() {
                token_to_id.insert(tok.as_str(), i);
            }

            // Build merge rank lookup (key = "a b" merge string, value = priority rank)
            let mut merge_rank: std::collections::HashMap<String, usize> =
                std::collections::HashMap::with_capacity(merges_raw.len());
            for (rank, m) in merges_raw.into_iter().enumerate() {
                merge_rank.insert(m, rank);
            }

            // GPT-2 byte→unicode table
            // Bytes 33-126, 161-172, 174-255 map to same codepoint.
            // All other bytes (0-32, 127-160, 173) map to 256+n sequentially.
            let byte_to_unicode: [char; 256] = {
                let mut t = ['\0'; 256];
                let mut n = 0u32;
                for b in 0u32..256 {
                    if matches!(b, 33..=126 | 161..=172 | 174..=255) {
                        t[b as usize] = char::from_u32(b).unwrap();
                    } else {
                        t[b as usize] = char::from_u32(256 + n).unwrap();
                        n += 1;
                    }
                }
                t
            };

            // Pre-tokenize: split on whitespace boundaries (space→next word)
            let bytes = text.as_bytes();
            let mut pre_tokens: Vec<String> = Vec::new();
            {
                let mut current = String::new();
                for &b in bytes {
                    let uc = byte_to_unicode[b as usize];
                    if b == b' ' || b == b'\n' || b == b'\t' || b == b'\r' {
                        if !current.is_empty() {
                            pre_tokens.push(current.clone());
                            current.clear();
                        }
                        current.push(uc);
                    } else {
                        current.push(uc);
                    }
                }
                if !current.is_empty() {
                    pre_tokens.push(current);
                }
            }

            // BPE merge each pre-token and collect IDs
            let mut result_ids: Vec<f32> = Vec::new();

            for pre_tok in &pre_tokens {
                // Start with individual characters
                let mut pieces: Vec<String> = pre_tok.chars().map(|c| c.to_string()).collect();

                // Iteratively merge highest-priority pair
                loop {
                    if pieces.len() < 2 { break; }
                    let mut best_rank = usize::MAX;
                    let mut best_i = 0usize;
                    for i in 0..pieces.len() - 1 {
                        let pair = format!("{} {}", pieces[i], pieces[i + 1]);
                        if let Some(&rank) = merge_rank.get(&pair) {
                            if rank < best_rank {
                                best_rank = rank;
                                best_i = i;
                            }
                        }
                    }
                    if best_rank == usize::MAX { break; }
                    let merged = format!("{}{}", pieces[best_i], pieces[best_i + 1]);
                    pieces[best_i] = merged;
                    pieces.remove(best_i + 1);
                }

                // Look up token IDs
                for piece in &pieces {
                    if let Some(&id) = token_to_id.get(piece.as_str()) {
                        result_ids.push(id as f32);
                    } else {
                        // Byte-level fallback: each char as individual token
                        for ch in piece.chars() {
                            let ch_str = ch.to_string();
                            if let Some(&id) = token_to_id.get(ch_str.as_str()) {
                                result_ids.push(id as f32);
                            }
                        }
                    }
                }
            }

            Ok(Some(ArrayResult::GpuFloats(result_ids)))
        }
        // ── GPU VM register read ──────────────────────────────────────
        // vm_read_register(vm_id, instance, reg_idx, count) → array of floats
        // vm_read_metrics(vm_id, offset, count) → array of floats
        "vm_read_metrics" | "loom_read_metrics" if args.len() == 3 => {
            let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let count = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
            if device_ptr == 0 {
                return Err(CliError::Runtime("vm_read_metrics: no Vulkan GPU available. Loom VM requires a GPU".into()));
            }
            let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
            let data = GPU_VMS.with(|vms| {
                let vms = vms.borrow();
                let vm = vms.get(&vm_id).ok_or_else(||
                    CliError::Runtime(format!("vm_read_metrics: unknown VM {}", vm_id)))?;
                octoflow_vulkan::vm::vm_read_metrics(gpu_dev, vm, offset, count)
                    .map_err(|e| CliError::Runtime(format!("vm_read_metrics: {}", e)))
            })?;
            Ok(Some(ArrayResult::GpuFloats(data)))
        }
        // vm_read_globals(vm_id, offset, count) → array of floats
        "vm_read_globals" | "loom_read_globals" if args.len() == 3 => {
            let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let count = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
            if device_ptr == 0 {
                return Err(CliError::Runtime("vm_read_globals: no Vulkan GPU available. Loom VM requires a GPU".into()));
            }
            let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
            let data = GPU_VMS.with(|vms| {
                let vms = vms.borrow();
                let vm = vms.get(&vm_id).ok_or_else(||
                    CliError::Runtime(format!("vm_read_globals: unknown VM {}", vm_id)))?;
                octoflow_vulkan::vm::vm_read_globals(gpu_dev, vm, offset, count)
                    .map_err(|e| CliError::Runtime(format!("vm_read_globals: {}", e)))
            })?;
            Ok(Some(ArrayResult::GpuFloats(data)))
        }
        // vm_read_control(vm_id, offset, count) → array of floats
        "vm_read_control" | "loom_read_control" if args.len() == 3 => {
            let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let offset = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let count = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
            if device_ptr == 0 {
                return Err(CliError::Runtime("vm_read_control: no Vulkan GPU available. Loom VM requires a GPU".into()));
            }
            let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
            let data = GPU_VMS.with(|vms| {
                let vms = vms.borrow();
                let vm = vms.get(&vm_id).ok_or_else(||
                    CliError::Runtime(format!("vm_read_control: unknown VM {}", vm_id)))?;
                octoflow_vulkan::vm::vm_read_control(gpu_dev, vm, offset, count)
                    .map_err(|e| CliError::Runtime(format!("vm_read_control: {}", e)))
            })?;
            Ok(Some(ArrayResult::GpuFloats(data)))
        }
        // ── Loom read (vm_read_register alias) ─────────────────────────
        // loom_read / vm_read_register(vm_id, instance, reg_idx, count) → array of floats
        "loom_read" | "vm_read_register" if args.len() == 4 => {
            let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let instance = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let reg_idx = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let count = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
                .as_float()? as u32;
            let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
            if device_ptr == 0 {
                return Err(CliError::Runtime("vm_read_register: no Vulkan GPU available. Loom VM requires a GPU".into()));
            }
            let gpu_dev = unsafe { &*(device_ptr as *const octoflow_vulkan::VulkanCompute) };
            let data = GPU_VMS.with(|vms| {
                let vms = vms.borrow();
                let vm = vms.get(&vm_id).ok_or_else(||
                    CliError::Runtime(format!("vm_read_register: unknown VM {}", vm_id)))?;
                octoflow_vulkan::vm::vm_read_reg(gpu_dev, vm, instance, reg_idx, count)
                    .map_err(|e| CliError::Runtime(format!("vm_read_register: {}", e)))
            })?;
            Ok(Some(ArrayResult::GpuFloats(data)))
        }
        "gpu_matmul" => {
            Err(CliError::Compile(format!(
                "gpu_matmul() requires 5 arguments: gpu_matmul(a, b, m, n, k) \
                 where A is m×k, B is k×n, result is m×n. Got {} argument(s).",
                args.len()
            )))
        }
        // Catch wrong-arity calls to GPU array-returning functions that have
        // arity guards above. Scalar-returning GPU functions (gpu_sum, gpu_min,
        // gpu_save_csv, etc.) are handled elsewhere and must fall through.
        "gpu_compute" | "gpu_scale" | "gpu_clamp" | "gpu_pow" |
        "gpu_add" | "gpu_sub" | "gpu_mul" | "gpu_div" |
        "gpu_where" | "gpu_cumsum" | "gpu_fill" | "gpu_range" |
        "gpu_load_csv" | "gpu_load_binary" | "gpu_reverse" |
        "gpu_random" | "gpu_ema" | "gpu_concat" | "gpu_gather" | "gpu_scatter" |
        "gpu_run" | "vm_read_metrics" | "loom_read_metrics" |
        "vm_read_globals" | "loom_read_globals" |
        "vm_read_control" | "loom_read_control" |
        "loom_read" | "vm_read_register" => {
            Err(CliError::Compile(format!(
                "{}() called with {} argument(s) — wrong number of arguments",
                fn_name, args.len()
            )))
        }
        _ => Ok(None),
    }
}

/// Decode compressed OctoPress data. Used by octopress_decode builtin and streaming.
fn octopress_decode_internal(compressed: &[f32]) -> Result<Vec<f32>, CliError> {
    if compressed.len() < 2 {
        return Err(CliError::Runtime("octopress_decode: empty data".into()));
    }
    let method = compressed[0] as u32;
    let count = compressed[1] as u32;

    match method {
        0 => {
            // Raw: data starts at index 2
            Ok(compressed[2..].to_vec())
        }
        1 => {
            // Delta: first_value at index 2, deltas start at index 3
            if count == 0 {
                Ok(vec![])
            } else {
                let first = compressed[2];
                let mut out = vec![first];
                for i in 0..(count as usize - 1) {
                    let prev = *out.last().unwrap();
                    let delta = compressed.get(3 + i).copied().unwrap_or(0.0);
                    out.push(prev + delta);
                }
                Ok(out)
            }
        }
        2 => {
            // Fractal decode: IFS iteration
            if compressed.len() < 4 {
                return Err(CliError::Runtime("octopress_decode: fractal data too short".into()));
            }
            let original_count = compressed[1] as usize;
            let block_size = compressed[2] as usize;
            let n_ranges = compressed[3] as usize;
            let range_size = block_size / 2;

            // Read transforms
            let transforms_start = 4;
            let mut transforms = Vec::with_capacity(n_ranges);
            for r in 0..n_ranges {
                let base = transforms_start + r * 3;
                if base + 2 >= compressed.len() { break; }
                let domain_idx = compressed[base] as usize;
                let scale = compressed[base + 1];
                let offset = compressed[base + 2];
                transforms.push((domain_idx, scale, offset));
            }

            // IFS iteration: start with zeros, apply transforms 8 times
            let mut current = vec![0.0f32; original_count];
            for _iter in 0..8 {
                let mut next = vec![0.0f32; original_count];
                for (r, &(domain_idx, scale, offset)) in transforms.iter().enumerate() {
                    let r_start = r * range_size;
                    let d_start = domain_idx * block_size;
                    for i in 0..range_size {
                        if r_start + i >= original_count { break; }
                        let d_i = d_start + i * 2;
                        let domain_val = if d_i + 1 < current.len() {
                            (current[d_i] + current[d_i + 1]) / 2.0
                        } else if d_i < current.len() {
                            current[d_i]
                        } else {
                            0.0
                        };
                        next[r_start + i] = (scale * domain_val + offset).clamp(-1e12, 1e12);
                    }
                }
                current = next;
            }

            current.truncate(original_count);
            Ok(current)
        }
        _ => Err(CliError::Runtime(format!("octopress_decode: unknown method {}", method))),
    }
}

// json_value_to_value, flatten_json, unflatten_to_json, value_to_json
// moved to crate::json_io (Phase 45 — serde_json removed)

/// Evaluate hashmap-returning functions (json_parse, load_data).
/// Returns Some(HashMap) if this is a hashmap-returning function, None otherwise.
#[allow(clippy::too_many_arguments)]
fn eval_hashmap_fn(
    fn_name: &str,
    args: &[ScalarExpr],
    streams: &HashMap<String, Vec<f32>>,
    scalars: &HashMap<String, Value>,
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    arrays: &mut HashMap<String, Vec<Value>>,
    hashmaps: &mut HashMap<String, HashMap<String, Value>>,
    scalar_fns: &HashMap<String, ScalarFnDef>,
    struct_defs: &HashMap<String, Vec<String>>,
    rng: &Cell<u64>,
    mutable_scalars: &std::collections::HashSet<String>,
) -> Result<Option<HashMap<String, Value>>, CliError> {
    match fn_name {
        "json_parse" if args.len() == 1 => {
            let json_str = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let s = json_str.as_str().map_err(|_| CliError::Compile("json_parse() argument must be a string".into()))?.to_string();
            let result = crate::json_io::parse_object(&s)?;
            Ok(Some(result))
        }
        "load_data" if args.len() == 1 => {
            let path_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let path = path_val.as_str().map_err(|_| CliError::Compile("load_data() path must be a string".into()))?.to_string();
            check_read_permission_for(&path)?;
            let content = std::fs::read_to_string(&path)
                .map_err(|e| CliError::Compile(format!("load_data(): cannot read '{}': {}", path, e)))?;
            let program = octoflow_parser::parse(&content)
                .map_err(|e| CliError::Compile(format!("load_data(): parse error in '{}': {}", path, e)))?;
            let mut result = HashMap::new();
            // Sandboxed mini-evaluator: only LetDecl and ArrayDecl allowed
            let empty_streams: HashMap<String, Vec<f32>> = HashMap::new();
            let mut local_scalars: HashMap<String, Value> = HashMap::new();
            let empty_gpu: Option<octoflow_vulkan::VulkanCompute> = None;
            let empty_scalar_fns: HashMap<String, ScalarFnDef> = HashMap::new();
            let empty_struct_defs: HashMap<String, Vec<String>> = HashMap::new();
            let mut empty_hashmaps: HashMap<String, HashMap<String, Value>> = HashMap::new();
            let local_rng: Cell<u64> = Cell::new(12345);
            let empty_mutable: std::collections::HashSet<String> = std::collections::HashSet::new();
            for (stmt, span) in &program.statements {
                match stmt {
                    Statement::LetDecl { name, value, .. } => {
                        let val = eval_scalar(value, &empty_streams, &local_scalars, &empty_gpu, arrays, &mut empty_hashmaps, &empty_scalar_fns, &empty_struct_defs, &local_rng, &empty_mutable)?;
                        local_scalars.insert(name.clone(), val.clone());
                        result.insert(name.clone(), val);
                    }
                    Statement::ArrayDecl { name, elements, .. } => {
                        let mut arr = Vec::new();
                        for el in elements {
                            let val = eval_scalar(el, &empty_streams, &local_scalars, &empty_gpu, arrays, &mut empty_hashmaps, &empty_scalar_fns, &empty_struct_defs, &local_rng, &empty_mutable)?;
                            arr.push(val);
                        }
                        arrays.insert(name.clone(), arr);
                        result.insert(name.clone(), Value::Str(format!("[array:{}]", name)));
                    }
                    _ => {
                        return Err(CliError::Compile(format!(
                            "load_data(): only 'let' declarations allowed in .od files, found {:?} at line {}",
                            std::mem::discriminant(stmt), span.line
                        )));
                    }
                }
            }
            Ok(Some(result))
        }
        // ── GPU device info (Phase 75a) ─────────────────────────────────
        "gpu_info" if args.is_empty() => {
            let mut info = HashMap::new();
            if let Some(ref gpu_dev) = gpu {
                let props = gpu_dev.gpu_properties();
                info.insert("name".to_string(), Value::Str(props.name));
                info.insert("type".to_string(), Value::Str(props.device_type));
                info.insert("api_version_major".to_string(), Value::Float(props.api_version_major as f32));
                info.insert("api_version_minor".to_string(), Value::Float(props.api_version_minor as f32));
                info.insert("vendor_id".to_string(), Value::Float(props.vendor_id as f32));
                info.insert("device_id".to_string(), Value::Float(props.device_id as f32));
                info.insert("max_workgroup_size_x".to_string(), Value::Float(props.max_compute_workgroup_size_x as f32));
                info.insert("max_workgroup_size_y".to_string(), Value::Float(props.max_compute_workgroup_size_y as f32));
                info.insert("max_workgroup_size_z".to_string(), Value::Float(props.max_compute_workgroup_size_z as f32));
                info.insert("max_workgroup_invocations".to_string(), Value::Float(props.max_compute_workgroup_invocations as f32));
                info.insert("max_storage_buffer".to_string(), Value::Float(props.max_storage_buffer_range as f32));
                info.insert("max_shared_memory".to_string(), Value::Float(props.max_compute_shared_memory as f32));
                info.insert("supports_f16".to_string(), Value::Float(if props.supports_f16 { 1.0 } else { 0.0 }));
                info.insert("supports_int64".to_string(), Value::Float(if props.supports_int64 { 1.0 } else { 0.0 }));
                info.insert("available".to_string(), Value::Float(1.0));
            } else {
                info.insert("name".to_string(), Value::Str("none".to_string()));
                info.insert("type".to_string(), Value::Str("cpu".to_string()));
                info.insert("available".to_string(), Value::Float(0.0));
            }
            Ok(Some(info))
        }
        // ── Web builtins (v1.2) ─────────────────────────────────────────
        "web_read" if args.len() == 1 => {
            check_net_permission()?;
            let url_val = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?;
            let url = url_val.as_str().map_err(|_| CliError::Compile("web_read() argument must be a string".into()))?.to_string();
            let result = crate::io::web::web_read(&url)?;
            match result {
                Value::Map(m) => Ok(Some(m)),
                _ => Err(CliError::Runtime("web_read(): unexpected result type".into())),
            }
        }
        _ => Ok(None),
    }
}

/// Serialize a hashmap to OctoData (.od) format string.
fn serialize_od(map: &HashMap<String, Value>) -> String {
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    let mut lines = Vec::new();
    for key in keys {
        let val = &map[key];
        match val {
            Value::Float(f) => {
                if *f == f.floor() && f.is_finite() && f.abs() < 1e10 {
                    lines.push(format!("let {} = {:.1}", key, f));
                } else {
                    lines.push(format!("let {} = {}", key, f));
                }
            }
            Value::Int(i) => {
                lines.push(format!("let {} = {}", key, i));
            }
            Value::Str(s) => {
                lines.push(format!("let {} = \"{}\"", key, s));
            }
            Value::Map(_) => {
                lines.push(format!("let {} = {}", key, val));
            }
            Value::None => {
                lines.push(format!("let {} = none", key));
            }
        }
    }
    lines.join("\n") + "\n"
}

/// All known builtin function names for "did you mean?" suggestions.
const KNOWN_BUILTINS: &[&str] = &[
    // Math
    "abs", "sqrt", "exp", "log", "ln", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "floor", "ceil", "round", "pow", "clamp", "min", "max", "sign", "fract", "lerp",
    "dot", "norm", "int", "float", "to_str", "to_float", "type_of", "is_float", "is_str",
    "is_map", "is_array", "is_none", "is_nan", "is_inf",
    // String
    "len", "split", "join", "trim", "replace", "starts_with", "ends_with", "contains",
    "to_upper", "to_lower", "char_at", "char_code", "from_char_code", "substr", "index_of",
    "repeat_str", "pad_left", "pad_right", "tokenize",
    // Array
    "push", "pop", "reverse", "sort_array", "sort_by", "map_each", "filter",
    "reduce", "range", "slice", "flatten", "unique", "zip", "enumerate",
    "extend", "array_copy", "array_extract", "array_new",
    // Map
    "map", "map_get", "map_set", "map_has", "map_keys", "map_values", "map_remove",
    // I/O
    "read_file", "read", "write_file", "append_file", "read_lines", "read_bytes", "write_bytes",
    "read_csv", "write_csv", "list_dir", "walk_dir", "file_exists", "file_mtime", "read_image", "write_image",
    // HTTP
    "http_get", "http_post", "http_put", "http_delete", "http_listen", "http_respond",
    "http_accept", "http_respond_json", "http_respond_html", "http_respond_png", "http_respond_with_headers",
    // JSON
    "json_parse", "json_stringify", "json_parse_array",
    // Net
    "tcp_connect", "tcp_send", "tcp_recv", "tcp_listen", "tcp_accept", "socket_close",
    "udp_socket", "udp_send_to", "udp_recv_from",
    // Regex
    "regex_match", "is_match", "regex_find", "regex_find_all", "regex_replace",
    "capture_groups",
    // System
    "time", "now_ms", "time_ms", "now_us", "sleep", "env", "os_name", "exec", "print_raw", "print_bytes",
    "clock", "exit",
    // GPU
    "gpu_fill", "gpu_add", "gpu_sub", "gpu_mul", "gpu_div", "gpu_sum", "gpu_min",
    "gpu_max", "gpu_mean", "gpu_count", "gpu_matmul", "gpu_dot", "sort", "gpu_sort",
    "gpu_topk", "gpu_topk_indices", "cosine_similarity",
    // GPU element-wise unary (Phase 79 kernels)
    "gpu_abs", "gpu_sqrt", "gpu_exp", "gpu_log", "gpu_negate",
    "gpu_floor", "gpu_ceil", "gpu_round", "gpu_sin", "gpu_cos",
    "gpu_gather", "gpu_scatter", "gpu_info",
    // Performance profiling (R-23)
    "gpu_timer_start", "gpu_timer_end",
    // Audio (R-25, R-26)
    "audio_play", "audio_play_file", "audio_stop",
    // Loom
    "loom_boot", "loom_dispatch", "loom_dispatch_jit", "loom_read", "loom_write", "loom_shutdown",
    "loom_set_globals", "loom_present", "loom_wait", "loom_set_heap",
    "loom_build", "loom_run", "loom_launch", "loom_poll", "loom_free",
    "loom_read_globals", "loom_read_metrics", "loom_read_control",
    "loom_write_metrics", "loom_write_control",
    "loom_status", "loom_pace", "loom_prefetch", "loom_copy",
    "loom_mailbox", "loom_mail_send", "loom_mail_recv", "loom_mail_poll", "loom_mail_depth",
    "loom_park", "loom_unpark", "loom_pool_size", "loom_pool_warm", "loom_auto_spawn", "loom_auto_release",
    "loom_max_vms", "loom_vram_budget", "loom_vram_used", "loom_vm_count",
    "loom_threads", "loom_cpu_count", "loom_async_read", "loom_await",
    "loom_vm_info", "loom_elapsed_us", "loom_dispatch_time", "loom_pool_info",
    "octopress_init", "octopress_analyze", "octopress_encode", "octopress_decode",
    "octopress_save", "octopress_load", "octopress_gpu_encode", "octopress_info",
    "octopress_stream_open", "octopress_stream_next", "octopress_stream_info",
    "octopress_stream_reset", "octopress_stream_close",
    "vm_boot", "vm_dispatch", "vm_read_globals", "vm_write_globals", "vm_set_heap",
    "vm_write_metrics", "vm_read_metrics", "vm_write_control", "vm_read_control",
    "vm_shutdown", "vm_present", "vm_wait",
    // LLM
    "gguf_load_tensor", "gguf_matvec", "gguf_infer_layer", "gguf_evict_layer",
    "gguf_cache_file", "gguf_load_vocab", "chat_emit_token", "gguf_tokens_per_sec",
    // Grammar-constrained decoding
    "grammar_load", "grammar_load_str", "grammar_mask", "grammar_advance",
    "grammar_reset", "grammar_active",
    // Window
    "window_open", "window_close", "window_alive", "window_draw", "window_poll",
    "window_event_key", "window_event_x", "window_event_y", "window_width", "window_height",
    "window_title", "window_event_char", "window_event_scroll",
    "window_capture_mouse", "window_release_mouse",
    "clipboard_get", "clipboard_set",
    "dialog_open_file", "dialog_save_file", "dialog_message",
    "window_set_cursor", "window_set_timer", "window_kill_timer", "window_event_timer_id",
    "window_create_menu", "menu_add_item", "menu_add_submenu", "window_set_menu",
    "window_dpi",
    // Game engine input (R-06, R-08)
    "gui_mouse_down", "gui_mouse_buttons", "gui_scroll_y", "window_key_held",
    // Terminal
    "term_clear", "term_move_up", "term_image", "term_supports_graphics",
    // Assert (new in v1.2)
    "assert", "panic", "format", "clone",
    // Web (new in v1.2)
    "web_search", "web_read",
];

/// Emit SPIR-V for the OctoPress GPU fractal domain search kernel.
/// Each thread handles one (range_idx, domain_idx) pair: computes the SSD
/// between a range block and a downsampled domain block.
/// Layout: globals[0..N] = input data, output at globals[out_offset..].
/// Push constants: [n_domains, range_size, block_size, out_offset, total_threads] (as uint via f32::from_bits).
fn emit_octopress_fractal_spirv(wg_size: u32) -> Vec<u8> {
    let mut s = Vec::<u32>::with_capacity(512);
    let mut n: u32 = 1;
    macro_rules! id { () => {{ let i = n; n += 1; i }} }
    fn em(s: &mut Vec<u32>, op: u32, a: &[u32]) {
        s.push(((a.len() as u32 + 1) << 16) | op);
        s.extend_from_slice(a);
    }

    // --- Pre-allocate IDs ---
    let t_void = id!(); let t_bool = id!(); let t_uint = id!(); let t_float = id!();
    let t_v3u = id!(); let t_rta = id!(); let t_sg = id!(); let t_pc = id!();
    let t_fn = id!();
    let tp_iv3 = id!(); let tp_iu = id!(); // ptr Input v3uint, ptr Input uint
    let tp_ss = id!(); let tp_sf = id!(); // ptr StorageBuffer struct, ptr StorageBuffer float
    let tp_pc = id!(); let tp_pu = id!(); // ptr PushConstant struct, ptr PushConstant uint
    // Constants
    let c0 = id!(); let c1 = id!(); let c2 = id!(); let c3 = id!(); let c4 = id!();
    let f0 = id!(); let f2 = id!();
    // Variables
    let vg = id!(); let vb = id!(); let vpc = id!();
    // Function + labels
    let fm = id!(); let le = id!(); let lm = id!(); let lw = id!();
    let lh = id!(); let lb = id!(); let lc = id!(); let lx = id!();

    // --- Header ---
    s.extend_from_slice(&[0x07230203, 0x00010000, 0, 0/*bound*/, 0]);

    // OpCapability Shader
    em(&mut s, 17, &[1]);
    // OpMemoryModel Logical GLSL450
    em(&mut s, 14, &[0, 1]);
    // OpEntryPoint GLCompute %fm "main" %vg
    {
        let nm = [0x6E69616Du32, 0u32]; // "main\0"
        let wc = (3 + nm.len() + 1) as u32;
        s.push((wc << 16) | 15);
        s.push(5); s.push(fm);
        s.extend_from_slice(&nm);
        s.push(vg);
    }
    // OpExecutionMode %fm LocalSize wg 1 1
    em(&mut s, 16, &[fm, 17, wg_size, 1, 1]);

    // --- Decorations ---
    em(&mut s, 71, &[vg, 11, 28]);       // BuiltIn GlobalInvocationId
    em(&mut s, 71, &[t_sg, 2]);          // Block
    em(&mut s, 72, &[t_sg, 0, 35, 0]);   // MemberDecorate Offset 0
    em(&mut s, 71, &[t_rta, 6, 4]);      // ArrayStride 4
    em(&mut s, 71, &[vb, 34, 0]);        // DescriptorSet 0
    em(&mut s, 71, &[vb, 33, 2]);        // Binding 2
    em(&mut s, 71, &[t_pc, 2]);          // Block
    for i in 0u32..5 { em(&mut s, 72, &[t_pc, i, 35, i * 4]); } // Offsets

    // --- Types ---
    em(&mut s, 19, &[t_void]);
    em(&mut s, 20, &[t_bool]);
    em(&mut s, 21, &[t_uint, 32, 0]);
    em(&mut s, 22, &[t_float, 32]);
    em(&mut s, 23, &[t_v3u, t_uint, 3]);
    em(&mut s, 29, &[t_rta, t_float]);
    em(&mut s, 30, &[t_sg, t_rta]);
    em(&mut s, 30, &[t_pc, t_uint, t_uint, t_uint, t_uint, t_uint]);
    em(&mut s, 33, &[t_fn, t_void]);
    em(&mut s, 32, &[tp_iv3, 1, t_v3u]);
    em(&mut s, 32, &[tp_iu, 1, t_uint]);
    em(&mut s, 32, &[tp_ss, 12, t_sg]);
    em(&mut s, 32, &[tp_sf, 12, t_float]);
    em(&mut s, 32, &[tp_pc, 9, t_pc]);
    em(&mut s, 32, &[tp_pu, 9, t_uint]);

    // --- Constants ---
    em(&mut s, 43, &[t_uint, c0, 0]);
    em(&mut s, 43, &[t_uint, c1, 1]);
    em(&mut s, 43, &[t_uint, c2, 2]);
    em(&mut s, 43, &[t_uint, c3, 3]);
    em(&mut s, 43, &[t_uint, c4, 4]);
    em(&mut s, 43, &[t_float, f0, 0f32.to_bits()]);
    em(&mut s, 43, &[t_float, f2, 2f32.to_bits()]);

    // --- Variables ---
    em(&mut s, 59, &[tp_iv3, vg, 1]);   // Input
    em(&mut s, 59, &[tp_ss, vb, 12]);   // StorageBuffer
    em(&mut s, 59, &[tp_pc, vpc, 9]);   // PushConstant

    // --- Function ---
    em(&mut s, 54, &[t_void, fm, 0, t_fn]); // OpFunction
    em(&mut s, 248, &[le]);                   // OpLabel entry

    // Load tid = GlobalInvocationId.x
    let p0 = id!(); em(&mut s, 65, &[tp_iu, p0, vg, c0]);  // AccessChain
    let tid = id!(); em(&mut s, 61, &[t_uint, tid, p0]);     // Load

    // Load push constants: nd, rs, bs, oo, tt
    let mut pc_vals = [0u32; 5];
    for i in 0..5u32 {
        let ci = [c0, c1, c2, c3, c4][i as usize];
        let pp = id!(); em(&mut s, 65, &[tp_pu, pp, vpc, ci]);
        let vv = id!(); em(&mut s, 61, &[t_uint, vv, pp]);
        pc_vals[i as usize] = vv;
    }
    let [nd, rs, bs, oo, tt] = pc_vals;

    // Bounds check: if tid >= total_threads → skip
    let ib = id!(); em(&mut s, 176, &[t_bool, ib, tid, tt]); // ULessThan
    em(&mut s, 247, &[lm, 0]); // SelectionMerge %lm None
    em(&mut s, 250, &[ib, lw, lm]); // BranchConditional

    // --- Work block ---
    em(&mut s, 248, &[lw]);
    // range_idx = tid / n_domains
    let ri = id!(); em(&mut s, 134, &[t_uint, ri, tid, nd]);
    // domain_idx = tid - range_idx * n_domains
    let rm = id!(); em(&mut s, 132, &[t_uint, rm, ri, nd]);
    let di = id!(); em(&mut s, 130, &[t_uint, di, tid, rm]); // ISub
    // r_start = range_idx * range_size
    let rst = id!(); em(&mut s, 132, &[t_uint, rst, ri, rs]);
    // d_start = domain_idx * block_size
    let dst = id!(); em(&mut s, 132, &[t_uint, dst, di, bs]);
    em(&mut s, 249, &[lh]); // Branch to loop header

    // --- Loop header ---
    em(&mut s, 248, &[lh]);
    let pi = id!(); // phi for i
    let ps = id!(); // phi for ssd
    // OpPhi: result_type result (val0, parent0) (val1, parent1) ...
    // phi_i: from lw→c0, from lc→i_next
    let i_next = id!(); // forward-declare
    let ns = id!();     // forward-declare new_ssd
    {
        let wc = 7u32; // 1 + 2 (result) + 2*2 (pairs)
        s.push((wc << 16) | 245); s.push(t_uint); s.push(pi);
        s.push(c0); s.push(lw); s.push(i_next); s.push(lc);
    }
    {
        let wc = 7u32;
        s.push((wc << 16) | 245); s.push(t_float); s.push(ps);
        s.push(f0); s.push(lw); s.push(ns); s.push(lc);
    }
    let lp = id!(); em(&mut s, 176, &[t_bool, lp, pi, rs]); // i < range_size
    em(&mut s, 246, &[lx, lc, 0]); // LoopMerge %lx %lc None
    em(&mut s, 250, &[lp, lb, lx]); // BranchConditional

    // --- Loop body ---
    em(&mut s, 248, &[lb]);
    // r_idx = r_start + i
    let rx = id!(); em(&mut s, 128, &[t_uint, rx, rst, pi]);
    // r_ptr → load r_val
    let rp = id!(); em(&mut s, 65, &[tp_sf, rp, vb, c0, rx]);
    let rv = id!(); em(&mut s, 61, &[t_float, rv, rp]);
    // d_base = d_start + 2*i
    let i2 = id!(); em(&mut s, 132, &[t_uint, i2, pi, c2]);
    let db = id!(); em(&mut s, 128, &[t_uint, db, dst, i2]);
    // load d_val0
    let dp0 = id!(); em(&mut s, 65, &[tp_sf, dp0, vb, c0, db]);
    let dv0 = id!(); em(&mut s, 61, &[t_float, dv0, dp0]);
    // d_base1 = d_base + 1, load d_val1
    let db1 = id!(); em(&mut s, 128, &[t_uint, db1, db, c1]);
    let dp1 = id!(); em(&mut s, 65, &[tp_sf, dp1, vb, c0, db1]);
    let dv1 = id!(); em(&mut s, 61, &[t_float, dv1, dp1]);
    // d_avg = (dv0 + dv1) / 2.0
    let ds = id!(); em(&mut s, 129, &[t_float, ds, dv0, dv1]); // FAdd
    let da = id!(); em(&mut s, 136, &[t_float, da, ds, f2]);   // FDiv
    // diff = r_val - d_avg
    let df = id!(); em(&mut s, 131, &[t_float, df, rv, da]);   // FSub
    // diff2 = diff * diff
    let d2 = id!(); em(&mut s, 133, &[t_float, d2, df, df]);   // FMul
    // new_ssd = ssd + diff2
    // (ns was forward-declared)
    em(&mut s, 129, &[t_float, ns, ps, d2]); // FAdd
    em(&mut s, 249, &[lc]); // Branch to continue

    // --- Loop continue ---
    em(&mut s, 248, &[lc]);
    // i_next = i + 1 (i_next was forward-declared)
    em(&mut s, 128, &[t_uint, i_next, pi, c1]);
    em(&mut s, 249, &[lh]); // Branch to header

    // --- Loop exit: store result ---
    em(&mut s, 248, &[lx]);
    // out_idx = out_offset + tid
    let ox = id!(); em(&mut s, 128, &[t_uint, ox, oo, tid]);
    let op_ = id!(); em(&mut s, 65, &[tp_sf, op_, vb, c0, ox]);
    em(&mut s, 62, &[op_, ps]); // Store ssd to globals[out_idx]
    em(&mut s, 249, &[lm]); // Branch to merge

    // --- Merge / return ---
    em(&mut s, 248, &[lm]);
    em(&mut s, 253, &[]); // OpReturn
    em(&mut s, 56, &[]);  // OpFunctionEnd

    // Fix bound
    s[3] = n;

    // Convert to bytes (little-endian)
    s.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn eval_scalar_fn(name: &str, args: &[f32]) -> Result<f32, CliError> {
    match (name, args.len()) {
        ("abs", 1) => Ok(args[0].abs()),
        ("sqrt", 1) => {
            if args[0] < 0.0 {
                return Err(CliError::Compile(format!("sqrt({}) — cannot take square root of a negative number", args[0])));
            }
            Ok(args[0].sqrt())
        }
        ("exp", 1) => Ok(args[0].exp()),
        ("log", 1) | ("ln", 1) => {
            if args[0] <= 0.0 {
                return Err(CliError::Compile(format!("log({}) — argument must be positive", args[0])));
            }
            Ok(args[0].ln())
        }
        ("sin", 1) => Ok(args[0].sin()),
        ("cos", 1) => Ok(args[0].cos()),
        ("floor", 1) => Ok(args[0].floor()),
        ("ceil", 1) => Ok(args[0].ceil()),
        ("round", 1) => Ok(args[0].round()),
        ("pow", 2) => Ok(args[0].powf(args[1])),
        ("clamp", 3) => Ok(args[0].clamp(args[1], args[2])),
        // 3D math / game engine (R-13, R-16)
        ("tan", 1) => Ok(args[0].tan()),
        ("asin", 1) => {
            if args[0] < -1.0 || args[0] > 1.0 {
                return Err(CliError::Compile(format!("asin({}) — argument must be in [-1, 1]", args[0])));
            }
            Ok(args[0].asin())
        }
        ("acos", 1) => {
            if args[0] < -1.0 || args[0] > 1.0 {
                return Err(CliError::Compile(format!("acos({}) — argument must be in [-1, 1]", args[0])));
            }
            Ok(args[0].acos())
        }
        ("atan", 1) => Ok(args[0].atan()),
        ("atan2", 2) => Ok(args[0].atan2(args[1])),
        ("min", 2) => Ok(args[0].min(args[1])),
        ("max", 2) => Ok(args[0].max(args[1])),
        ("sign", 1) => Ok(if args[0] > 0.0 { 1.0 } else if args[0] < 0.0 { -1.0 } else { 0.0 }),
        ("fract", 1) => Ok(args[0].fract()),
        ("lerp", 3) => Ok(args[0] + (args[1] - args[0]) * args[2]),
        _ => {
            let mut msg = format!(
                "unknown function '{}' with {} argument{}",
                name, args.len(), if args.len() == 1 { "" } else { "s" }
            );
            if let Some(suggestion) = crate::suggest_closest(name, KNOWN_BUILTINS) {
                msg.push_str(&format!(". Did you mean '{}'?", suggestion));
            }
            Err(CliError::Compile(msg))
        }
    }
}

/// Execute a user-defined scalar function with the given argument values.
/// `scalar_args`: (param_index, value) for scalar parameters.
/// `array_bindings`: (param_name, original_array_name) for array parameters.
/// `caller_arrays`: the caller's arrays map — arrays are cloned in, mutated copies written back.
/// `caller_mutable`: the caller's mutable set — determines if array params are mutable.
fn execute_user_fn(
    fn_def: &ScalarFnDef,
    scalar_args: &[(usize, Value)],
    array_bindings: &[(String, String)],
    hashmap_bindings: &[(String, String)],
    caller_scalars: &HashMap<String, Value>,
    caller_arrays: &mut HashMap<String, Vec<Value>>,
    caller_hashmaps: &mut HashMap<String, HashMap<String, Value>>,
    caller_mutable: &std::collections::HashSet<String>,
    streams: &HashMap<String, Vec<f32>>,
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    scalar_fns: &HashMap<String, ScalarFnDef>,
    struct_defs: &HashMap<String, Vec<String>>,
    rng: &Cell<u64>,
) -> Result<Value, CliError> {
    // Recursion depth guard — prevent stack overflow from runaway recursion
    let depth = RECURSION_DEPTH.with(|c| {
        let d = c.get() + 1;
        c.set(d);
        d
    });
    if depth > MAX_RECURSION_DEPTH {
        RECURSION_DEPTH.with(|c| c.set(c.get() - 1));
        return Err(CliError::Compile(format!(
            "maximum recursion depth ({}) exceeded", MAX_RECURSION_DEPTH
        )));
    }

    // Create local scope — functions inherit caller scope (snapshot semantics for scalars,
    // shared mutation for arrays/hashmaps). Module constants are visible.
    let mut local_scalars: HashMap<String, Value> = caller_scalars.clone();
    let mut local_mutable: std::collections::HashSet<String> = caller_mutable.clone();
    let mut local_arrays: HashMap<String, Vec<Value>> = caller_arrays.clone();
    let mut local_hashmaps: HashMap<String, HashMap<String, Value>> = caller_hashmaps.clone();

    // R-04: Merge captured module constants (lower priority than caller scope)
    for (k, v) in &fn_def.captured_scalars {
        local_scalars.entry(k.clone()).or_insert_with(|| v.clone());
    }
    for (k, v) in &fn_def.captured_arrays {
        local_arrays.entry(k.clone()).or_insert_with(|| v.clone());
    }

    // Bind scalar parameters
    for (i, val) in scalar_args {
        local_scalars.insert(fn_def.params[*i].clone(), val.clone());
    }

    // Bind array parameters — remap caller array name to param name if different
    for (param_name, original_name) in array_bindings {
        if param_name != original_name {
            gpu_array_materialize(original_name, &mut local_arrays);
            if let Some(arr) = local_arrays.get(original_name) {
                let arr_clone = arr.clone();
                local_arrays.insert(param_name.clone(), arr_clone);
                if caller_mutable.contains(original_name) {
                    local_mutable.insert(param_name.clone());
                }
            }
        }
    }

    // Bind hashmap parameters — remap caller hashmap name to param name if different
    for (param_name, original_name) in hashmap_bindings {
        if param_name != original_name {
            if let Some(hm) = local_hashmaps.get(original_name) {
                let hm_clone = hm.clone();
                local_hashmaps.insert(param_name.clone(), hm_clone);
                if caller_mutable.contains(original_name) {
                    local_mutable.insert(param_name.clone());
                }
            }
        }
    }

    // Execute body using execute_loop_body (handles all statement types + return)
    let ctrl = match execute_loop_body(
        &fn_def.body, streams, &mut local_scalars, &mut local_mutable,
        struct_defs, gpu, &mut local_arrays, &mut local_hashmaps, scalar_fns, rng,
    ) {
        Ok(c) => c,
        Err(e) => {
            RECURSION_DEPTH.with(|c| c.set(c.get() - 1));
            return Err(e);
        }
    };

    // Copy back parameter-renamed arrays first (param_name → original_name)
    for (param_name, original_name) in array_bindings {
        if param_name != original_name {
            if let Some(arr) = local_arrays.remove(param_name) {
                caller_arrays.insert(original_name.clone(), arr);
            }
            local_arrays.remove(original_name);
        }
    }
    // Copy back remaining mutated arrays by name
    for (name, arr) in local_arrays {
        if caller_mutable.contains(&name) || caller_arrays.contains_key(&name) {
            caller_arrays.insert(name, arr);
        }
    }
    // Copy back parameter-renamed hashmaps
    for (param_name, original_name) in hashmap_bindings {
        if param_name != original_name {
            if let Some(hm) = local_hashmaps.remove(param_name) {
                caller_hashmaps.insert(original_name.clone(), hm);
            }
            local_hashmaps.remove(original_name);
        }
    }
    // Copy back remaining mutated hashmaps by name
    for (name, hm) in local_hashmaps {
        if caller_mutable.contains(&name) || caller_hashmaps.contains_key(&name) {
            caller_hashmaps.insert(name, hm);
        }
    }

    // R-05: Propagate mutated scalars via thread-local side-channel
    // Collect scalars that were declared mutable by caller and modified in function
    // (excluding function parameters, which are local bindings)
    let mut writeback: Vec<(String, Value)> = Vec::new();
    for (name, value) in &local_scalars {
        if caller_mutable.contains(name.as_str()) && !fn_def.params.contains(name) {
            // Only write back if value differs from caller's original
            if let Some(orig) = caller_scalars.get(name) {
                if orig != value {
                    writeback.push((name.clone(), value.clone()));
                }
            }
        }
    }
    if !writeback.is_empty() {
        SCALAR_WRITEBACK.with(|sw| {
            *sw.borrow_mut() = Some(writeback);
        });
    }

    // Decrement recursion depth on exit
    RECURSION_DEPTH.with(|c| c.set(c.get() - 1));

    match ctrl {
        LoopControl::Return(v) => Ok(v),
        LoopControl::Normal => Ok(Value::Float(0.0)),  // Implicit return 0.0
        LoopControl::Break => Err(CliError::Compile(
            "'break' can only be used inside a loop".into()
        )),
        LoopControl::Continue => Err(CliError::Compile(
            "'continue' can only be used inside a loop".into()
        )),
    }
}

/// Dispatch a reduce operation on GPU, or fall back to CPU.
fn dispatch_reduce_op(
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    op_name: &str,
    data: &[f32],
) -> Result<f32, CliError> {
    // count() is CPU-only — just return element count
    if op_name == "count" {
        return Ok(data.len() as f32);
    }
    let reduce_op = match op_name {
        "min" => ReduceOp::Min,
        "max" => ReduceOp::Max,
        "sum" => ReduceOp::Sum,
        "mul" => ReduceOp::Mul,
        _ => return Err(CliError::UnknownOperation(op_name.into())),
    };
    if let Some(gpu) = gpu {
        octoflow_vulkan::dispatch_reduce(gpu, reduce_op, data)
            .map_err(|e| CliError::Gpu(format!("{}", e)))
    } else {
        Ok(cpu_reduce(reduce_op, data))
    }
}

/// Evaluate an expression, returning its data.
fn eval_expr(
    expr: &Expr,
    streams: &HashMap<String, Vec<f32>>,
    scalars: &HashMap<String, Value>,
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    base_dir: &str,
    functions: &HashMap<String, FnDef>,
    image_dims: &mut HashMap<String, (u32, u32)>,
) -> Result<Vec<f32>, CliError> {
    match expr {
        Expr::Tap { path } => {
            let full_path = resolve_path(base_dir, path)?;
            if crate::image_io::is_image_path(&full_path) {
                let (data, w, h) = crate::image_io::read_image(&full_path)?;
                image_dims.insert(path.clone(), (w, h));
                Ok(data)
            } else if crate::octo_io::is_octo_path(&full_path) {
                crate::octo_io::read_octo(&full_path)
            } else {
                csv_read_floats(&full_path)
            }
        }
        Expr::RandomStream { count, lo, hi } => {
            // Evaluate count (may reference a scalar like N)
            let n_val = eval_scalar(count, streams, scalars, gpu,
                &mut std::collections::HashMap::new(),
                &mut std::collections::HashMap::new(),
                &std::collections::HashMap::new(),
                &std::collections::HashMap::new(),
                &std::cell::Cell::new(0u64),
                &std::collections::HashSet::new())?;
            let n = n_val.as_float()
                .map_err(|_| CliError::Compile("random_stream: count must be a number".into()))? as usize;

            // xorshift64* CPU RNG — fast, uniform, no GPU round-trip needed.
            // The GPU handles computation (map/reduce), CPU handles data generation.
            let seed = RANDOM_SEED.with(|s| {
                let v = s.get();
                s.set(v.wrapping_mul(1664525).wrapping_add(1013904223));
                v
            });
            let range = (*hi - *lo) as f64;
            let lo_f  = *lo as f64;
            let mut state: u64 = seed as u64 | 1;
            let data: Vec<f32> = (0..n).map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let bits = state.wrapping_mul(2685821657736338717u64);
                let f = (bits >> 11) as f64 / (1u64 << 53) as f64;
                (lo_f + f * range) as f32
            }).collect();
            Ok(data)
        }
        Expr::Cache { key, inner } => {
            // Check session cache first — zero GPU dispatch on hit.
            let cached = STREAM_CACHE.with(|c| c.borrow().get(key).cloned());
            if let Some(data) = cached {
                return Ok(data);
            }
            // Cache miss: evaluate inner expression, store result.
            let data = eval_expr(inner, streams, scalars, gpu, base_dir, functions, image_dims)?;
            STREAM_CACHE.with(|c| c.borrow_mut().insert(key.clone(), data.clone()));
            Ok(data)
        }
        Expr::Ref { name } => streams
            .get(name)
            .cloned()
            .ok_or_else(|| CliError::UndefinedStream(name.clone())),
        Expr::Pipe { input, stages } => {
            let mut data = eval_expr(input, streams, scalars, gpu, base_dir, functions, image_dims)?;
            let mut i = 0;
            while i < stages.len() {
                // N-stage map fusion: greedily collect consecutive MapOps into a single kernel.
                // Each MapOp is a pure element-wise transform — no dependencies between elements,
                // so any chain can be composed into one SPIR-V kernel with no intermediate buffers.
                {
                    let mut map_ops: Vec<MapOp> = Vec::new();
                    let mut j = i;
                    while j < stages.len() && functions.get(&stages[j].operation).is_none() {
                        match compile_map_stage(&stages[j], scalars) {
                            Ok(op) => { map_ops.push(op); j += 1; }
                            Err(_) => break,
                        }
                    }
                    if map_ops.len() >= 2 {
                        data = dispatch_map_chain(gpu, &map_ops, &data)?;
                        i = j;
                        continue;
                    }
                    // Single MapOp: fall through to existing dispatch logic below
                }
                // Try 2-stage structural fusion patterns (ScaleShift, Normalize)
                if i + 1 < stages.len() {
                    if let Some((fused_op, consumed)) = try_fuse(&stages[i..], scalars) {
                        data = dispatch_fused(gpu, fused_op, &data)?;
                        i += consumed;
                        continue;
                    }
                }
                // Check if this is a function call
                if let Some(fn_def) = functions.get(&stages[i].operation) {
                    let inlined = inline_fn_call(fn_def, &stages[i].args);
                    for stage in &inlined {
                        // Recursively check for nested function calls
                        if let Some(nested_fn) = functions.get(&stage.operation) {
                            let nested = inline_fn_call(nested_fn, &stage.args);
                            for ns in &nested {
                                data = dispatch_stage(ns, &data, scalars, gpu)?;
                            }
                        } else {
                            data = dispatch_stage(stage, &data, scalars, gpu)?;
                        }
                    }
                    i += 1;
                    continue;
                }
                // Single stage dispatch
                data = dispatch_stage(&stages[i], &data, scalars, gpu)?;
                i += 1;
            }
            Ok(data)
        }
    }
}

/// Dispatch a single pipeline stage.
fn dispatch_stage(
    call: &StageCall,
    input: &[f32],
    scalars: &HashMap<String, Value>,
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
) -> Result<Vec<f32>, CliError> {
    match call.operation.as_str() {
        // Map operations
        "multiply" | "add" | "subtract" | "divide" | "abs" | "sqrt"
        | "negate" | "mod" | "pow" | "exp" | "log" | "floor" | "ceil" | "round"
        | "min" | "max" | "clamp" | "sin" | "cos" => {
            let op = compile_map_stage(call, scalars)?;
            dispatch_map(gpu, op, input)
        }
        // Temporal operations
        "ema" => {
            let alpha = resolve_arg(&call.args[0], scalars)? as f32;
            dispatch_temporal_op(gpu, TemporalOp::Ema(alpha), input)
        }
        "decay" => {
            let factor = resolve_arg(&call.args[0], scalars)? as f32;
            dispatch_temporal_op(gpu, TemporalOp::Decay(factor), input)
        }
        // Scan operations
        "prefix_sum" => {
            dispatch_scan_op(gpu, input)
        }
        // Channel-aware operations (RGB interleaved, CPU)
        "warm" => {
            let amount = resolve_one_arg(call, scalars)? as f32;
            Ok(cpu_channel_shift(input, amount, 0.0, -amount))
        }
        "cool" => {
            let amount = resolve_one_arg(call, scalars)? as f32;
            Ok(cpu_channel_shift(input, -amount, 0.0, amount))
        }
        "tint" => {
            let (r_shift, b_shift) = resolve_two_args(call, scalars)?;
            Ok(cpu_channel_shift(input, r_shift as f32, 0.0, b_shift as f32))
        }
        _ => Err(CliError::UnknownOperation(call.operation.clone())),
    }
}

/// Map a StageCall to a MapOp, resolving scalar refs in args.
fn compile_map_stage(call: &StageCall, scalars: &HashMap<String, Value>) -> Result<MapOp, CliError> {
    match call.operation.as_str() {
        "multiply" => {
            let arg = resolve_one_arg(call, scalars)?;
            Ok(MapOp::Multiply(arg as f32))
        }
        "add" => {
            let arg = resolve_one_arg(call, scalars)?;
            Ok(MapOp::Add(arg as f32))
        }
        "subtract" => {
            let arg = resolve_one_arg(call, scalars)?;
            Ok(MapOp::Subtract(arg as f32))
        }
        "divide" => {
            let arg = resolve_one_arg(call, scalars)?;
            Ok(MapOp::Divide(arg as f32))
        }
        "abs" => Ok(MapOp::Abs),
        "sqrt" => Ok(MapOp::Sqrt),
        "negate" => Ok(MapOp::Negate),
        "mod" => {
            let arg = resolve_one_arg(call, scalars)?;
            Ok(MapOp::Mod(arg as f32))
        }
        "pow" => {
            let arg = resolve_one_arg(call, scalars)?;
            Ok(MapOp::Pow(arg as f32))
        }
        "exp" => Ok(MapOp::Exp),
        "log" => Ok(MapOp::Log),
        "floor" => Ok(MapOp::Floor),
        "ceil" => Ok(MapOp::Ceil),
        "round" => Ok(MapOp::Round),
        "min" => {
            let arg = resolve_one_arg(call, scalars)?;
            Ok(MapOp::Min(arg as f32))
        }
        "max" => {
            let arg = resolve_one_arg(call, scalars)?;
            Ok(MapOp::Max(arg as f32))
        }
        "clamp" => {
            let (lo, hi) = resolve_two_args(call, scalars)?;
            Ok(MapOp::Clamp(lo as f32, hi as f32))
        }
        "sin" => Ok(MapOp::Sin),
        "cos" => Ok(MapOp::Cos),
        _ => Err(CliError::UnknownOperation(call.operation.clone())),
    }
}

fn resolve_one_arg(call: &StageCall, scalars: &HashMap<String, Value>) -> Result<f64, CliError> {
    if call.args.len() != 1 {
        return Err(CliError::Compile(format!(
            "{}() requires exactly 1 argument, got {}",
            call.operation,
            call.args.len()
        )));
    }
    resolve_arg(&call.args[0], scalars)
}

fn resolve_two_args(call: &StageCall, scalars: &HashMap<String, Value>) -> Result<(f64, f64), CliError> {
    if call.args.len() != 2 {
        return Err(CliError::Compile(format!(
            "{}() requires exactly 2 arguments, got {}",
            call.operation,
            call.args.len()
        )));
    }
    let a = resolve_arg(&call.args[0], scalars)?;
    let b = resolve_arg(&call.args[1], scalars)?;
    Ok((a, b))
}

/// Resolve a stage argument: literal or scalar reference.
fn resolve_arg(arg: &Arg, scalars: &HashMap<String, Value>) -> Result<f64, CliError> {
    match arg {
        Arg::Literal(n) => Ok(*n),
        Arg::IntLiteral(n) => Ok(*n as f64),
        Arg::Ref(name) => {
            let v = scalars.get(name)
                .ok_or_else(|| CliError::UndefinedScalar(name.clone()))?;
            Ok(v.as_float()? as f64)
        }
    }
}

// ── Fusion ──────────────────────────────────────────────────────────

/// Try to fuse adjacent stages into a single kernel.
/// Returns `(FusedOp, stages_consumed)` if fusion is possible.
fn try_fuse(stages: &[StageCall], scalars: &HashMap<String, Value>) -> Option<(FusedOp, usize)> {
    if stages.len() < 2 {
        return None;
    }

    let a = &stages[0];
    let b = &stages[1];

    // subtract(a) |> divide(b) → Normalize { min: a, max: a + b }
    if a.operation == "subtract" && b.operation == "divide" {
        let va = resolve_arg(&a.args[0], scalars).ok()? as f32;
        let vb = resolve_arg(&b.args[0], scalars).ok()? as f32;
        return Some((FusedOp::Normalize { min: va, max: va + vb }, 2));
    }

    // multiply(a) |> add(b) → ScaleShift { scale: a, bias: b }
    if a.operation == "multiply" && b.operation == "add" {
        let va = resolve_arg(&a.args[0], scalars).ok()? as f32;
        let vb = resolve_arg(&b.args[0], scalars).ok()? as f32;
        return Some((FusedOp::ScaleShift { scale: va, bias: vb }, 2));
    }

    None
}

// ── Dispatch helpers ────────────────────────────────────────────────

/// Dispatch a map operation on GPU, or fall back to CPU.
fn dispatch_map(
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    op: MapOp,
    input: &[f32],
) -> Result<Vec<f32>, CliError> {
    if let Some(gpu) = gpu {
        // Check GPU quota: map dispatch allocates ~2x input (input + output buffers)
        check_gpu_quota((input.len() * std::mem::size_of::<f32>() * 2) as u64)?;
        octoflow_vulkan::dispatch_map_op(gpu, op, input)
            .map_err(|e| CliError::Gpu(format!("{}", e)))
    } else {
        Ok(cpu_map(op, input))
    }
}

/// Dispatch a chain of map operations as a single fused GPU kernel.
///
/// All N ops execute in one Vulkan dispatch — intermediate values stay in
/// GPU registers, no intermediate buffer round-trips to CPU memory.
/// This is the core optimization that closes the gap with CUDA kernels.
fn dispatch_map_chain(
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    ops: &[MapOp],
    input: &[f32],
) -> Result<Vec<f32>, CliError> {
    if ops.is_empty() {
        return Ok(input.to_vec());
    }
    // Apply ops sequentially — each through dispatch_map_op (pre-built kernels).
    // Future: fused multi-op kernel from .flow emitter.
    let mut data = input.to_vec();
    for &op in ops {
        if let Some(gpu) = gpu {
            data = octoflow_vulkan::dispatch_map_op(gpu, op, &data)
                .map_err(|e| CliError::Gpu(format!("{}", e)))?;
        } else {
            data = cpu_map(op, &data);
        }
    }
    Ok(data)
}

/// Dispatch a fused operation on GPU, or fall back to CPU.
fn dispatch_fused(
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    op: FusedOp,
    input: &[f32],
) -> Result<Vec<f32>, CliError> {
    if let Some(gpu) = gpu {
        static KERNEL_NORMALIZE: &[u8] = include_bytes!("../../../stdlib/loom/kernels/math/normalize_pc.spv");
        static KERNEL_SCALE_SHIFT: &[u8] = include_bytes!("../../../stdlib/loom/kernels/math/scale_shift_pc.spv");
        let (kernel, pc) = match op {
            FusedOp::Normalize { min, max } => (KERNEL_NORMALIZE, vec![min, max]),
            FusedOp::ScaleShift { scale, bias } => (KERNEL_SCALE_SHIFT, vec![scale, bias]),
        };
        octoflow_vulkan::dispatch_compute_pc(gpu, kernel, input, 256, &pc)
            .map_err(|e| CliError::Gpu(format!("{}", e)))
    } else {
        Ok(cpu_fused(op, input))
    }
}

/// Dispatch a temporal operation (CPU — GPU temporal kernel removed).
fn dispatch_temporal_op(
    _gpu: &Option<octoflow_vulkan::VulkanCompute>,
    op: TemporalOp,
    input: &[f32],
) -> Result<Vec<f32>, CliError> {
    Ok(cpu_temporal(op, input))
}

/// Dispatch a prefix sum scan (CPU — GPU scan kernel removed; use gpu_dlb_scan.flow for GPU).
fn dispatch_scan_op(
    _gpu: &Option<octoflow_vulkan::VulkanCompute>,
    input: &[f32],
) -> Result<Vec<f32>, CliError> {
    Ok(cpu_prefix_sum(input))
}

// ── CPU fallbacks ───────────────────────────────────────────────────

/// CPU fallback for map operations.
fn cpu_map(op: MapOp, input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| match op {
            MapOp::Multiply(s) => x * s,
            MapOp::Add(s) => x + s,
            MapOp::Subtract(s) => x - s,
            MapOp::Divide(s) => x / s,
            MapOp::Abs => x.abs(),
            MapOp::Sqrt => x.sqrt(),
            MapOp::Negate => -x,
            MapOp::Mod(s) => x % s,
            MapOp::Pow(s) => x.powf(s),
            MapOp::Exp => x.exp(),
            MapOp::Log => x.ln(),
            MapOp::Floor => x.floor(),
            MapOp::Ceil => x.ceil(),
            MapOp::Round => x.round(),
            MapOp::Min(s) => x.min(s),
            MapOp::Max(s) => x.max(s),
            MapOp::Clamp(lo, hi) => x.clamp(lo, hi),
            MapOp::Sin => x.sin(),
            MapOp::Cos => x.cos(),
        })
        .collect()
}

/// CPU fallback for fused operations.
fn cpu_fused(op: FusedOp, input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| match op {
            FusedOp::Normalize { min, max } => (x - min) / (max - min),
            FusedOp::ScaleShift { scale, bias } => x * scale + bias,
        })
        .collect()
}

/// CPU fallback for reduce operations.
/// CPU fallback for matrix multiplication.
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn cpu_transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut b = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            b[j * rows + i] = a[i * cols + j];
        }
    }
    b
}

/// Extract array arg as Vec<f32> from a ScalarExpr::Ref.
fn extract_array_arg(
    fn_name: &str,
    arg: &ScalarExpr,
    arrays: &HashMap<String, Vec<Value>>,
) -> Result<Vec<f32>, CliError> {
    if let ScalarExpr::Ref(name) = arg {
        // Fast path: check GPU arrays first (clone of Vec<f32> = 40MB vs 560MB for Vec<Value>)
        if let Some(gpu_arr) = gpu_array_get(name) {
            return Ok(gpu_arr);
        }
        // Slow path: convert from Value
        let arr = arrays.get(name).ok_or_else(|| {
            CliError::Compile(format!("{}(): array '{}' not found", fn_name, name))
        })?;
        Ok(arr.iter().map(|v| v.as_float().unwrap_or(0.0)).collect())
    } else {
        Err(CliError::Compile(format!("{}() argument must be an array name", fn_name)))
    }
}

/// GPU map dispatch with CPU fallback.
fn dispatch_gpu_map(
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    op: MapOp,
    data: &[f32],
) -> Result<Vec<f32>, CliError> {
    if let Some(ref gpu_dev) = gpu {
        octoflow_vulkan::dispatch_map_op(gpu_dev, op, data)
            .map_err(|e| CliError::Gpu(format!("{}", e)))
    } else {
        Ok(cpu_map(op, data))
    }
}

/// GPU binary op dispatch with CPU fallback.
fn dispatch_gpu_binop(
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    op: BinaryOp,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, CliError> {
    if a.len() != b.len() {
        return Err(CliError::Compile(format!("gpu binary op: arrays must have same length ({} vs {})", a.len(), b.len())));
    }
    if let Some(ref gpu_dev) = gpu {
        octoflow_vulkan::dispatch_binop(gpu_dev, op, a, b)
            .map_err(|e| CliError::Gpu(format!("{}", e)))
    } else {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| match op {
            BinaryOp::Add => x + y,
            BinaryOp::Sub => x - y,
            BinaryOp::Mul => x * y,
            BinaryOp::Div => x / y,
        }).collect())
    }
}

/// GPU select dispatch with CPU fallback.
fn dispatch_gpu_select(
    gpu: &Option<octoflow_vulkan::VulkanCompute>,
    cond: &[f32],
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, CliError> {
    if cond.len() != a.len() || a.len() != b.len() {
        return Err(CliError::Compile("gpu_where(): all arrays must have same length".into()));
    }
    if let Some(ref gpu_dev) = gpu {
        octoflow_vulkan::dispatch_select(gpu_dev, cond, a, b)
            .map_err(|e| CliError::Gpu(format!("{}", e)))
    } else {
        Ok(cond.iter().zip(a.iter()).zip(b.iter()).map(|((&c, &x), &y)| {
            if c != 0.0 { x } else { y }
        }).collect())
    }
}

fn cpu_reduce(op: ReduceOp, data: &[f32]) -> f32 {
    match op {
        ReduceOp::Sum => data.iter().sum(),
        ReduceOp::Min => data.iter().cloned().fold(f32::INFINITY, f32::min),
        ReduceOp::Max => data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        ReduceOp::Mul => data.iter().cloned().fold(1.0, |a, b| a * b),
    }
}

/// CPU fallback for temporal operations (1D: single instrument).
fn cpu_temporal(op: TemporalOp, data: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    if data.is_empty() {
        return out;
    }
    out[0] = data[0];
    for t in 1..data.len() {
        out[t] = match op {
            TemporalOp::Ema(alpha) => alpha * data[t] + (1.0 - alpha) * out[t - 1],
            TemporalOp::Decay(factor) => data[t] + factor * out[t - 1],
        };
    }
    out
}

/// CPU fallback for prefix sum.
fn cpu_prefix_sum(data: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    if data.is_empty() {
        return out;
    }
    out[0] = data[0];
    for i in 1..data.len() {
        out[i] = out[i - 1] + data[i];
    }
    out
}

/// CPU channel-aware shift for RGB interleaved data.
///
/// Adds `r_shift`, `g_shift`, `b_shift` to the respective channels.
/// Operates on flat [R0,G0,B0,R1,G1,B1,...] layout with stride 3.
fn cpu_channel_shift(input: &[f32], r_shift: f32, g_shift: f32, b_shift: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    let mut i = 0;
    while i + 2 < out.len() {
        out[i] += r_shift;
        out[i + 1] += g_shift;
        out[i + 2] += b_shift;
        i += 3;
    }
    out
}

/// Resolve a path relative to the base directory of the .flow file.
///
/// Security: rejects path traversal (`..` components) to prevent .flow
/// programs from escaping their directory subtree. Absolute paths are
/// allowed (needed for CLI-rewritten paths from OctoMedia).
fn resolve_path(base_dir: &str, path: &str) -> Result<String, CliError> {
    // Reject path traversal via ".." components
    for component in std::path::Path::new(path).components() {
        if let std::path::Component::ParentDir = component {
            return Err(CliError::Security(format!(
                "path traversal ('..') is not allowed: '{}'", path)));
        }
    }

    // Absolute paths pass through directly (from CLI rewrites)
    if std::path::Path::new(path).is_absolute() {
        return Ok(path.to_string());
    }

    let base = std::path::Path::new(base_dir);
    Ok(base.join(path).to_string_lossy().into_owned())
}

/// Resolve a module path against the stdlib directory (relative to the binary).
/// Searches multiple candidate locations relative to the executable.
fn resolve_stdlib_module(module: &str) -> Result<String, CliError> {
    // Build list of candidate stdlib directories
    let mut candidates = Vec::new();

    // CWD-relative: stdlib/ from the current working directory or parent
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("stdlib"));
        candidates.push(cwd.join("..").join("stdlib"));
    }

    // Exe-relative: various layouts
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            // Installed layout: bin/octoflow + ../stdlib/
            candidates.push(exe_dir.join("..").join("stdlib"));
            // Dev layout: target/debug/octoflow + ../../stdlib/
            candidates.push(exe_dir.join("..").join("..").join("stdlib"));
            // Test layout: target/debug/deps/ + ../../../stdlib/
            candidates.push(exe_dir.join("..").join("..").join("..").join("stdlib"));
            // Flat layout: octoflow + stdlib/
            candidates.push(exe_dir.join("stdlib"));
        }
    }

    for stdlib_dir in &candidates {
        let candidate = stdlib_dir.join(format!("{}.flow", module));
        if let Ok(cp) = candidate.canonicalize() {
            return Ok(cp.to_string_lossy().into_owned());
        }
    }

    // Last resort: try OCTOFLOW_STDLIB env var
    if let Ok(stdlib_dir) = std::env::var("OCTOFLOW_STDLIB") {
        let candidate = std::path::Path::new(&stdlib_dir).join(format!("{}.flow", module));
        if let Ok(cp) = candidate.canonicalize() {
            return Ok(cp.to_string_lossy().into_owned());
        }
    }
    Err(CliError::Compile(format!(
        "cannot resolve module '{}': not found locally or in stdlib", module)))
}

// ── CSV helpers (inlined from csv_io.rs — Phase 120) ─────────────────

/// Read a single-column CSV file of f32 values.
fn csv_read_floats(path: &str) -> Result<Vec<f32>, CliError> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
    if metadata.len() > crate::MAX_CSV_FILE_BYTES {
        return Err(CliError::Security(format!(
            "{}: file size {} bytes exceeds limit ({} MB)",
            path, metadata.len(), crate::MAX_CSV_FILE_BYTES / (1024 * 1024))));
    }
    let content = std::fs::read_to_string(path)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
    let mut values = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
        let value: f32 = trimmed.parse().map_err(|_| {
            CliError::Csv(format!("{}:{}: invalid number: {}", path, i + 1, trimmed))
        })?;
        values.push(value);
        if values.len() > crate::MAX_CSV_VALUES {
            return Err(CliError::Security(format!(
                "{}: exceeds {} value limit", path, crate::MAX_CSV_VALUES)));
        }
    }
    if values.is_empty() {
        return Err(CliError::Csv(format!("{}: no data found", path)));
    }
    Ok(values)
}

/// Write a single-column CSV file of f32 values.
fn csv_write_floats(path: &str, data: &[f32]) -> Result<(), CliError> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
        }
    }
    let mut output = String::new();
    for value in data {
        output.push_str(&format!("{}\n", value));
    }
    std::fs::write(path, output)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))
}

/// Parse a CSV line respecting quoted fields.
fn csv_parse_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') { current.push('"'); chars.next(); }
                else { in_quotes = false; }
            } else { current.push(ch); }
        } else {
            match ch {
                '"' => in_quotes = true,
                ',' => { fields.push(current.trim().to_string()); current = String::new(); }
                _ => current.push(ch),
            }
        }
    }
    fields.push(current.trim().to_string());
    fields
}

/// Read structured CSV with headers into array of maps.
fn csv_read_structured(path: &str) -> Result<Vec<HashMap<String, Value>>, CliError> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::Io(format!("read_csv(\"{}\"): {}", path, e)))?;
    if metadata.len() > crate::MAX_CSV_FILE_BYTES {
        return Err(CliError::Security(format!(
            "read_csv(\"{}\"): file size {} bytes exceeds limit ({} MB)",
            path, metadata.len(), crate::MAX_CSV_FILE_BYTES / (1024 * 1024))));
    }
    let content = std::fs::read_to_string(path)
        .map_err(|e| CliError::Io(format!("read_csv(\"{}\"): {}", path, e)))?;
    let mut lines = content.lines();
    let headers = loop {
        match lines.next() {
            Some(line) => {
                let t = line.trim();
                if !t.is_empty() && !t.starts_with('#') { break csv_parse_line(t); }
            }
            None => return Err(CliError::Csv(format!("read_csv(\"{}\"): no header row found", path))),
        }
    };
    if headers.is_empty() {
        return Err(CliError::Csv(format!("read_csv(\"{}\"): empty header row", path)));
    }
    let mut rows: Vec<HashMap<String, Value>> = Vec::new();
    for line in lines {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') { continue; }
        let fields = csv_parse_line(t);
        let mut row = HashMap::new();
        for (i, header) in headers.iter().enumerate() {
            let val = fields.get(i).map(|s| s.as_str()).unwrap_or("");
            row.insert(header.clone(), if let Ok(f) = val.parse::<f32>() {
                Value::Float(f)
            } else {
                Value::Str(val.to_string())
            });
        }
        rows.push(row);
        if rows.len() > crate::MAX_CSV_VALUES {
            return Err(CliError::Security(format!(
                "read_csv(\"{}\"): exceeds {} row limit", path, crate::MAX_CSV_VALUES)));
        }
    }
    Ok(rows)
}

/// Write structured CSV from array of Value::Map records.
fn csv_write_structured(path: &str, rows: &[Value]) -> Result<(), CliError> {
    if rows.is_empty() {
        return Err(CliError::Compile("write_csv(): empty array, nothing to write".into()));
    }
    let first_map = match &rows[0] {
        Value::Map(m) => m,
        _ => return Err(CliError::Compile("write_csv(): array elements must be maps".into())),
    };
    let mut headers: Vec<String> = first_map.keys().cloned().collect();
    headers.sort();
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CliError::Io(format!("write_csv(\"{}\"): {}", path, e)))?;
        }
    }
    let mut output = String::new();
    let csv_esc = |s: &str| -> String {
        if s.contains(',') || s.contains('"') || s.contains('\n') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else { s.to_string() }
    };
    output.push_str(&headers.iter().map(|h| csv_esc(h)).collect::<Vec<_>>().join(","));
    output.push('\n');
    for row_val in rows {
        let map = match row_val {
            Value::Map(m) => m,
            _ => return Err(CliError::Compile("write_csv(): all elements must be maps".into())),
        };
        let fields: Vec<String> = headers.iter().map(|h| {
            match map.get(h) {
                Some(Value::Float(f)) => {
                    if *f == (*f as i64 as f32) && f.is_finite() { format!("{}", *f as i64) }
                    else { format!("{}", f) }
                }
                Some(Value::Int(i)) => format!("{}", i),
                Some(Value::Str(s)) => csv_esc(s),
                Some(Value::Map(_)) => "{}".to_string(),
                Some(Value::None) => String::new(),
                None => String::new(),
            }
        }).collect();
        output.push_str(&fields.join(","));
        output.push('\n');
    }
    std::fs::write(path, output)
        .map_err(|e| CliError::Io(format!("write_csv(\"{}\"): {}", path, e)))
}


#[cfg(test)]
#[path = "tests.rs"]
mod tests;


/// Walk an expression tree to find source image dimensions.
fn find_source_dims(
    expr: &Expr,
    image_dims: &HashMap<String, (u32, u32)>,
) -> Option<(u32, u32)> {
    match expr {
        Expr::Tap { path } => image_dims.get(path).copied(),
        Expr::RandomStream { .. } => None,
        Expr::Cache { inner, .. } => find_source_dims(inner, image_dims),
        Expr::Ref { name } => image_dims.get(name).copied(),
        Expr::Pipe { input, .. } => find_source_dims(input, image_dims),
    }
}
