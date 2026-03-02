#!/usr/bin/env python3
"""
GPU Double Benchmark — Python + CUDA Driver API via ctypes
Equivalent to OctoFlow's test_vk_dispatch.flow.

Uses pre-compiled PTX (no NVRTC needed) + CUDA Driver API (nvcuda.dll).
Cold-start benchmark: includes all init, just like OctoFlow.
"""
import ctypes
import ctypes.util
import time
import os
import struct

# ── Locate CUDA driver library ───────────────────────────────────────
nvcuda = ctypes.CDLL("nvcuda.dll")

# ── CUDA Driver API function prototypes ──────────────────────────────
cuInit = nvcuda.cuInit
cuDeviceGet = nvcuda.cuDeviceGet
cuDeviceGetName = nvcuda.cuDeviceGetName
cuCtxCreate = nvcuda.cuCtxCreate_v2
cuCtxDestroy = nvcuda.cuCtxDestroy_v2
cuModuleLoadData = nvcuda.cuModuleLoadData
cuModuleGetFunction = nvcuda.cuModuleGetFunction
cuModuleUnload = nvcuda.cuModuleUnload
cuMemAlloc = nvcuda.cuMemAlloc_v2
cuMemFree = nvcuda.cuMemFree_v2
cuMemcpyHtoD = nvcuda.cuMemcpyHtoD_v2
cuMemcpyDtoH = nvcuda.cuMemcpyDtoH_v2
cuLaunchKernel = nvcuda.cuLaunchKernel
cuCtxSynchronize = nvcuda.cuCtxSynchronize

# Set return types for 64-bit pointer returns
cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]

N = 256
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PTX_PATH = os.path.join(SCRIPT_DIR, "cuda_kernels", "double_it.ptx")


def check(result, name=""):
    if result != 0:
        raise RuntimeError(f"CUDA error in {name}: {result}")


def bench_cuda_driver():
    """Full CUDA pipeline: init + load PTX + dispatch + verify."""
    t_start = time.perf_counter()

    # ── Phase 1: Init driver + create context ────────────────────────
    check(cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check(cuCtxCreate(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
    t_init = time.perf_counter()

    # ── Phase 2: Load pre-compiled PTX ───────────────────────────────
    with open(PTX_PATH, "rb") as f:
        ptx_data = f.read()
    module = ctypes.c_void_p()
    check(cuModuleLoadData(ctypes.byref(module), ptx_data), "cuModuleLoadData")
    func = ctypes.c_void_p()
    check(cuModuleGetFunction(ctypes.byref(func), module, b"double_it"), "cuModuleGetFunction")
    t_shader = time.perf_counter()

    # ── Phase 3: Allocate GPU memory + upload ────────────────────────
    buf_size = N * 4  # float32 = 4 bytes
    d_in = ctypes.c_uint64()
    d_out = ctypes.c_uint64()
    check(cuMemAlloc(ctypes.byref(d_in), buf_size), "cuMemAlloc in")
    check(cuMemAlloc(ctypes.byref(d_out), buf_size), "cuMemAlloc out")

    # Build input: [1.0, 2.0, ..., 256.0]
    h_in = struct.pack(f"{N}f", *[float(i + 1) for i in range(N)])
    check(cuMemcpyHtoD(d_in, h_in, buf_size), "cuMemcpyHtoD")
    t_buffer = time.perf_counter()

    # ── Phase 4: Dispatch ────────────────────────────────────────────
    n_val = ctypes.c_int(N)
    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.pointer(d_in), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(d_out), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(n_val), ctypes.c_void_p),
    )
    check(cuLaunchKernel(
        func,
        1, 1, 1,        # grid (1 block)
        256, 1, 1,       # block (256 threads)
        0, None,         # shared mem, stream
        args, None       # args, extra
    ), "cuLaunchKernel")
    check(cuCtxSynchronize(), "cuCtxSynchronize")
    t_dispatch = time.perf_counter()

    # ── Phase 5: Readback + verify ───────────────────────────────────
    h_out = (ctypes.c_char * buf_size)()
    check(cuMemcpyDtoH(h_out, d_out, buf_size), "cuMemcpyDtoH")
    results = struct.unpack(f"{N}f", bytes(h_out))
    t_readback = time.perf_counter()

    # Verify
    errors = 0
    for i in range(N):
        expected = float(i + 1) * 2.0
        if abs(results[i] - expected) > 0.001:
            if errors < 3:
                print(f"  MISMATCH [{i}]: got {results[i]}, expected {expected}")
            errors += 1

    # Cleanup
    cuMemFree(d_in)
    cuMemFree(d_out)
    cuModuleUnload(module)
    cuCtxDestroy(ctx)

    t_total = time.perf_counter()

    total_ms = (t_total - t_start) * 1000
    init_ms = (t_init - t_start) * 1000
    shader_ms = (t_shader - t_init) * 1000
    buffer_ms = (t_buffer - t_shader) * 1000
    dispatch_ms = (t_dispatch - t_buffer) * 1000
    readback_ms = (t_readback - t_dispatch) * 1000
    cleanup_ms = (t_total - t_readback) * 1000

    return {
        "ok": errors == 0,
        "total_ms": total_ms,
        "init_ms": init_ms,
        "shader_ms": shader_ms,
        "buffer_ms": buffer_ms,
        "dispatch_ms": dispatch_ms,
        "readback_ms": readback_ms,
        "cleanup_ms": cleanup_ms,
        "errors": errors,
    }


if __name__ == "__main__":
    print(f"=== GPU Double Benchmark: Python + CUDA Driver API (N={N}) ===")
    print(f"PTX: {PTX_PATH}")
    print()

    try:
        # Get GPU name
        dev = ctypes.c_int()
        cuInit(0)
        cuDeviceGet(ctypes.byref(dev), 0)
        name_buf = ctypes.create_string_buffer(256)
        cuDeviceGetName(name_buf, 256, dev)
        print(f"GPU: {name_buf.value.decode()}")
        print()

        # Cold start run (the one that matters for comparison)
        print("--- Cold Start (full pipeline) ---")
        r = bench_cuda_driver()
        if r["ok"]:
            print(f"Python+CUDA: ALL PASS ({N} elements)")
        else:
            print(f"Python+CUDA: FAIL ({r['errors']} mismatches)")
        print(f"  Total:    {r['total_ms']:.1f}ms")
        print(f"  Init:     {r['init_ms']:.1f}ms  (cuInit + context)")
        print(f"  Shader:   {r['shader_ms']:.1f}ms  (load PTX + get function)")
        print(f"  Buffer:   {r['buffer_ms']:.1f}ms  (alloc + upload)")
        print(f"  Dispatch: {r['dispatch_ms']:.1f}ms  (launch + sync)")
        print(f"  Readback: {r['readback_ms']:.1f}ms  (download + verify)")
        print(f"  Cleanup:  {r['cleanup_ms']:.1f}ms")
        print()

        # Warm runs (context reuse would matter for real apps)
        print("--- Warm Runs (3x, new context each time) ---")
        times = []
        for i in range(3):
            r = bench_cuda_driver()
            times.append(r["total_ms"])
            status = "PASS" if r["ok"] else "FAIL"
            print(f"  Run {i+1}: {r['total_ms']:.1f}ms [{status}]")
        print(f"  Average: {sum(times)/len(times):.1f}ms")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
