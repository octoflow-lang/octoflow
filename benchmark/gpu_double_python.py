#!/usr/bin/env python3
"""
GPU Double Benchmark — Python + PyCUDA
Equivalent to OctoFlow's test_vk_dispatch.flow:
  - Allocates 256-element float32 input on GPU
  - Runs a compute shader that doubles each element
  - Reads back and verifies output[i] == input[i] * 2.0
"""
import time
import numpy as np

try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

N = 256

def bench_pycuda():
    """PyCUDA implementation — compile CUDA C kernel, dispatch, verify."""
    mod = SourceModule("""
    __global__ void double_it(float *in, float *out, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            out[idx] = in[idx] * 2.0f;
        }
    }
    """)
    double_it = mod.get_function("double_it")

    # Allocate and upload
    h_in = np.arange(1, N + 1, dtype=np.float32)
    d_in = drv.mem_alloc(h_in.nbytes)
    d_out = drv.mem_alloc(h_in.nbytes)

    t0 = time.perf_counter()
    drv.memcpy_htod(d_in, h_in)
    double_it(d_in, d_out, np.int32(N), block=(256, 1, 1), grid=(1, 1, 1))
    drv.Context.synchronize()
    h_out = np.empty_like(h_in)
    drv.memcpy_dtoh(h_out, d_out)
    t1 = time.perf_counter()

    # Verify
    expected = h_in * 2.0
    if np.allclose(h_out, expected):
        print(f"PyCUDA: ALL PASS ({N} elements) in {(t1-t0)*1000:.2f}ms")
    else:
        print("PyCUDA: FAIL")

    return t1 - t0

def bench_cupy():
    """CuPy implementation — high-level array API."""
    t0 = time.perf_counter()
    d_in = cp.arange(1, N + 1, dtype=cp.float32)
    d_out = d_in * 2.0
    cp.cuda.Device().synchronize()
    h_out = d_out.get()
    t1 = time.perf_counter()

    expected = np.arange(1, N + 1, dtype=np.float32) * 2.0
    if np.allclose(h_out, expected):
        print(f"CuPy:   ALL PASS ({N} elements) in {(t1-t0)*1000:.2f}ms")
    else:
        print("CuPy: FAIL")

    return t1 - t0

def bench_numpy():
    """NumPy baseline (CPU only)."""
    t0 = time.perf_counter()
    h_in = np.arange(1, N + 1, dtype=np.float32)
    h_out = h_in * 2.0
    t1 = time.perf_counter()

    expected = np.arange(1, N + 1, dtype=np.float32) * 2.0
    if np.allclose(h_out, expected):
        print(f"NumPy:  ALL PASS ({N} elements) in {(t1-t0)*1000:.4f}ms (CPU baseline)")
    else:
        print("NumPy: FAIL")

    return t1 - t0

if __name__ == "__main__":
    print(f"=== GPU Double Benchmark (N={N}) ===")
    print(f"Python + GPU libraries vs OctoFlow")
    print()

    # Warm up
    _ = np.zeros(1)

    bench_numpy()

    if HAS_PYCUDA:
        # Warm up GPU
        bench_pycuda()
        # Actual benchmark (3 runs)
        times = [bench_pycuda() for _ in range(3)]
        print(f"PyCUDA avg: {np.mean(times)*1000:.2f}ms")
    else:
        print("PyCUDA: not installed (pip install pycuda)")

    if HAS_CUPY:
        bench_cupy()
        times = [bench_cupy() for _ in range(3)]
        print(f"CuPy avg:   {np.mean(times)*1000:.2f}ms")
    else:
        print("CuPy:   not installed (pip install cupy-cuda12x)")

    print()
    print("Lines of code comparison:")
    print(f"  Python+PyCUDA:  ~45 lines (+ numpy + pycuda deps)")
    print(f"  Python+CuPy:    ~10 lines (+ numpy + cupy deps)")
    print(f"  OctoFlow native: test_vk_dispatch.flow (~480 lines, zero deps)")
    print(f"  OctoFlow self-hosted: eval.flow interprets test_vk_dispatch.flow")
    print()
    print("Dependency comparison:")
    print(f"  Python+PyCUDA:  Python runtime (~150MB) + NumPy (~30MB) + PyCUDA (~15MB) + CUDA toolkit (~4GB)")
    print(f"  Python+CuPy:    Python runtime (~150MB) + NumPy (~30MB) + CuPy (~500MB) + CUDA toolkit (~4GB)")
    print(f"  OctoFlow:       Single binary (~2MB) + Vulkan driver (pre-installed)")
