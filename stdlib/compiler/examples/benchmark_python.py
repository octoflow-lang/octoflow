#!/usr/bin/env python3
"""
benchmark_python.py — Python GPU sigmoid benchmark
Same 1M-element sigmoid task as benchmark_1m.flow.
Uses the same bench_1m.csv data file.

Runners:
  1. NumPy  (CPU vectorized)
  2. PyTorch CUDA  (GPU tensor ops — closest to "Python+CUDA")
  3. Numba CUDA kernel  (raw CUDA kernel in Python)

Run:
  python stdlib/compiler/examples/benchmark_python.py
"""

import time
import numpy as np

DATA_PATH = "stdlib/compiler/examples/bench_1m.csv"

print("=== Python GPU Benchmark — 1 Million Elements ===")
print("  sigmoid(x) = 1 / (1 + exp(-x))")
print()

# ── Load data ────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
data_np = np.loadtxt(DATA_PATH, dtype=np.float32)
t1 = time.perf_counter()
print(f"  Loaded {len(data_np):,} floats in {(t1-t0)*1000:.1f} ms")
print()

# ── 1. NumPy CPU ──────────────────────────────────────────────────────────────
print("[1] NumPy (CPU vectorized)")
t0 = time.perf_counter()
result_np = np.sum(1.0 / (1.0 + np.exp(-data_np)))
t1 = time.perf_counter()
t_numpy = (t1 - t0) * 1000
print(f"  sum(sigmoid(x)) = {result_np:.4f}")
print(f"  time: {t_numpy:.1f} ms")
print()

# ── 2. PyTorch CUDA ───────────────────────────────────────────────────────────
print("[2] PyTorch CUDA (GPU tensor ops)")
try:
    import torch
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
    else:
        device = torch.cuda.get_device_name(0)
        print(f"  GPU: {device}")

        # Warm up GPU (first CUDA call includes driver init overhead)
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        data_t = torch.tensor(data_np, device="cuda", dtype=torch.float32)
        result_t = torch.sum(1.0 / (1.0 + torch.exp(-data_t))).item()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_torch = (t1 - t0) * 1000

        print(f"  sum(sigmoid(x)) = {result_t:.4f}")
        print(f"  time: {t_torch:.1f} ms  (includes host->device transfer)")

        # Transfer only (separate measurement)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        data_t2 = torch.tensor(data_np, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_transfer = (t1 - t0) * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result_t2 = torch.sum(1.0 / (1.0 + torch.exp(-data_t2))).item()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_compute = (t1 - t0) * 1000

        print(f"  breakdown: transfer={t_transfer:.1f} ms  compute={t_compute:.1f} ms")
        print()

except ImportError:
    print("  SKIP: torch not installed")
    t_torch = None
    print()

# ── 3. Numba CUDA kernel ──────────────────────────────────────────────────────
print("[3] Numba CUDA kernel (raw CUDA in Python)")
try:
    from numba import cuda
    import math

    @cuda.jit
    def sigmoid_kernel(x, out):
        i = cuda.grid(1)
        if i < x.shape[0]:
            out[i] = 1.0 / (1.0 + math.exp(-x[i]))

    N = len(data_np)
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block

    # Warm up (Numba compiles kernel on first call)
    d_x = cuda.to_device(data_np[:1000])
    d_out = cuda.device_array(1000, dtype=np.float32)
    sigmoid_kernel[4, 256](d_x, d_out)
    cuda.synchronize()

    t0 = time.perf_counter()
    d_x = cuda.to_device(data_np)
    d_out = cuda.device_array(N, dtype=np.float32)
    sigmoid_kernel[blocks, threads_per_block](d_x, d_out)
    cuda.synchronize()
    result_numba = d_out.copy_to_host().sum()
    t1 = time.perf_counter()
    t_numba = (t1 - t0) * 1000

    print(f"  sum(sigmoid(x)) = {result_numba:.4f}")
    print(f"  time: {t_numba:.1f} ms  (includes host->device + kernel + device->host)")
    print()

except (ImportError, Exception) as e:
    print(f"  SKIP: {e}")
    t_numba = None
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=== SUMMARY ===")
print(f"  NumPy  (CPU):        {t_numpy:.1f} ms")
if 't_torch' in dir() and t_torch is not None:
    print(f"  PyTorch CUDA (GPU):  {t_torch:.1f} ms  (transfer + compute)")
if 't_numba' in dir() and t_numba is not None:
    print(f"  Numba CUDA (GPU):    {t_numba:.1f} ms  (transfer + kernel + copy)")
print()
print("  OctoFlow results (from benchmark_1m.flow):")
print("  OctoFlow CPU scalar: 1652.5 ms")
print("  OctoFlow GPU pipeline: 203.3 ms  (8.13x vs OctoFlow CPU)")
print()
print("  LOC comparison for the GPU pipeline:")
print("  OctoFlow:      6 lines (stream pipeline)")
print("  PyTorch CUDA:  3 lines (tensor ops, but needs import + warm-up boilerplate)")
print("  Numba CUDA:    10+ lines (kernel def + grid config + transfer + sync)")
