"""
10M Element-wise benchmark — Python NumPy + PyTorch CUDA
Equivalent to OctoFlow gpu_add / gpu_mul on 10M floats

Run: python examples/benchmark_elementwise_10m.py
"""

import numpy as np
import time

N = 10_000_000

a = np.arange(N, dtype=np.float32) * 0.0001
b = np.array([((i * 3 + 7) % 10000) * 0.0001 for i in range(N)], dtype=np.float32)

print(f"=== Element-wise Benchmark ({N//1_000_000}M floats) ===\n")

# --- NumPy CPU ---
_ = a + b  # warmup

t0 = time.perf_counter()
c_add = a + b
t1 = time.perf_counter()
numpy_add_ms = (t1 - t0) * 1000

t0 = time.perf_counter()
c_mul = a * b
t1 = time.perf_counter()
numpy_mul_ms = (t1 - t0) * 1000

print(f"NumPy add 10M:  {numpy_add_ms:.1f} ms")
print(f"NumPy mul 10M:  {numpy_mul_ms:.1f} ms")

# --- PyTorch CUDA ---
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nGPU: {gpu_name}")

        a_t = torch.from_numpy(a).to(device)
        b_t = torch.from_numpy(b).to(device)

        # Warmup
        for _ in range(3):
            _ = a_t + b_t
        torch.cuda.synchronize()

        # Compute-only timing
        t0 = time.perf_counter()
        c_t = a_t + b_t
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pytorch_add_ms = (t1 - t0) * 1000

        t0 = time.perf_counter()
        c_t = a_t * b_t
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pytorch_mul_ms = (t1 - t0) * 1000

        print(f"PyTorch add 10M: {pytorch_add_ms:.2f} ms  (compute only, data on GPU)")
        print(f"PyTorch mul 10M: {pytorch_mul_ms:.2f} ms  (compute only, data on GPU)")

        # Full pipeline: create + transfer + compute + readback
        t0 = time.perf_counter()
        a_full = torch.from_numpy(a).to(device)
        b_full = torch.from_numpy(b).to(device)
        c_full = a_full + b_full
        torch.cuda.synchronize()
        result = c_full.cpu().numpy()
        t1 = time.perf_counter()
        pytorch_full_ms = (t1 - t0) * 1000

        print(f"PyTorch add 10M: {pytorch_full_ms:.1f} ms  (full: transfer + compute + readback)")
    else:
        print("\nPyTorch: CUDA not available")
except ImportError:
    print("\nPyTorch not installed")
