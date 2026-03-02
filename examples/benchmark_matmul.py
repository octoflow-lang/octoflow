"""
1024x1024 Matrix Multiply benchmark — Python NumPy + PyTorch CUDA
Equivalent to OctoFlow mat_mul(a, b, 1024, 1024, 1024)

Run: python examples/benchmark_matmul.py
"""

import numpy as np
import time
import sys

N = 1024

print(f"=== Matrix Multiply Benchmark ({N}x{N}) ===\n")

# --- NumPy CPU ---
a_np = np.random.randn(N, N).astype(np.float32)
b_np = np.random.randn(N, N).astype(np.float32)

# Warmup
_ = a_np @ b_np

t0 = time.perf_counter()
c_np = a_np @ b_np
t1 = time.perf_counter()
numpy_ms = (t1 - t0) * 1000

print(f"NumPy (CPU, OpenBLAS):  {numpy_ms:.1f} ms")
print(f"  c[0,0] = {c_np[0,0]:.4f}")

# --- PyTorch CUDA ---
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nPyTorch CUDA GPU: {gpu_name}")

        a_t = torch.from_numpy(a_np).to(device)
        b_t = torch.from_numpy(b_np).to(device)

        # Warmup (cuBLAS init + JIT)
        for _ in range(3):
            _ = torch.mm(a_t, b_t)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        c_t = torch.mm(a_t, b_t)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pytorch_ms = (t1 - t0) * 1000

        c_check = c_t.cpu().numpy()
        print(f"PyTorch CUDA (cuBLAS): {pytorch_ms:.2f} ms")
        print(f"  c[0,0] = {c_check[0,0]:.4f}")

        # Full pipeline: numpy array → CUDA → matmul → back to CPU
        t0 = time.perf_counter()
        a_full = torch.from_numpy(a_np).to(device)
        b_full = torch.from_numpy(b_np).to(device)
        c_full = torch.mm(a_full, b_full)
        torch.cuda.synchronize()
        c_result = c_full.cpu().numpy()
        t1 = time.perf_counter()
        pytorch_full_ms = (t1 - t0) * 1000

        print(f"PyTorch CUDA (full):   {pytorch_full_ms:.1f} ms  (transfer + compute + readback)")
    else:
        print("\nPyTorch: CUDA not available")
        pytorch_ms = 0
        pytorch_full_ms = 0
except ImportError:
    print("\nPyTorch not installed")
    pytorch_ms = 0
    pytorch_full_ms = 0

print(f"\n--- Summary ---")
print(f"NumPy CPU:         {numpy_ms:.1f} ms")
if pytorch_ms > 0:
    print(f"PyTorch cuBLAS:    {pytorch_ms:.2f} ms  (compute only)")
    print(f"PyTorch full:      {pytorch_full_ms:.1f} ms  (transfer + compute + readback)")
