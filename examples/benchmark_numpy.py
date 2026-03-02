"""
Python + NumPy CPU Benchmark
Equivalent operation to benchmark.flow: 1M float scale-shift-sum

Run: python examples/benchmark_numpy.py

Comparison target for OctoFlow GPU benchmark.
"""

import numpy as np
import time
import os

DATA_FILE = "examples/test_data/bench_1m.csv"

if not os.path.exists(DATA_FILE):
    print(f"ERROR: {DATA_FILE} not found.")
    print("Generate it first: flowgpu-cli run examples/benchmark.flow")
    raise SystemExit(1)

# --- Warm-up (prime OS file cache) ---
_ = np.loadtxt(DATA_FILE, dtype=np.float32)

# --- Benchmark ---
t0 = time.perf_counter()

data = np.loadtxt(DATA_FILE, dtype=np.float32)   # file I/O
scaled = data * 2.0                               # map: multiply
shifted = scaled + 1.0                            # map: add
result = shifted.sum()                            # reduce: sum

t1 = time.perf_counter()

elapsed_ms = (t1 - t0) * 1000.0

print("=== Python + NumPy CPU Benchmark ===")
print(f"Operation : 1M float scale-shift-sum (CPU NumPy)")
print(f"Result    : {result:.2f}")
print(f"Time      : {elapsed_ms:.2f} ms")
print()

# --- Memory-only timing (no file I/O) ---
data = np.loadtxt(DATA_FILE, dtype=np.float32)

t0 = time.perf_counter()
result2 = (data * 2.0 + 1.0).sum()
t1 = time.perf_counter()

compute_ms = (t1 - t0) * 1000.0
print(f"Compute only (no I/O): {compute_ms:.3f} ms  (result={result2:.2f})")
