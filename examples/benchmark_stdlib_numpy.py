"""
Python + NumPy stdlib benchmark — equivalent to benchmark_stdlib.flow
Measures: array creation, stats, encoding, path ops

Run: python examples/benchmark_stdlib_numpy.py
"""

import numpy as np
import os
import time
import codecs

t0 = time.perf_counter()

# Array creation: 10K elements
data = np.arange(10000, dtype=np.float32) * 0.01

# Statistics
m = np.mean(data)
med = np.median(data)
sd = np.std(data, ddof=1)   # sample stddev
v = np.var(data, ddof=1)    # sample variance

# Array queries
f = np.where(data == 50.0)[0]
mn = np.min(data)
mx = np.max(data)
fi = data[0]
la = data[-1]

# Slice + range
sl = data[0:100]
rng = np.arange(1000)

# Linear algebra (100-element vectors)
va = np.arange(100, dtype=np.float32)
vb = 100.0 - va
d = np.dot(va, vb)
n = np.linalg.norm(va)

# Path utilities
ext = os.path.splitext("/home/user/data/results.csv")[1].lstrip(".")
fn1 = os.path.basename("/home/user/data/results.csv")
dn = os.path.dirname("/home/user/data/results.csv")

# Date arithmetic
ts = time.time()
ts2 = ts + 24 * 3600
diff = (ts - ts2) / 3600

# Hex encoding
encoded = "OctoFlow GPU Language".encode().hex()
decoded = bytes.fromhex(encoded).decode()

t1 = time.perf_counter()
elapsed = (t1 - t0) * 1000

print("=== Python + NumPy Stdlib Benchmark ===")
print(f"Elements  : 10,000")
print(f"Mean      : {m:.4f}")
print(f"Median    : {med:.4f}")
print(f"StdDev    : {sd:.4f}")
print(f"Variance  : {v:.4f}")
print(f"Min/Max   : {mn:.2f}/{mx:.2f}")
print(f"Dot prod  : {d:.2f}")
print(f"Norm      : {n:.4f}")
print(f"Path ext  : {ext}")
print(f"Filename  : {fn1}")
print(f"Dirname   : {dn}")
print(f"Hex round : {decoded}")
print(f"Time      : {elapsed:.1f} ms")
