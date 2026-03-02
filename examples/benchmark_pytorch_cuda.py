"""
OctoFlow vs PyTorch+CUDA — GPU-to-GPU Benchmark
Same GPU, same operations, same element counts.

Tests:
  1. gpu_fill 2x10M
  2. Element-wise add 10M
  3. Element-wise mul 10M
  4. Reduction sum 10M
  5. Matrix multiply 256/512/1024
  6. 5-step pipeline (add->mul->scale->abs->sum)
  7. Scale (scalar multiply)
  8. Abs

torch.cuda.synchronize() ensures accurate GPU timing.

Run: python examples/benchmark_pytorch_cuda.py
Compare: flowgpu-cli run examples/benchmark_octoflow_gpu.flow
"""
import time
import sys

try:
    import torch
except ImportError:
    print("ERROR: pip install torch --index-url https://download.pytorch.org/whl/cu124")
    sys.exit(1)

if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
print(f"=== PyTorch + CUDA Benchmark ===")
print(f"GPU: {gpu_name}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print()

N = 10_000_000

# ── Warmup ──
a_warm = torch.ones(N, device=device)
b_warm = torch.ones(N, device=device)
_ = torch.add(a_warm, b_warm)
torch.cuda.synchronize()
del a_warm, b_warm

# ── Test 1: gpu_fill equivalent ──
torch.cuda.synchronize()
t0 = time.perf_counter()
a = torch.full((N,), 1.5, device=device, dtype=torch.float32)
b = torch.full((N,), 2.5, device=device, dtype=torch.float32)
torch.cuda.synchronize()
t1 = time.perf_counter()
fill_ms = (t1 - t0) * 1000
print(f"gpu_fill 2x10M:  {fill_ms:.2f} ms")

# ── Test 2: Element-wise add ──
torch.cuda.synchronize()
t0 = time.perf_counter()
c_add = torch.add(a, b)
torch.cuda.synchronize()
t1 = time.perf_counter()
add_ms = (t1 - t0) * 1000
print(f"gpu_add 10M:     {add_ms:.2f} ms  [0]={c_add[0].item()}")

# ── Test 3: Element-wise mul ──
torch.cuda.synchronize()
t0 = time.perf_counter()
c_mul = torch.mul(a, b)
torch.cuda.synchronize()
t1 = time.perf_counter()
mul_ms = (t1 - t0) * 1000
print(f"gpu_mul 10M:     {mul_ms:.2f} ms  [0]={c_mul[0].item()}")

# ── Test 4: Scale (scalar multiply) ──
torch.cuda.synchronize()
t0 = time.perf_counter()
scaled = a * 2.0
torch.cuda.synchronize()
t1 = time.perf_counter()
scale_ms = (t1 - t0) * 1000
print(f"gpu_scale 10M:   {scale_ms:.2f} ms")

# ── Test 5: Abs ──
neg = torch.full((N,), -3.14, device=device, dtype=torch.float32)
torch.cuda.synchronize()
t0 = time.perf_counter()
ab = torch.abs(neg)
torch.cuda.synchronize()
t1 = time.perf_counter()
abs_ms = (t1 - t0) * 1000
print(f"gpu_abs 10M:     {abs_ms:.2f} ms")

# ── Test 6: Reduction sum ──
big = torch.full((N,), 3.0, device=device, dtype=torch.float32)
torch.cuda.synchronize()
t0 = time.perf_counter()
s = torch.sum(big)
torch.cuda.synchronize()
t1 = time.perf_counter()
sum_ms = (t1 - t0) * 1000
print(f"gpu_sum 10M:     {sum_ms:.2f} ms  result={s.item()}")

# ── Test 7: Matmul 256x256 ──
M3 = 256
a_m3 = torch.ones((M3, M3), device=device, dtype=torch.float32)
b_m3 = torch.ones((M3, M3), device=device, dtype=torch.float32)
_ = torch.matmul(a_m3, b_m3); torch.cuda.synchronize()
torch.cuda.synchronize()
t0 = time.perf_counter()
c_m3 = torch.matmul(a_m3, b_m3)
torch.cuda.synchronize()
t1 = time.perf_counter()
mat3_ms = (t1 - t0) * 1000
print(f"matmul 256:      {mat3_ms:.2f} ms  c[0,0]={c_m3[0,0].item()}")

# ── Test 8: Matmul 512x512 ──
M2 = 512
a_m2 = torch.ones((M2, M2), device=device, dtype=torch.float32)
b_m2 = torch.ones((M2, M2), device=device, dtype=torch.float32)
_ = torch.matmul(a_m2, b_m2); torch.cuda.synchronize()
torch.cuda.synchronize()
t0 = time.perf_counter()
c_m2 = torch.matmul(a_m2, b_m2)
torch.cuda.synchronize()
t1 = time.perf_counter()
mat2_ms = (t1 - t0) * 1000
print(f"matmul 512:      {mat2_ms:.2f} ms  c[0,0]={c_m2[0,0].item()}")

# ── Test 9: Matmul 1024x1024 ──
M1 = 1024
a_m1 = torch.ones((M1, M1), device=device, dtype=torch.float32)
b_m1 = torch.ones((M1, M1), device=device, dtype=torch.float32)
_ = torch.matmul(a_m1, b_m1); torch.cuda.synchronize()
torch.cuda.synchronize()
t0 = time.perf_counter()
c_m1 = torch.matmul(a_m1, b_m1)
torch.cuda.synchronize()
t1 = time.perf_counter()
mat1_ms = (t1 - t0) * 1000
print(f"matmul 1024:     {mat1_ms:.2f} ms  c[0,0]={c_m1[0,0].item()}")

# ── Test 10: 5-step pipeline (add->mul->scale->abs->sum) ──
a_pipe = torch.full((N,), 3.0, device=device, dtype=torch.float32)
b_pipe = torch.full((N,), 2.0, device=device, dtype=torch.float32)
c_pipe = torch.full((N,), 3.0, device=device, dtype=torch.float32)
_ = torch.sum(torch.abs(torch.mul(torch.add(a_pipe, b_pipe), c_pipe) * 0.5))
torch.cuda.synchronize()

torch.cuda.synchronize()
t0 = time.perf_counter()
step1 = torch.add(a_pipe, b_pipe)       # 3+2=5
step2 = torch.mul(step1, c_pipe)         # 5*3=15
step3 = step2 * 0.5                      # 15*0.5=7.5
step4 = torch.abs(step3)                 # 7.5
step5 = torch.sum(step4)                 # 7.5*10M=75M
torch.cuda.synchronize()
t1 = time.perf_counter()
pipe_ms = (t1 - t0) * 1000
print(f"5-step pipeline: {pipe_ms:.2f} ms  result={step5.item()}")

print()
print("--- done ---")
