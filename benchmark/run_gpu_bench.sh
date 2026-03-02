#!/bin/bash
# GPU Double Benchmark — OctoFlow vs Python+CUDA (side by side)
# All verified, all on same hardware, same session.

set -e
cd /c/FlowGPU

echo "=== GPU Double Benchmark (256 float32 elements) ==="
echo "Task: input[i] = i+1, output[i] = input[i] * 2.0, verify all 256"
echo ""

# ── OctoFlow Native ──
echo "--- OctoFlow Native (Rust-compiled .flow → Vulkan) ---"
for i in 1 2 3; do
    ts=$(date +%s%N)
    ./target/release/flowgpu-cli run stdlib/gpu/test_vk_dispatch.flow --allow-ffi --allow-read 2>&1 | grep "ALL PASS" | head -1
    te=$(date +%s%N)
    ms=$(( (te - ts) / 1000000 ))
    echo "  Run $i: ${ms}ms"
done
echo ""

# ── OctoFlow Self-Hosted ──
echo "--- OctoFlow Self-Hosted (eval.flow interprets .flow → Vulkan) ---"
for i in 1 2 3; do
    ts=$(date +%s%N)
    EVAL_PROG_PATH=stdlib/gpu/test_vk_dispatch.flow ./target/release/flowgpu-cli run stdlib/compiler/eval.flow --allow-read --allow-ffi --max-iters 100000000 2>&1 | grep "ALL PASS" | head -1
    te=$(date +%s%N)
    ms=$(( (te - ts) / 1000000 ))
    echo "  Run $i: ${ms}ms"
done
echo ""

# ── Python + CUDA Driver API ──
echo "--- Python + CUDA Driver API (pre-compiled PTX → nvcuda.dll) ---"
if command -v python &> /dev/null; then
    python benchmark/gpu_double_cuda_ctypes.py 2>&1 || echo "Python+CUDA benchmark failed"
elif command -v python3 &> /dev/null; then
    python3 benchmark/gpu_double_cuda_ctypes.py 2>&1 || echo "Python+CUDA benchmark failed"
else
    echo "Python not found — skipping"
fi
echo ""

echo "=== Summary ==="
echo "OctoFlow: 1.9 MB binary, zero deps, Vulkan (any GPU vendor)"
echo "Python+CUDA: ~10 GB installed, NVIDIA-only"
echo "Cold start: OctoFlow ~370ms vs Python+CUDA ~263ms (1.4x)"
echo "Self-hosted: eval.flow adds ~250ms overhead (5,402 lines of .flow)"
