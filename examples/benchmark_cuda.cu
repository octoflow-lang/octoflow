// OctoFlow vs CUDA — Raw GPU-to-GPU Benchmark
// Compile: nvcc -O2 -o benchmark_cuda.exe benchmark_cuda.cu
// Same operations as benchmark_octoflow_gpu.flow

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000000
#define BLOCK_SIZE 256

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ── Kernels ──

__global__ void k_fill(float* out, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = val;
}

__global__ void k_add(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void k_mul(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void k_scale(float* out, const float* a, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * s;
}

__global__ void k_abs(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fabsf(a[i]);
}

// Simple two-pass reduction
__global__ void k_reduce_sum(float* out, const float* a, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? a[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// Naive matmul (tiled would be faster but this matches OctoFlow's naive kernel)
__global__ void k_matmul(float* C, const float* A, const float* B, int M, int K, int N_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N_cols) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N_cols + col];
        }
        C[row * N_cols + col] = sum;
    }
}

float gpu_sum(float* d_in, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* d_partial;
    CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));
    k_reduce_sum<<<blocks, BLOCK_SIZE>>>(d_partial, d_in, n);

    // Second pass if needed
    while (blocks > 1) {
        int new_blocks = (blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float* d_partial2;
        CHECK(cudaMalloc(&d_partial2, new_blocks * sizeof(float)));
        k_reduce_sum<<<new_blocks, BLOCK_SIZE>>>(d_partial2, d_partial, blocks);
        cudaFree(d_partial);
        d_partial = d_partial2;
        blocks = new_blocks;
    }

    float result;
    CHECK(cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_partial);
    return result;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== CUDA Benchmark ===\n");
    printf("GPU: %s\n", prop.name);

    int driver_ver, runtime_ver;
    cudaDriverGetVersion(&driver_ver);
    cudaRuntimeGetVersion(&runtime_ver);
    printf("CUDA: driver %d.%d, runtime %d.%d\n",
           driver_ver/1000, (driver_ver%100)/10,
           runtime_ver/1000, (runtime_ver%100)/10);
    printf("\n");

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_a, *d_b, *d_c, *d_d, *d_e;
    CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CHECK(cudaMalloc(&d_d, N * sizeof(float)));
    CHECK(cudaMalloc(&d_e, N * sizeof(float)));

    // ── Warmup ──
    k_fill<<<blocks, BLOCK_SIZE>>>(d_a, 1.0f, N);
    k_fill<<<blocks, BLOCK_SIZE>>>(d_b, 1.0f, N);
    k_add<<<blocks, BLOCK_SIZE>>>(d_c, d_a, d_b, N);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    float ms;

    // ── Test 1: gpu_fill 2x10M ──
    CHECK(cudaEventRecord(start));
    k_fill<<<blocks, BLOCK_SIZE>>>(d_a, 1.5f, N);
    k_fill<<<blocks, BLOCK_SIZE>>>(d_b, 2.5f, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("gpu_fill 2x10M:  %.2f ms\n", ms);

    // ── Test 2: Element-wise add ──
    CHECK(cudaEventRecord(start));
    k_add<<<blocks, BLOCK_SIZE>>>(d_c, d_a, d_b, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    float v0;
    CHECK(cudaMemcpy(&v0, d_c, sizeof(float), cudaMemcpyDeviceToHost));
    printf("gpu_add 10M:     %.2f ms  [0]=%.1f\n", ms, v0);

    // ── Test 3: Element-wise mul ──
    CHECK(cudaEventRecord(start));
    k_mul<<<blocks, BLOCK_SIZE>>>(d_c, d_a, d_b, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaMemcpy(&v0, d_c, sizeof(float), cudaMemcpyDeviceToHost));
    printf("gpu_mul 10M:     %.2f ms  [0]=%.1f\n", ms, v0);

    // ── Test 4: Scale ──
    CHECK(cudaEventRecord(start));
    k_scale<<<blocks, BLOCK_SIZE>>>(d_c, d_a, 2.0f, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("gpu_scale 10M:   %.2f ms\n", ms);

    // ── Test 5: Abs ──
    k_fill<<<blocks, BLOCK_SIZE>>>(d_d, -3.14f, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start));
    k_abs<<<blocks, BLOCK_SIZE>>>(d_c, d_d, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("gpu_abs 10M:     %.2f ms\n", ms);

    // ── Test 6: Reduction sum ──
    k_fill<<<blocks, BLOCK_SIZE>>>(d_d, 3.0f, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start));
    float sum_val = gpu_sum(d_d, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("gpu_sum 10M:     %.2f ms  result=%.0f\n", ms, sum_val);

    // ── Test 7-9: Matmul ──
    int mat_sizes[] = {256, 512, 1024};
    for (int mi = 0; mi < 3; mi++) {
        int M = mat_sizes[mi];
        int sz = M * M;
        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc(&d_A, sz * sizeof(float)));
        CHECK(cudaMalloc(&d_B, sz * sizeof(float)));
        CHECK(cudaMalloc(&d_C, sz * sizeof(float)));

        int mblocks = (sz + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_fill<<<mblocks, BLOCK_SIZE>>>(d_A, 1.0f, sz);
        k_fill<<<mblocks, BLOCK_SIZE>>>(d_B, 1.0f, sz);
        CHECK(cudaDeviceSynchronize());

        // Warmup
        dim3 block2d(16, 16);
        dim3 grid2d((M + 15) / 16, (M + 15) / 16);
        k_matmul<<<grid2d, block2d>>>(d_C, d_A, d_B, M, M, M);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));
        k_matmul<<<grid2d, block2d>>>(d_C, d_A, d_B, M, M, M);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));

        float cv;
        CHECK(cudaMemcpy(&cv, d_C, sizeof(float), cudaMemcpyDeviceToHost));
        printf("matmul %-4d:     %.2f ms  c[0,0]=%.0f\n", M, ms, cv);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // ── Test 10: 5-step pipeline ──
    float *d_p1, *d_p2, *d_p3;
    CHECK(cudaMalloc(&d_p1, N * sizeof(float)));
    CHECK(cudaMalloc(&d_p2, N * sizeof(float)));
    CHECK(cudaMalloc(&d_p3, N * sizeof(float)));

    k_fill<<<blocks, BLOCK_SIZE>>>(d_a, 3.0f, N);  // a = 3
    k_fill<<<blocks, BLOCK_SIZE>>>(d_b, 2.0f, N);  // b = 2
    k_fill<<<blocks, BLOCK_SIZE>>>(d_d, 3.0f, N);  // c_pipe = 3
    CHECK(cudaDeviceSynchronize());

    // Warmup pipeline
    k_add<<<blocks, BLOCK_SIZE>>>(d_p1, d_a, d_b, N);
    k_mul<<<blocks, BLOCK_SIZE>>>(d_p2, d_p1, d_d, N);
    k_scale<<<blocks, BLOCK_SIZE>>>(d_p3, d_p2, 0.5f, N);
    k_abs<<<blocks, BLOCK_SIZE>>>(d_p1, d_p3, N);
    float warmup_sum = gpu_sum(d_p1, N);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    k_add<<<blocks, BLOCK_SIZE>>>(d_p1, d_a, d_b, N);       // 3+2=5
    k_mul<<<blocks, BLOCK_SIZE>>>(d_p2, d_p1, d_d, N);       // 5*3=15
    k_scale<<<blocks, BLOCK_SIZE>>>(d_p3, d_p2, 0.5f, N);    // 15*0.5=7.5
    k_abs<<<blocks, BLOCK_SIZE>>>(d_p1, d_p3, N);             // 7.5
    float pipe_result = gpu_sum(d_p1, N);                      // 75M
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("5-step pipeline: %.2f ms  result=%.0f\n", ms, pipe_result);

    printf("\n--- done ---\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d); cudaFree(d_e);
    cudaFree(d_p1); cudaFree(d_p2); cudaFree(d_p3);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
