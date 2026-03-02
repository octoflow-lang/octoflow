extern "C" __global__ void double_it(float *in_arr, float *out_arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out_arr[idx] = in_arr[idx] * 2.0f;
    }
}
