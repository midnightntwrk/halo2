#ifndef RAND_MSM_HELPER_KERNELS_CUH
#define RAND_MSM_HELPER_KERNELS_CUH

__global__ void scalar_subtraction_kernel(scalar_t* __restrict__ actual_scalars,
                                          const scalar_t* __restrict__ random_scalars,
                                          size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    if (tid >= size) return;

    scalar_t a_input1 = actual_scalars[tid];

    scalar_t r_input1 = random_scalars[tid];

    a_input1 = a_input1 - r_input1;

    actual_scalars[tid] = a_input1;
}

__global__ void random_scalars_kernel(uint32_t* output, size_t N, unsigned long seed) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    curandState state;
    curand_init(seed, tid, 0, &state);

    for (size_t i = tid; i < N; i += stride) {
        output[i] = curand(&state);
    }
}

__global__ void montgomery_conv(fr_t* __restrict__ inout, size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    if (tid >= size) return;

    fr_t input1 = inout[tid];

    input1.to();

    inout[tid] = input1;
}

#endif // RAND_MSM_HELPER_KERNELS_CUH