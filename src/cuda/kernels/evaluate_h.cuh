#ifndef EVALUATE_H_KERNELS_CUH
#define EVALUATE_H_KERNELS_CUH

// ADDITION POINTER POINTER
__global__ void add_pp_kernel(fr_t* input1, fr_t* input2, fr_t* output,
                              const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1[tid];
    fr_t input2_local = input2[tid];

    input1_local = input1_local + input2_local;

    output[tid] = input1_local;
}

// ADDITION POINTER CONSTANT
__global__ void add_pc_kernel(fr_t* input1, fr_t input2, fr_t* output,
                              const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1[tid];
    fr_t input2_local = input2;

    input1_local = input1_local + input2_local;

    output[tid] = input1_local;
}

// ADDITION CONSTANT POINTER
__global__ void add_cp_kernel(fr_t input1, fr_t* input2, fr_t* output,
                              const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1;
    fr_t input2_local = input2[tid];

    input1_local = input1_local + input2_local;

    output[tid] = input1_local;
}

// ADDITION CONSTANT CONSTANT
__global__ void add_cc_kernel(fr_t input1, fr_t input2, fr_t* output, const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1;
    fr_t input2_local = input2;

    input1_local = input1_local + input2_local;

    output[tid] = input1_local;
}

// SUBTRACTION POINTER POINTER
__global__ void sub_pp_kernel(fr_t* input1, fr_t* input2, fr_t* output,
                              const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1[tid];
    fr_t input2_local = input2[tid];

    input1_local = input1_local - input2_local;

    output[tid] = input1_local;
}

// SUBTRACTION POINTER CONSTANT
__global__ void sub_pc_kernel(fr_t* input1, fr_t input2, fr_t* output,
                              const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1[tid];
    fr_t input2_local = input2;

    input1_local = input1_local - input2_local;

    output[tid] = input1_local;
}

// SUBTRACTION CONSTANT POINTER
__global__ void sub_cp_kernel(fr_t input1, fr_t* input2, fr_t* output,
                              const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1;
    fr_t input2_local = input2[tid];

    input1_local = input1_local - input2_local;

    output[tid] = input1_local;
}

// SUBTRACTION CONSTANT CONSTANT
__global__ void sub_cc_kernel(fr_t input1, fr_t input2, fr_t* output, const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1;
    fr_t input2_local = input2;

    input1_local = input1_local - input2_local;

    output[tid] = input1_local;
}

// MULTIPLICATION POINTER POINTER
__global__ void mul_pp_kernel(fr_t* input1, fr_t* input2, fr_t* output,
                              const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1[tid];
    fr_t input2_local = input2[tid];

    input1_local = input1_local * input2_local;

    output[tid] = input1_local;
}

// MULTIPLICATION POINTER CONSTANT
__global__ void mul_pc_kernel(fr_t* input1, fr_t input2, fr_t* output,
                              const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1[tid];
    fr_t input2_local = input2;

    input1_local = input1_local * input2_local;

    output[tid] = input1_local;
}

// MULTIPLICATION CONSTANT POINTER
__global__ void mul_cp_kernel(fr_t input1, fr_t* input2, fr_t* output,
                              const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1;
    fr_t input2_local = input2[tid];

    input1_local = input1_local * input2_local;

    output[tid] = input1_local;
}

// MULTIPLICATION CONSTANT CONSTANT
__global__ void mul_cc_kernel(fr_t input1, fr_t input2, fr_t* output, const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1;
    fr_t input2_local = input2;

    input1_local = input1_local * input2_local;

    output[tid] = input1_local;
}

// SQUARE POINTER
__global__ void square_p_kernel(fr_t* input1, fr_t* output, const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1[tid];

    input1_local = input1_local * input1_local;

    output[tid] = input1_local;
}

// SQUARE CONSTANT
__global__ void square_c_kernel(fr_t input1, fr_t* output, const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1;

    input1_local = input1_local * input1_local;

    output[tid] = input1_local;
}

// DOUBLE POINTER
__global__ void double_p_kernel(fr_t* input1, fr_t* output, const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1[tid];

    input1_local = input1_local + input1_local;

    output[tid] = input1_local;
}

// DOUBLE CONSTANT
__global__ void double_c_kernel(fr_t input1, fr_t* output, const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1;

    input1_local = input1_local + input1_local;

    output[tid] = input1_local;
}

// NEGATE POINTER
__global__ void negate_p_kernel(fr_t* input1, fr_t* output, const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1[tid];

    input1_local = -input1_local;

    output[tid] = input1_local;
}

// NEGATE CONSTANT
__global__ void negate_c_kernel(fr_t input1, fr_t* output, const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1;

    input1_local = -input1_local;

    output[tid] = input1_local;
}

__global__ void horner_kernel(fr_t* input1, fr_t* input2, fr_t* output, const fr_t y,
                              const size_t* horner_index, size_t horner_index_size,
                              const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t y_local = y;

    fr_t value = input1[tid];

    for (int i = 0; i < horner_index_size; i++) {
        size_t horner_index_i = horner_index[i];
        horner_index_i = horner_index_i * size;

        fr_t part = input2[tid + horner_index_i];

        value = value * y_local;
        value = value + part;
    }

    output[tid] = value;
}

__global__ void horner_c_kernel(fr_t input1, fr_t* input2, fr_t* output, const fr_t y,
                                const size_t* horner_index, size_t horner_index_size,
                                const size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t y_local = y;

    fr_t value = input1;

    for (int i = 0; i < horner_index_size; i++) {
        size_t horner_index_i = horner_index[i];
        horner_index_i = horner_index_i * size;

        fr_t part = input2[tid + horner_index_i];

        value = value * y_local;
        value = value + part;
    }

    output[tid] = value;
}

// isize should be power of 2!
__device__ inline std::size_t get_rotation_idx(std::size_t idx, int rot, int rot_scale,
                                               int isize) {
    int a = static_cast<int>(idx) + rot * rot_scale;

    unsigned ua = static_cast<unsigned>(a);
    return static_cast<std::size_t>(ua & (isize - 1));
}

__global__ void store_kernel(fr_t* input1, fr_t* output, const int rot,
                             const int rot_scale, const int isize, const size_t size,
                             const int offset) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    size_t index = tid + offset;
    size_t rotate_index = get_rotation_idx(index, rot, rot_scale, isize);

    fr_t rotate_input = input1[rotate_index];

    output[tid] = rotate_input;
}

#endif // EVALUATE_H_KERNELS_CUH