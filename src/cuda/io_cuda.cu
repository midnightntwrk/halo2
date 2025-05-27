#include <cuda.h>

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381.hpp>
#else
# error "No FEATURE! It has to be BLS12-381."
#endif

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <ntt/ntt.cuh>
#include <msm/pippenger.cuh>

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <iostream>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

extern "C"
RustError::by_value sppark_ntt(size_t device_id,
                                fr_t* inout, uint32_t lg_domain_size,
                                NTT::InputOutputOrder ntt_order,
                                NTT::Direction ntt_direction,
                                NTT::Type ntt_type)
{
    auto& gpu = select_gpu(device_id);
    return NTT::Base(gpu, inout, lg_domain_size, ntt_order, ntt_direction, ntt_type);
}

__global__ void scalar_subtraction_kernel(scalar_t* __restrict__ actual_scalars, const scalar_t* __restrict__ random_scalars, size_t size)
{
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    if (tid >= size)
        return;

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

__global__ void montgomery_conv(fr_t* __restrict__ inout, size_t size)
{
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    if (tid >= size)
        return;

    fr_t input1 = inout[tid];

    input1.to();
  
    inout[tid] = input1;
}

class RandomMSM {
public:
    enum class Group : int { G = 0, G_LAGRANGE = 1 };

    static RandomMSM& get_instance() {
        static RandomMSM instance;
        return instance;
    }

    scalar_t* get_randoms(const affine_t points[],
                          size_t npoints,
                          size_t ffi_affine_sz,
                          const gpu_t& gpu_s,
                          Group group)
    {
        auto& data = groups_[static_cast<int>(group)];

        if (!data.initialized || data.size != npoints) {
            if (data.random_scalars) {
                cudaFree(data.random_scalars);
                cudaDeviceSynchronize();
            }

            data.size = npoints;
            cudaMallocAsync(&data.random_scalars,
                            data.size * sizeof(scalar_t),
                            gpu_s);

            uint32_t loop = sizeof(scalar_t) / sizeof(uint32_t);
            uint64_t size_loop = data.size * loop;
            int num_sms = gpu_s.sm_count();
            const int threads_per_block = 512;
            int needed_blocks = int((size_loop + threads_per_block - 1) / threads_per_block);
            int blocks = std::min(needed_blocks, num_sms);

            uint32_t* raw_ptr = reinterpret_cast<uint32_t*>(data.random_scalars);
            random_scalars_kernel<<<blocks, threads_per_block, 0, gpu_s>>>(
                raw_ptr, size_loop, static_cast<uint32_t>(time(nullptr)));
            CUDA_OK(cudaGetLastError());

            size_t grid_size = (data.size + threads_per_block - 1) / threads_per_block;
            montgomery_conv<<<grid_size, threads_per_block, 0, gpu_s>>>(
                data.random_scalars, data.size);
            CUDA_OK(cudaGetLastError());

            msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
            msm.invoke(data.random_point,
                       points,
                       npoints,
                       data.random_scalars,
                       true,
                       ffi_affine_sz);

            data.initialized = true;
        }

        return data.random_scalars;
    }

    point_t get_point(Group group) const {
        const auto& data = groups_[static_cast<int>(group)];
        if (!data.initialized)
            throw std::runtime_error("Group " + std::to_string(int(group))
                                     + " not initialized yet!");
        return data.random_point;
    }

    ~RandomMSM() {
        for (auto& data : groups_) {
            if (data.random_scalars)
                cudaFree(data.random_scalars);
        }
    }

    RandomMSM(const RandomMSM&) = delete;
    RandomMSM& operator=(const RandomMSM&) = delete;

private:
    RandomMSM() = default;

    struct GroupData {
        point_t   random_point;
        scalar_t* random_scalars = nullptr;
        uint64_t  size           = 0;
        bool      initialized    = false;
    };

    GroupData groups_[2];
};

extern "C"
RustError::by_value gpu_msm(point_t* out1, point_t* out2, const affine_t points[],
                                size_t npoints, const scalar_t scalars[],
                                size_t ffi_affine_sz)
{   
    
    const gpu_t& gpu = select_gpu();

    try {
        gpu_ptr_t<fr_t> d_scalars((scalar_t*)gpu.Dmalloc(npoints * sizeof(scalar_t)));
        scalar_t* d_scalars_pointer = &d_scalars[0];
        gpu.HtoD(d_scalars_pointer, scalars, npoints);

        scalar_t* d_rand_scalars_pointer = RandomMSM::get_instance().get_randoms(points, npoints, ffi_affine_sz, gpu, RandomMSM::Group::G);

        const int threads_per_block = 512;
        size_t grid_size = (int)((npoints + threads_per_block - 1) / threads_per_block);

        scalar_subtraction_kernel<<< grid_size, threads_per_block, 0, gpu>>>(
            d_scalars_pointer, d_rand_scalars_pointer, npoints);

        *out2 = RandomMSM::get_instance().get_point(RandomMSM::Group::G);  

        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        msm.invoke(*out1, points, npoints, d_scalars_pointer, true, ffi_affine_sz);
        CUDA_OK(cudaGetLastError());

        
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

extern "C"
RustError::by_value gpu_msm_lagrange(point_t* out1, point_t* out2, const affine_t points[],
                                size_t npoints, const scalar_t scalars[],
                                size_t ffi_affine_sz)
{   
    
    const gpu_t& gpu = select_gpu();

    try {
        gpu_ptr_t<fr_t> d_scalars((scalar_t*)gpu.Dmalloc(npoints * sizeof(scalar_t)));
        scalar_t* d_scalars_pointer = &d_scalars[0];
        gpu.HtoD(d_scalars_pointer, scalars, npoints);

        scalar_t* d_rand_scalars_pointer = RandomMSM::get_instance().get_randoms(points, npoints, ffi_affine_sz, gpu, RandomMSM::Group::G_LAGRANGE);

        const int threads_per_block = 512;
        size_t grid_size = (int)((npoints + threads_per_block - 1) / threads_per_block);

        scalar_subtraction_kernel<<< grid_size, threads_per_block, 0, gpu>>>(
            d_scalars_pointer, d_rand_scalars_pointer, npoints);

        *out2 = RandomMSM::get_instance().get_point(RandomMSM::Group::G_LAGRANGE);  

        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        msm.invoke(*out1, points, npoints, d_scalars_pointer, true, ffi_affine_sz);
        CUDA_OK(cudaGetLastError());

        
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}
