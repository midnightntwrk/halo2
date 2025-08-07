#include <cuda.h>

#if defined(FEATURE_BLS12_381)
#include <ff/bls12-381.hpp>
#else
#error "No FEATURE! It has to be BLS12-381."
#endif

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>
#include <cstring>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <iostream>
#include <memory>
#include <msm/pippenger.cuh>
#include <ntt/ntt.cuh>
#include <vector>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include "kernels/evaluate_h.cuh"
#include "kernels/rand_msm_helper.cuh"

extern "C" RustError::by_value sppark_ntt(size_t device_id, fr_t* inout,
                                          uint32_t lg_domain_size,
                                          NTT::InputOutputOrder ntt_order,
                                          NTT::Direction ntt_direction,
                                          NTT::Type ntt_type) {
    auto& gpu = select_gpu(device_id);
    return NTT::Base(gpu, inout, lg_domain_size, ntt_order, ntt_direction, ntt_type);
}

class RandomMSM {
   public:
    enum class Group : int { G = 0, G_LAGRANGE = 1 };

    static RandomMSM& get_instance() {
        static thread_local RandomMSM instance;
        return instance;
    }

    scalar_t* get_randoms(const affine_t points[], size_t npoints, size_t ffi_affine_sz,
                          const gpu_t& gpu_s, Group group) {
        auto& data = groups_[static_cast<int>(group)];

        if (!data.initialized || data.size != npoints) {
            if (data.random_scalars) {
                cudaFree(data.random_scalars);
                cudaDeviceSynchronize();
            }

            data.size = npoints;
            cudaMallocAsync(&data.random_scalars, data.size * sizeof(scalar_t), gpu_s);

            uint32_t loop = sizeof(scalar_t) / sizeof(uint32_t);
            uint64_t size_loop = data.size * loop;
            int num_sms = gpu_s.sm_count();
            const int threads_per_block = 512;
            int needed_blocks =
                int((size_loop + threads_per_block - 1) / threads_per_block);
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
            msm.invoke(data.random_point, points, npoints, data.random_scalars, true,
                       ffi_affine_sz);

            data.initialized = true;
        }

        return data.random_scalars;
    }

    point_t get_point(Group group) const {
        const auto& data = groups_[static_cast<int>(group)];
        if (!data.initialized)
            throw std::runtime_error("Group " + std::to_string(int(group)) +
                                     " not initialized yet!");
        return data.random_point;
    }

    ~RandomMSM() {
        for (auto& data : groups_) {
            if (data.random_scalars) cudaFree(data.random_scalars);
        }
    }

    RandomMSM(const RandomMSM&) = delete;
    RandomMSM& operator=(const RandomMSM&) = delete;

   private:
    RandomMSM() = default;

    struct GroupData {
        point_t random_point;
        scalar_t* random_scalars = nullptr;
        uint64_t size = 0;
        bool initialized = false;
    };

    GroupData groups_[2];
};

extern "C" RustError::by_value gpu_msm(point_t* out1, point_t* out2,
                                       const affine_t points[], size_t npoints,
                                       const scalar_t scalars[], size_t ffi_affine_sz) {
    const gpu_t& gpu = select_gpu();

    try {
        gpu_ptr_t<fr_t> d_scalars((scalar_t*)gpu.Dmalloc(npoints * sizeof(scalar_t)));
        scalar_t* d_scalars_pointer = &d_scalars[0];
        gpu.HtoD(d_scalars_pointer, scalars, npoints);

        scalar_t* d_rand_scalars_pointer = RandomMSM::get_instance().get_randoms(
            points, npoints, ffi_affine_sz, gpu, RandomMSM::Group::G);

        const int threads_per_block = 512;
        size_t grid_size = (int)((npoints + threads_per_block - 1) / threads_per_block);

        scalar_subtraction_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
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

extern "C" RustError::by_value gpu_msm_lagrange(point_t* out1, point_t* out2,
                                                const affine_t points[], size_t npoints,
                                                const scalar_t scalars[],
                                                size_t ffi_affine_sz) {
    const gpu_t& gpu = select_gpu();

    try {
        gpu_ptr_t<fr_t> d_scalars((scalar_t*)gpu.Dmalloc(npoints * sizeof(scalar_t)));
        scalar_t* d_scalars_pointer = &d_scalars[0];
        gpu.HtoD(d_scalars_pointer, scalars, npoints);

        scalar_t* d_rand_scalars_pointer = RandomMSM::get_instance().get_randoms(
            points, npoints, ffi_affine_sz, gpu, RandomMSM::Group::G_LAGRANGE);

        const int threads_per_block = 512;
        size_t grid_size = (int)((npoints + threads_per_block - 1) / threads_per_block);

        scalar_subtraction_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
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

///////////////////////////////////////////////////////////////
//                     !!! TEST FIELD !!!
///////////////////////////////////////////////////////////////

extern "C" {

enum class ValueSourceKind : uint8_t {
    Constant,
    Intermediate,
    Fixed,
    Advice,
    Instance,
    Challenge,
    Beta,
    Gamma,
    Theta,
    Y,
    PreviousValue
};

struct ValueSourceFFI {
    ValueSourceKind kind;
    size_t param0;
    size_t param1;
};

enum class CalculationKind : uint8_t {
    Add,
    Sub,
    Mul,
    Square,
    Double,
    Negate,
    Store,
    Horner
};

struct CalculationFFI {
    CalculationKind kind;
    ValueSourceFFI a;
    ValueSourceFFI b;
    ValueSourceFFI extra;
    const ValueSourceFFI* horner_parts_ptr;
    size_t horner_parts_len;
};

struct CalculationInfoFFI {
    CalculationFFI calculation;
    size_t target;
};

struct ResolvedInput {
    fr_t* pointer = nullptr;
    fr_t constant = fr_t{};
    bool is_constant = false;
};

ResolvedInput get_resolve_input(const ValueSourceFFI& src, fr_t constants,
                                fr_t* intermediates, fr_t* fixed, fr_t* advice,
                                fr_t* instance, fr_t challenges, fr_t beta, fr_t gamma,
                                fr_t theta, fr_t y, fr_t* prev, size_t chunk_offset,
                                size_t poly_offset) {
    switch (src.kind) {
        case ValueSourceKind::Constant: {
            return {nullptr, constants, true};
        }
        case ValueSourceKind::Intermediate: {
            return {intermediates + (src.param0 * chunk_offset), fr_t{}, false};
        }
        case ValueSourceKind::Fixed: {
            return {fixed + (src.param0 * poly_offset), fr_t{}, false};
        }
        case ValueSourceKind::Advice: {
            return {advice + (src.param0 * poly_offset), fr_t{}, false};
        }
        case ValueSourceKind::Instance: {
            return {instance + (src.param0 * poly_offset), fr_t{}, false};
        }
        case ValueSourceKind::Challenge: {
            return {nullptr, challenges, true};
        }
        case ValueSourceKind::Beta: {
            return {nullptr, beta, true};
        }
        case ValueSourceKind::Gamma: {
            return {nullptr, gamma, true};
        }
        case ValueSourceKind::Theta: {
            return {nullptr, theta, true};
        }
        case ValueSourceKind::Y: {
            return {nullptr, y, true};
        }
        case ValueSourceKind::PreviousValue: {
            return {prev, fr_t{}, false};
        }
        default:
            throw std::invalid_argument("Unknown ValueSourceKind");
    }
}

} // extern "C"

__global__ void zero_padding_kernel(fr_t* input, fr_t* output, const fr_t g_coset,
                                    const fr_t g_coset_inv, int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < polysize) {
        fr_t input_r = input[tid];
        int index = tid % 3;
        if (index == 1) {
            input_r = input_r * g_coset;
        } else if (index == 2) {
            input_r = input_r * g_coset_inv;
        } else {
        }
        output[tid] = input_r;
    } else {
        fr_t value;
        value.zero();
        output[tid] = value;
    }
}

class MemoryPool {
   public:
    static MemoryPool& get_instance() {
        static thread_local MemoryPool instance;
        return instance;
    }

    void store_gpu(const gpu_t& gpu, const fr_t* const* fixed_ptrs, size_t fixed_ptr_len,
                   size_t advice_ptr_len, size_t instance_ptr_len, const fr_t* l0_ptr,
                   const fr_t* l_last_ptr, const fr_t* l_active_row_ptr,
                   const fr_t* value_ptr, int isize, bool flag) {
        if (flag) {
            // if (all_data) {
            //     cudaFreeAsync(all_data, gpu);
            //     cudaStreamSynchronize(gpu);
            // }

            size_t total_poly_size =
                ((fixed_ptr_len + advice_ptr_len + instance_ptr_len) * isize) +
                (4 * isize);
            cudaMallocAsync(&all_data, total_poly_size * sizeof(fr_t), gpu);

            CUDA_OK(cudaGetLastError());
            fixed_device_ptrs = all_data;
            advice_device_ptrs = fixed_device_ptrs + (fixed_ptr_len * isize);
            instance_device_ptrs = advice_device_ptrs + (advice_ptr_len * isize);
            l0_device_ptr = instance_device_ptrs + (instance_ptr_len * isize);
            l_last_device_ptr = l0_device_ptr + isize;
            l_active_row_device_ptr = l_last_device_ptr + isize;
            value_device_ptr = l_active_row_device_ptr + isize;

            for (int i = 0; i < fixed_ptr_len; i++) {
                gpu.HtoD(fixed_device_ptrs + (i * isize), fixed_ptrs[i], isize);
                CUDA_OK(cudaGetLastError());
            }
            gpu.sync();

            gpu.HtoD(l0_device_ptr, l0_ptr, isize);
            CUDA_OK(cudaGetLastError());
            gpu.sync();

            gpu.HtoD(l_last_device_ptr, l_last_ptr, isize);
            CUDA_OK(cudaGetLastError());
            gpu.sync();

            gpu.HtoD(l_active_row_device_ptr, l_active_row_ptr, isize);
            CUDA_OK(cudaGetLastError());
            gpu.sync();

            gpu.HtoD(value_device_ptr, value_ptr, isize);
            CUDA_OK(cudaGetLastError());
            gpu.sync();

            initialized = true;
        }
    }

    ~MemoryPool() {
        if (all_data) cudaFree(all_data);
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    fr_t* fix_ptr() {
        if (initialized) {
            return fixed_device_ptrs;
        } else {
            throw std::invalid_argument("Memory Pool was not created yet!");
        }
    }

    fr_t* advice_ptr() {
        if (initialized) {
            return advice_device_ptrs;
        } else {
            throw std::invalid_argument("Memory Pool was not created yet!");
        }
    }

    fr_t* instance_ptr() {
        if (initialized) {
            return instance_device_ptrs;
        } else {
            throw std::invalid_argument("Memory Pool was not created yet!");
        }
    }

    fr_t* l0_ptr() {
        if (initialized) {
            return l0_device_ptr;
        } else {
            throw std::invalid_argument("Memory Pool was not created yet!");
        }
    }

    fr_t* l_last_ptr() {
        if (initialized) {
            return l_last_device_ptr;
        } else {
            throw std::invalid_argument("Memory Pool was not created yet!");
        }
    }

    fr_t* l_active_row_ptr() {
        if (initialized) {
            return l_active_row_device_ptr;
        } else {
            throw std::invalid_argument("Memory Pool was not created yet!");
        }
    }

    fr_t* value_ptr() {
        if (initialized) {
            return value_device_ptr;
        } else {
            throw std::invalid_argument("Memory Pool was not created yet!");
        }
    }

   private:
    MemoryPool() = default;

    scalar_t* all_data = nullptr;
    bool initialized = false;

    fr_t* fixed_device_ptrs = nullptr;
    fr_t* advice_device_ptrs = nullptr;
    fr_t* instance_device_ptrs = nullptr;
    fr_t* l0_device_ptr = nullptr;
    fr_t* l_last_device_ptr = nullptr;
    fr_t* l_active_row_device_ptr = nullptr;
    fr_t* value_device_ptr = nullptr;
};

extern "C" RustError::by_value custom_gates_evaluation(
    const CalculationInfoFFI* calculations, size_t calculations_count,
    const fr_t* const* fixed_ptrs, size_t fixed_ptr_len, const fr_t* const* advice_ptrs,
    size_t advice_ptr_len, const fr_t* const* instance_ptrs, size_t instance_ptr_len,
    const fr_t* challenges, size_t challenges_ptr_len, const fr_t* beta,
    const fr_t* gamma, const fr_t* theta, const fr_t* y, fr_t* output,
    const fr_t* constants, size_t constants_ptr_len, int* rotation_value,
    size_t rotation_ptr_len, int rot_scale, int isize,

    const fr_t* l0_ptr, const fr_t* l_last_ptr, const fr_t* l_active_row_ptr,

    const fr_t* g_coset, const fr_t* g_coset_inv, int small_size, const int flag) {
    constexpr size_t CHUNK_SIZE = 1 << 18; // To minimize memory usage
    const gpu_t& gpu = select_gpu();

    size_t csize = isize;
    size_t num_parts = 1;
    if (isize > CHUNK_SIZE) {
        csize = CHUNK_SIZE;
        num_parts = isize / CHUNK_SIZE;
    }

    const int threads_per_block = 512;
    size_t grid_size = (int)((csize + threads_per_block - 1) / threads_per_block);

    bool flag_c = flag == 1 ? true : false;

    try {
        MemoryPool::get_instance().store_gpu(
            gpu, fixed_ptrs, fixed_ptr_len, advice_ptr_len, instance_ptr_len, l0_ptr,
            l_last_ptr, l_active_row_ptr, output, isize, true);

        fr_t* fixed_device_ptrs = MemoryPool::get_instance().fix_ptr();
        fr_t* advice_device_ptrs = MemoryPool::get_instance().advice_ptr();
        fr_t* instance_device_ptrs = MemoryPool::get_instance().instance_ptr();
        fr_t* prev_device_ptrs = MemoryPool::get_instance().value_ptr();

        size_t total_poly_size = (advice_ptr_len + instance_ptr_len) * small_size;
        gpu_ptr_t<fr_t> input_polys((fr_t*)gpu.Dmalloc(total_poly_size * sizeof(fr_t)));
        CUDA_OK(cudaGetLastError());
        fr_t* before_ntt_advice_device_ptrs = &input_polys[0];
        fr_t* before_ntt_instance_device_ptrs =
            before_ntt_advice_device_ptrs + (advice_ptr_len * small_size);

        for (int i = 0; i < advice_ptr_len; i++) {
            gpu.HtoD(before_ntt_advice_device_ptrs + (i * small_size), advice_ptrs[i],
                     small_size);
            CUDA_OK(cudaGetLastError());
        }
        gpu.sync();

        for (int i = 0; i < instance_ptr_len; i++) {
            gpu.HtoD(before_ntt_instance_device_ptrs + (i * small_size), instance_ptrs[i],
                     small_size);
            CUDA_OK(cudaGetLastError());
        }
        gpu.sync();

        size_t total_intermediate_size = calculations_count * csize;
        gpu_ptr_t<fr_t> intermediate_values(
            (fr_t*)gpu.Dmalloc(total_intermediate_size * sizeof(fr_t)));
        CUDA_OK(cudaGetLastError());
        fr_t* intermediate_device_ptrs = &intermediate_values[0];

        std::vector<size_t> horner_index;
        const auto& calculations_in = calculations[calculations_count - 1].calculation;
        for (size_t i = 0; i < calculations_in.horner_parts_len; i++) {
            const auto& part = calculations_in.horner_parts_ptr[i];
            horner_index.push_back(part.param0);
        }

        gpu_ptr_t<size_t> horner_index_values(
            (size_t*)gpu.Dmalloc(horner_index.size() * sizeof(size_t)));
        size_t* horner_index_device_ptrs = &horner_index_values[0];
        gpu.HtoD(horner_index_device_ptrs, horner_index.data(), horner_index.size());
        CUDA_OK(cudaGetLastError());
        gpu.sync();

        uint32_t log_isize = uint32_t(log2(isize));
        size_t grid_size_pad = (int)((isize + threads_per_block - 1) / threads_per_block);

        for (int i = 0; i < advice_ptr_len; i++) {
            zero_padding_kernel<<<grid_size_pad, threads_per_block, 0, gpu>>>(
                before_ntt_advice_device_ptrs + (i * small_size),
                advice_device_ptrs + (i * isize), *g_coset, *g_coset_inv, small_size);
            NTT::Base_dev_ptr(gpu, advice_device_ptrs + (i * isize), log_isize,
                              NTT::InputOutputOrder::NN, NTT::Direction::forward,
                              NTT::Type::standard);
        }

        for (int i = 0; i < instance_ptr_len; i++) {
            zero_padding_kernel<<<grid_size_pad, threads_per_block, 0, gpu>>>(
                before_ntt_instance_device_ptrs + (i * small_size),
                instance_device_ptrs + (i * isize), *g_coset, *g_coset_inv, small_size);
            NTT::Base_dev_ptr(gpu, instance_device_ptrs + (i * isize), log_isize,
                              NTT::InputOutputOrder::NN, NTT::Direction::forward,
                              NTT::Type::standard);
        }

        ///////////////////////////////////
        const fr_t zero_fr = fr_t{};
        auto const_at = [&](size_t idx, const fr_t* arr, size_t arr_len) -> fr_t {
            return (arr_len && idx < arr_len) ? arr[idx] : zero_fr;
        };
        ///////////////////////////////////

        for (size_t outer = 0; outer < num_parts; outer++) {
            for (size_t i = 0; i < calculations_count; ++i) {
                const auto& info = calculations[i];
                const auto& calc = info.calculation;

                switch (calc.kind) {
                    case CalculationKind::Add: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            add_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            add_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            add_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            add_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Sub: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            sub_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            sub_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            sub_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            sub_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Mul: {
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            mul_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            mul_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            mul_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            mul_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Square: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            square_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            square_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Double: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            double_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            double_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Negate: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            negate_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            negate_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Store: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        fr_t* output_d = intermediate_device_ptrs + (info.target * csize);

                        int offset_in = outer * csize;

                        store_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                            in_a.pointer, output_d, rotation_value[calc.a.param1],
                            rot_scale, isize, csize, offset_in);
                        CUDA_OK(cudaGetLastError());

                        break;
                    }
                    case CalculationKind::Horner: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        // Intermediate
                        ResolvedInput in_c = get_resolve_input(
                            calc.extra,
                            (constants_ptr_len == 0 ? fr_t{}
                                                    : constants[calc.extra.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.extra.param0]),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        fr_t* output_d = intermediate_device_ptrs + (info.target * csize);

                        size_t horner_size = horner_index.size();

                        horner_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                            in_a.pointer + (outer * csize), intermediate_device_ptrs,
                            output_d, in_c.constant, horner_index_device_ptrs,
                            horner_size, csize);
                        gpu.sync();
                        CUDA_OK(cudaGetLastError());

                        break;
                    }
                    default:
                        throw std::invalid_argument("Unknown Calculation");
                }
            }

            int offset_in = outer * csize;

            if (flag_c) {
                gpu.DtoH(output + offset_in,
                         intermediate_device_ptrs + ((calculations_count - 1) * csize),
                         csize);

                CUDA_OK(cudaGetLastError());
                gpu.sync();

                MemoryPool::get_instance().~MemoryPool();
                gpu.sync();
            } else {
                cudaMemcpyAsync(
                    prev_device_ptrs + offset_in,
                    intermediate_device_ptrs + ((calculations_count - 1) * csize),
                    csize * sizeof(fr_t), cudaMemcpyDeviceToDevice, gpu);
            }

            gpu.sync();
        }

        CUDA_OK(cudaGetLastError());
        gpu.sync();

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

enum class AnyFFI : uint8_t { Advice, Fixed, Instance };

struct ColumnFFI {
    size_t index;
    AnyFFI column_type;
};

fr_t* get_any_input(const ColumnFFI& src, fr_t* advice, fr_t* fixed, fr_t* instance,
                    size_t poly_offset) {
    switch (src.column_type) {
        case AnyFFI::Advice: {
            return advice + (src.index * poly_offset);
        }
        case AnyFFI::Fixed: {
            return fixed + (src.index * poly_offset);
        }
        case AnyFFI::Instance: {
            return instance + (src.index * poly_offset);
        }
        default:
            throw std::invalid_argument("Unknown ValueSourceKind");
    }
}

__global__ void pow_kernel(fr_t base, fr_t* __restrict__ out) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    fr_t base_g = base;
    uint32_t p = tid;

    fr_t sqr = base_g;
    base_g = fr_t::csel(base_g, fr_t::one(), p & 1);

#pragma unroll 1
    while (p >>= 1) {
        sqr *= sqr;
        if (p & 1) base_g *= sqr;
    }

    out[tid] = base_g;
}

__global__ void pow_mul_kernel(fr_t base, fr_t* __restrict__ out,
                               const fr_t delta_start) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    fr_t delta_start_r = delta_start;
    fr_t base_r = base;
    uint32_t p = tid;

    fr_t sqr = base_r;
    base_r = fr_t::csel(base_r, fr_t::one(), p & 1);

#pragma unroll 1
    while (p >>= 1) {
        sqr *= sqr;
        if (p & 1) base_r *= sqr;
    }

    fr_t result = base_r * delta_start_r;
    out[tid] = result;
}

__global__ void mul_kernel(fr_t* value, const fr_t constant) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    fr_t constant_r = constant;
    fr_t value_r = value[tid];

    value_r = value_r * constant_r;

    value[tid] = value_r;
}

__global__ void permutation_stage1_kernel(fr_t* value, fr_t* perm, fr_t* first_perm,
                                          fr_t* last_perm, fr_t* l0, fr_t* l_last, fr_t y,
                                          int rot, int rot_scale, int polysize,
                                          int setlen) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    const fr_t y_r = y;
    fr_t value_r = value[tid];
    const fr_t l0_r = l0[tid];
    fr_t first_perm_r = first_perm[tid];

    value_r = value_r * y_r;
    fr_t temp = fr_t::one() - first_perm_r;
    value_r = value_r + (temp * l0_r);

    fr_t last_perm_r = last_perm[tid];
    fr_t l_last_r = l_last[tid];

    value_r = value_r * y_r;
    temp = last_perm_r * last_perm_r;
    temp = temp - last_perm_r;
    temp = temp * l_last_r;
    value_r = value_r + temp;

    size_t rotate_index = get_rotation_idx(tid, rot, rot_scale, polysize);

    for (int i = 1; i < setlen; i++) {
        int index1 = i * polysize;
        int index2 = index1 - polysize; // ((i - 1) * polysize)
        fr_t perm1_r = perm[index1 + tid];
        fr_t perm2_r = perm[index2 + rotate_index];

        value_r = value_r * y_r;
        temp = perm1_r - perm2_r;
        temp = temp * l0_r;
        value_r = value_r + temp;
    }

    value[tid] = value_r;
}

__global__ void permutation_stage2_left_kernel(fr_t* left, const fr_t* perm,
                                               const fr_t* column, const fr_t* coset,
                                               const fr_t beta, const fr_t gamma, int rot,
                                               int rot_scale, const int polysize,
                                               const bool first) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    fr_t input;
    if (first) {
        size_t rotate_index = get_rotation_idx(tid, rot, rot_scale, polysize);
        input = perm[rotate_index];
    } else {
        input = left[tid];
    }

    fr_t column_r = column[tid];
    fr_t coset_r = coset[tid];
    fr_t beta_r = beta;
    fr_t gamma_r = gamma;

    fr_t temp = coset_r * beta_r;
    temp = temp + column_r;
    temp = temp + gamma_r;

    input = input * temp;

    left[tid] = input;
}

__global__ void permutation_stage2_right_kernel(fr_t* right, const fr_t* perm,
                                                const fr_t* column,
                                                const fr_t* current_delta,
                                                const fr_t gamma, const int polysize,
                                                const bool first) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    fr_t input;
    if (first) {
        input = perm[tid];
    } else {
        input = right[tid];
    }

    fr_t column_r = column[tid];
    fr_t current_delta_r = current_delta[tid];
    fr_t gamma_r = gamma;

    fr_t temp = column_r + current_delta_r;
    temp = temp + gamma_r;

    input = input * temp;

    right[tid] = input;
}

__global__ void permutation_stage2_kernel(fr_t* value, const fr_t* left,
                                          const fr_t* right, const fr_t* l_active_row,
                                          const fr_t y, const int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    const fr_t y_r = y;
    fr_t value_r = value[tid];

    fr_t left_r = left[tid];
    fr_t right_r = right[tid];
    fr_t l_active_row_r = l_active_row[tid];

    value_r = value_r * y;

    fr_t temp = left_r - right_r;
    temp = temp * l_active_row_r;

    value_r = value_r + temp;

    value[tid] = value_r;
}

extern "C" RustError::by_value permutations_evaluation(
    const ColumnFFI* column, size_t column_count, const fr_t* const* permutation_ptrs,
    size_t permutation_ptr_len, const fr_t* const* pk_coset_ptrs, size_t pk_coset_ptr_len,
    const fr_t* const* advice_ptrs, size_t advice_ptr_len,
    const fr_t* const* instance_ptrs, size_t instance_ptr_len,
    const fr_t* const* fixed_ptrs, size_t fixed_ptr_len,

    fr_t* value,

    const fr_t* l0_ptr, size_t l0_ptr_len, const fr_t* l_last_ptr, size_t l_last_ptr_len,
    const fr_t* l_active_row_ptr, size_t l_active_row_ptr_len,

    const fr_t* delta_start, const fr_t* delta, const fr_t* beta, const fr_t* gamma,
    const fr_t* y, const fr_t* extended_omega,

    int chunk_len, int last_rotation_value, int rot_scale, int isize,

    const fr_t* g_coset, const fr_t* g_coset_inv, int small_size, const int flag) {
    const gpu_t& gpu = select_gpu();

    size_t csize = isize;
    const int threads_per_block = 256;
    size_t grid_size = (int)((csize + threads_per_block - 1) / threads_per_block);

    bool flag_c = flag == 1 ? true : false;

    try {
        fr_t* fixed_device_ptrs = MemoryPool::get_instance().fix_ptr();
        fr_t* advice_device_ptrs = MemoryPool::get_instance().advice_ptr();
        fr_t* instance_device_ptrs = MemoryPool::get_instance().instance_ptr();
        fr_t* value_device_ptrs = MemoryPool::get_instance().value_ptr();

        size_t total_poly_size = ((permutation_ptr_len + pk_coset_ptr_len) * isize) +
                                 isize + isize + isize + isize +
                                 (permutation_ptr_len * small_size);
        gpu_ptr_t<fr_t> input_polys((fr_t*)gpu.Dmalloc(total_poly_size * sizeof(fr_t)));
        CUDA_OK(cudaGetLastError());
        fr_t* permutation_device_ptrs = &input_polys[0];
        fr_t* pk_coset_device_ptrs =
            permutation_device_ptrs + (permutation_ptr_len * isize);
        fr_t* l0_device_ptrs = pk_coset_device_ptrs + (pk_coset_ptr_len * isize);
        fr_t* l_last_device_ptrs = l0_device_ptrs + (isize);
        fr_t* l_active_row_device_ptrs = l_last_device_ptrs + (isize);
        fr_t* before_ntt_permutation_device_ptrs = l_active_row_device_ptrs + (isize);

        for (int i = 0; i < permutation_ptr_len; i++) {
            gpu.HtoD(before_ntt_permutation_device_ptrs + (i * small_size),
                     permutation_ptrs[i], small_size);
            CUDA_OK(cudaGetLastError());
        }
        gpu.sync();

        for (int i = 0; i < pk_coset_ptr_len; i++) {
            gpu.HtoD(pk_coset_device_ptrs + (i * isize), pk_coset_ptrs[i], isize);
            CUDA_OK(cudaGetLastError());
        }
        gpu.sync();

        gpu.HtoD(l0_device_ptrs, l0_ptr, isize);
        CUDA_OK(cudaGetLastError());

        gpu.HtoD(l_last_device_ptrs, l_last_ptr, isize);
        CUDA_OK(cudaGetLastError());

        gpu.HtoD(l_active_row_device_ptrs, l_active_row_ptr, isize);
        CUDA_OK(cudaGetLastError());

        uint32_t log_isize = uint32_t(log2(isize));
        size_t grid_size_pad = (int)((isize + threads_per_block - 1) / threads_per_block);

        for (int i = 0; i < permutation_ptr_len; i++) {
            zero_padding_kernel<<<grid_size_pad, threads_per_block, 0, gpu>>>(
                before_ntt_permutation_device_ptrs + (i * small_size),
                permutation_device_ptrs + (i * isize), *g_coset, *g_coset_inv,
                small_size);
            NTT::Base_dev_ptr(gpu, permutation_device_ptrs + (i * isize), log_isize,
                              NTT::InputOutputOrder::NN, NTT::Direction::forward,
                              NTT::Type::standard);
        }

        gpu_ptr_t<fr_t> power_memory((fr_t*)gpu.Dmalloc(csize * sizeof(fr_t)));
        CUDA_OK(cudaGetLastError());
        fr_t* power_memory_ptrs = &power_memory[0];

        // pow_kernel<<<grid_size, threads_per_block, 0, gpu>>>(*extended_omega,
        // power_memory_ptrs);
        pow_mul_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
            *extended_omega, power_memory_ptrs, *delta_start);
        CUDA_OK(cudaGetLastError());

        fr_t* first_permutation_device_ptrs = permutation_device_ptrs;
        fr_t* last_permutation_device_ptrs =
            permutation_device_ptrs + ((permutation_ptr_len - 1) * isize);

        permutation_stage1_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
            value_device_ptrs, permutation_device_ptrs, first_permutation_device_ptrs,
            last_permutation_device_ptrs, l0_device_ptrs, l_last_device_ptrs, *y,
            last_rotation_value, rot_scale, isize, permutation_ptr_len);
        CUDA_OK(cudaGetLastError());

        gpu_ptr_t<fr_t> left_right_memory(
            (fr_t*)gpu.Dmalloc((isize + isize) * sizeof(fr_t)));
        CUDA_OK(cudaGetLastError());
        fr_t* left_ptrs = &left_right_memory[0];
        fr_t* right_ptrs = left_ptrs + isize;

        size_t num_chunks = (column_count + chunk_len - 1) / chunk_len;
        assert(permutation_ptr_len == num_chunks);
        for (int i = 0; i < num_chunks; i++) {
            int begin = i * chunk_len;
            std::size_t end =
                std::min(static_cast<size_t>(begin + chunk_len), column_count);

            fr_t* perm_in_ptr = permutation_device_ptrs + (i * isize);

            bool first_kernel = true;
            for (int j = 0; j < end - begin; j++) {
                fr_t* pointer_value =
                    get_any_input(column[begin + j], advice_device_ptrs,
                                  fixed_device_ptrs, instance_device_ptrs, isize);

                fr_t* coset_ptrs = pk_coset_device_ptrs + ((begin + j) * isize);

                permutation_stage2_left_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                    left_ptrs, perm_in_ptr, pointer_value, coset_ptrs, *beta, *gamma, 1,
                    rot_scale, isize, first_kernel);
                CUDA_OK(cudaGetLastError());

                first_kernel = false;
            }

            first_kernel = true;
            for (int j = 0; j < end - begin; j++) {
                fr_t* pointer_value =
                    get_any_input(column[begin + j], advice_device_ptrs,
                                  fixed_device_ptrs, instance_device_ptrs, isize);

                permutation_stage2_right_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                    right_ptrs, perm_in_ptr, pointer_value, power_memory_ptrs, *gamma,
                    isize, first_kernel);
                CUDA_OK(cudaGetLastError());

                mul_kernel<<<grid_size, threads_per_block, 0, gpu>>>(power_memory_ptrs,
                                                                     *delta);
                CUDA_OK(cudaGetLastError());

                first_kernel = false;
            }

            permutation_stage2_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                value_device_ptrs, left_ptrs, right_ptrs, l_active_row_device_ptrs, *y,
                isize);
            CUDA_OK(cudaGetLastError());
        }

        if (flag_c) {
            gpu.DtoH(value, value_device_ptrs, isize);
            gpu.sync();

            CUDA_OK(cudaGetLastError());
            gpu.sync();

            MemoryPool::get_instance().~MemoryPool();
            gpu.sync();
        }

        CUDA_OK(cudaGetLastError());
        gpu.sync();

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

__global__ void lookups_stage1_kernel(fr_t* value, fr_t* product_coset, fr_t* l0,
                                      fr_t* l_last, fr_t y, int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    const fr_t y_r = y;
    fr_t value_r = value[tid];
    fr_t product_coset_r = product_coset[tid];
    const fr_t l0_r = l0[tid];
    fr_t l_last_r = l_last[tid];

    value_r = value_r * y_r;
    fr_t temp = fr_t::one() - product_coset_r;
    value_r = value_r + (temp * l0_r);

    value_r = value_r * y_r;
    temp = product_coset_r * product_coset_r;
    temp = temp - product_coset_r;
    temp = temp * l_last_r;
    value_r = value_r + temp;

    value[tid] = value_r;
}

__global__ void lookups_stage2_kernel(fr_t* value, fr_t* table_value, fr_t* product_coset,
                                      fr_t* permuted_input_coset,
                                      fr_t* permuted_table_coset, fr_t* l_active_row,
                                      const fr_t y, const fr_t beta, const fr_t gamma,
                                      int rot_scale, int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= polysize) return;

    fr_t value_r = value[tid];
    value_r = value_r * y;

    fr_t temp1;
    {
        size_t r_next = get_rotation_idx(tid, 1, rot_scale, polysize);
        const fr_t product_coset_r_next = product_coset[r_next];
        const fr_t permuted_input_coset_r = permuted_input_coset[tid];
        temp1 = product_coset_r_next * (permuted_input_coset_r + beta);

        const fr_t permuted_table_coset_r = permuted_table_coset[tid];
        temp1 = temp1 * (permuted_table_coset_r + gamma);

        const fr_t product_coset_r = product_coset[tid];
        const fr_t table_value_r = table_value[tid];
        temp1 = temp1 - (product_coset_r * table_value_r);

        const fr_t l_active_row_r = l_active_row[tid];
        temp1 = temp1 * l_active_row_r;
    }

    value_r = value_r + temp1;
    value[tid] = value_r;
}

__global__ void lookups_stage3_kernel(fr_t* value, fr_t* permuted_input_coset,
                                      fr_t* permuted_table_coset, fr_t* l0,
                                      fr_t* l_active_row, const fr_t y, const fr_t beta,
                                      const fr_t gamma, int rot_scale, int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= polysize) return;

    fr_t value_r = value[tid];
    value_r = value_r * y;

    fr_t permuted_input_coset_r = permuted_input_coset[tid];
    fr_t permuted_table_coset_r = permuted_table_coset[tid];
    fr_t a_minus_s_r = permuted_input_coset_r - permuted_table_coset_r;

    fr_t l0_r = l0[tid];
    value_r = value_r + (a_minus_s_r * l0_r);

    value_r = value_r * y;

    size_t r_prev = get_rotation_idx(tid, -1, rot_scale, polysize);
    fr_t permuted_input_coset_r_prev = permuted_input_coset[r_prev];
    fr_t temp = a_minus_s_r * (permuted_input_coset_r - permuted_input_coset_r_prev);

    fr_t l_active_row_r = l_active_row[tid];
    temp = temp * l_active_row_r;
    value_r = value_r + temp;
    value[tid] = value_r;
}

extern "C" RustError::by_value lookups_evaluation(
    const CalculationInfoFFI* calculations, size_t calculations_count,
    const fr_t* challenges, size_t challenges_ptr_len, const fr_t* beta,
    const fr_t* gamma, const fr_t* theta, const fr_t* y, fr_t* value,
    const fr_t* constants, size_t constants_ptr_len, int* rotation_value,
    size_t rotation_ptr_len, int rot_scale, int isize,

    const fr_t* product_coset, const fr_t* permuted_input_coset,
    const fr_t* permuted_table_coset, int small_size,

    const fr_t* g_coset, const fr_t* g_coset_inv, const int flag) {
    constexpr size_t CHUNK_SIZE = 1 << 18; // To minimize memory usage
    const gpu_t& gpu = select_gpu();

    size_t csize = isize;
    size_t num_parts = 1;
    if (isize > CHUNK_SIZE) {
        csize = CHUNK_SIZE;
        num_parts = isize / CHUNK_SIZE;
    }

    const int threads_per_block = 512;
    size_t grid_size = (int)((csize + threads_per_block - 1) / threads_per_block);

    bool flag_c = flag == 1 ? true : false;

    try {
        fr_t* fixed_device_ptrs = MemoryPool::get_instance().fix_ptr();
        fr_t* advice_device_ptrs = MemoryPool::get_instance().advice_ptr();
        fr_t* instance_device_ptrs = MemoryPool::get_instance().instance_ptr();
        fr_t* l0_device_ptrs = MemoryPool::get_instance().l0_ptr();
        fr_t* l_last_device_ptrs = MemoryPool::get_instance().l_last_ptr();
        fr_t* l_active_row_device_ptrs = MemoryPool::get_instance().l_active_row_ptr();
        fr_t* value_device_ptrs = MemoryPool::get_instance().value_ptr();

        size_t total_poly_size = (5 * isize) + (3 * small_size);
        gpu_ptr_t<fr_t> input_polys((fr_t*)gpu.Dmalloc(total_poly_size * sizeof(fr_t)));
        CUDA_OK(cudaGetLastError());

        fr_t* product_coset_device_ptrs = &input_polys[0];
        fr_t* permuted_input_coset_device_ptrs = product_coset_device_ptrs + isize;
        fr_t* permuted_table_coset_device_ptrs = permuted_input_coset_device_ptrs + isize;
        fr_t* table_value_device_ptrs = permuted_table_coset_device_ptrs + isize;
        fr_t* before_ntt_product_coset_device_ptrs = table_value_device_ptrs + isize;
        fr_t* before_ntt_permuted_input_coset_device_ptrs =
            before_ntt_product_coset_device_ptrs + small_size;
        fr_t* before_ntt_permuted_table_coset_device_ptrs =
            before_ntt_permuted_input_coset_device_ptrs + small_size;

        gpu.HtoD(before_ntt_product_coset_device_ptrs, product_coset, small_size);
        CUDA_OK(cudaGetLastError());

        gpu.HtoD(before_ntt_permuted_input_coset_device_ptrs, permuted_input_coset,
                 small_size);
        CUDA_OK(cudaGetLastError());

        gpu.HtoD(before_ntt_permuted_table_coset_device_ptrs, permuted_table_coset,
                 small_size);
        CUDA_OK(cudaGetLastError());

        fr_t* prev_device_ptrs; // nullptr

        size_t total_intermediate_size = calculations_count * csize;
        gpu_ptr_t<fr_t> intermediate_values(
            (fr_t*)gpu.Dmalloc(total_intermediate_size * sizeof(fr_t)));
        CUDA_OK(cudaGetLastError());
        fr_t* intermediate_device_ptrs = &intermediate_values[0];

        uint32_t log_isize = uint32_t(log2(isize));
        size_t grid_size_pad = (int)((isize + threads_per_block - 1) / threads_per_block);

        zero_padding_kernel<<<grid_size_pad, threads_per_block, 0, gpu>>>(
            before_ntt_product_coset_device_ptrs, product_coset_device_ptrs, *g_coset,
            *g_coset_inv, small_size);
        NTT::Base_dev_ptr(gpu, product_coset_device_ptrs, log_isize,
                          NTT::InputOutputOrder::NN, NTT::Direction::forward,
                          NTT::Type::standard);

        zero_padding_kernel<<<grid_size_pad, threads_per_block, 0, gpu>>>(
            before_ntt_permuted_input_coset_device_ptrs, permuted_input_coset_device_ptrs,
            *g_coset, *g_coset_inv, small_size);
        NTT::Base_dev_ptr(gpu, permuted_input_coset_device_ptrs, log_isize,
                          NTT::InputOutputOrder::NN, NTT::Direction::forward,
                          NTT::Type::standard);

        zero_padding_kernel<<<grid_size_pad, threads_per_block, 0, gpu>>>(
            before_ntt_permuted_table_coset_device_ptrs, permuted_table_coset_device_ptrs,
            *g_coset, *g_coset_inv, small_size);
        NTT::Base_dev_ptr(gpu, permuted_table_coset_device_ptrs, log_isize,
                          NTT::InputOutputOrder::NN, NTT::Direction::forward,
                          NTT::Type::standard);

        ///////////////////////////////////
        const fr_t zero_fr = fr_t{};
        auto const_at = [&](size_t idx, const fr_t* arr, size_t arr_len) -> fr_t {
            return (arr_len && idx < arr_len) ? arr[idx] : zero_fr;
        };
        ///////////////////////////////////

        for (size_t outer = 0; outer < num_parts; outer++) {
            for (size_t i = 0; i < calculations_count; ++i) {
                const auto& info = calculations[i];
                const auto& calc = info.calculation;

                switch (calc.kind) {
                    case CalculationKind::Add: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            add_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            add_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            add_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            add_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Sub: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            sub_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            sub_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            sub_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            sub_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Mul: {
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            mul_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            mul_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            mul_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            mul_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Square: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            square_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            square_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Double: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            double_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            double_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Negate: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            negate_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * csize);

                            negate_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Store: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        fr_t* output_d = intermediate_device_ptrs + (info.target * csize);

                        int offset_in = outer * csize;

                        store_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                            in_a.pointer, output_d, rotation_value[calc.a.param1],
                            rot_scale, isize, csize, offset_in);
                        CUDA_OK(cudaGetLastError());

                        break;
                    }
                    case CalculationKind::Horner: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        // Intermediate
                        ResolvedInput in_c = get_resolve_input(
                            calc.extra,
                            (constants_ptr_len == 0 ? fr_t{}
                                                    : constants[calc.extra.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.extra.param0]),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);

                        fr_t* output_d = intermediate_device_ptrs + (info.target * csize);

                        std::vector<size_t> horner_index;
                        for (size_t i = 0; i < calc.horner_parts_len; i++) {
                            const auto& part = calc.horner_parts_ptr[i];
                            horner_index.push_back(part.param0);
                        }

                        gpu_ptr_t<size_t> horner_index_values(
                            (size_t*)gpu.Dmalloc(horner_index.size() * sizeof(size_t)));
                        gpu.sync();
                        size_t* horner_index_device_ptrs = &horner_index_values[0];
                        gpu.HtoD(horner_index_device_ptrs, horner_index.data(),
                                 horner_index.size());
                        CUDA_OK(cudaGetLastError());
                        gpu.sync();

                        size_t horner_size = horner_index.size();

                        if (!in_a.is_constant) {
                            horner_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer + (outer * csize), intermediate_device_ptrs,
                                output_d, in_c.constant, horner_index_device_ptrs,
                                horner_size, csize);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            horner_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, intermediate_device_ptrs, output_d,
                                in_c.constant, horner_index_device_ptrs, horner_size,
                                csize);
                            CUDA_OK(cudaGetLastError());
                        }

                        gpu.sync();
                        CUDA_OK(cudaGetLastError());

                        break;
                    }
                    default:
                        throw std::invalid_argument("Unknown Calculation");
                }
            }

            int offset_in = outer * csize;
            cudaMemcpyAsync(table_value_device_ptrs + offset_in,
                            intermediate_device_ptrs + ((calculations_count - 1) * csize),
                            csize * sizeof(fr_t), cudaMemcpyDeviceToDevice, gpu);

            gpu.sync();
        }

        const int threads_per_block2 = 256;
        size_t grid_size2 = (int)((isize + threads_per_block2 - 1) / threads_per_block2);

        lookups_stage1_kernel<<<grid_size2, threads_per_block2, 0, gpu>>>(
            value_device_ptrs, product_coset_device_ptrs, l0_device_ptrs,
            l_last_device_ptrs, *y, isize);
        CUDA_OK(cudaGetLastError());

        lookups_stage2_kernel<<<grid_size2, threads_per_block2, 0, gpu>>>(
            value_device_ptrs, table_value_device_ptrs, product_coset_device_ptrs,
            permuted_input_coset_device_ptrs, permuted_table_coset_device_ptrs,
            l_active_row_device_ptrs, *y, *beta, *gamma, rot_scale, isize);
        CUDA_OK(cudaGetLastError());

        lookups_stage3_kernel<<<grid_size2, threads_per_block2, 0, gpu>>>(
            value_device_ptrs, permuted_input_coset_device_ptrs,
            permuted_table_coset_device_ptrs, l0_device_ptrs, l_active_row_device_ptrs,
            *y, *beta, *gamma, rot_scale, isize);
        CUDA_OK(cudaGetLastError());
        gpu.sync();

        if (flag_c) {
            gpu.DtoH(value, value_device_ptrs, isize);
            gpu.sync();

            CUDA_OK(cudaGetLastError());
            gpu.sync();

            MemoryPool::get_instance().~MemoryPool();
            gpu.sync();
        }

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}
