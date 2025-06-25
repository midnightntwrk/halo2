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
        static RandomMSM instance;
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

extern "C" RustError::by_value custom_gates_evaluation(
    const CalculationInfoFFI* calculations, size_t calculations_count,
    const fr_t* const* fixed_ptrs, size_t fixed_ptr_len, const fr_t* const* advice_ptrs,
    size_t advice_ptr_len, const fr_t* const* instance_ptrs, size_t instance_ptr_len,
    const fr_t* challenges, size_t challenges_ptr_len, const fr_t* beta,
    const fr_t* gamma, const fr_t* theta, const fr_t* y, fr_t* output,
    const fr_t* constants, size_t constants_ptr_len, int* rotation_value,
    size_t rotation_ptr_len, int rot_scale, int isize) {
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

    try {
        size_t total_poly_size =
            ((fixed_ptr_len + advice_ptr_len + instance_ptr_len) * isize) + isize;
        gpu_ptr_t<fr_t> input_polys((fr_t*)gpu.Dmalloc(total_poly_size * sizeof(fr_t)));
        fr_t* fixed_device_ptrs = &input_polys[0];
        fr_t* advice_device_ptrs = fixed_device_ptrs + (fixed_ptr_len * isize);
        fr_t* instance_device_ptrs = advice_device_ptrs + (advice_ptr_len * isize);
        fr_t* prev_device_ptrs = instance_device_ptrs + (instance_ptr_len * isize);

        for (int i = 0; i < fixed_ptr_len; i++) {
            gpu.HtoD(fixed_device_ptrs + (i * isize), fixed_ptrs[i], isize);
        }

        for (int i = 0; i < advice_ptr_len; i++) {
            gpu.HtoD(advice_device_ptrs + (i * isize), advice_ptrs[i], isize);
        }

        for (int i = 0; i < instance_ptr_len; i++) {
            gpu.HtoD(instance_device_ptrs + (i * isize), instance_ptrs[i], isize);
        }

        gpu.HtoD(prev_device_ptrs, output, isize);

        size_t total_intermediate_size = calculations_count * csize;
        gpu_ptr_t<fr_t> intermediate_values(
            (fr_t*)gpu.Dmalloc(total_intermediate_size * sizeof(fr_t)));
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

        for (size_t outer = 0; outer < num_parts; outer++) {
            for (size_t i = 0; i < calculations_count; ++i) {
                const auto& info = calculations[i];
                const auto& calc = info.calculation;

                switch (calc.kind) {
                    case CalculationKind::Add: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.a.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.a.param0]),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.b.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.b.param0]),
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
                            calc.a,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.a.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.a.param0]),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.b.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.b.param0]),
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
                        ResolvedInput in_a = get_resolve_input(
                            calc.a,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.a.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.a.param0]),
                            *beta, *gamma, *theta, *y, prev_device_ptrs, csize, isize);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.b.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.b.param0]),
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
                            calc.a,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.a.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.a.param0]),
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
                            calc.a,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.a.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.a.param0]),
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
                            calc.a,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.a.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.a.param0]),
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
                            calc.a,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.a.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.a.param0]),
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
                            calc.a,
                            (constants_ptr_len == 0 ? fr_t{} : constants[calc.a.param0]),
                            intermediate_device_ptrs, fixed_device_ptrs,
                            advice_device_ptrs, instance_device_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.a.param0]),
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
            gpu.DtoH(output + offset_in,
                     intermediate_device_ptrs + ((calculations_count - 1) * csize),
                     csize);
            gpu.sync();
        }

        CUDA_OK(cudaGetLastError());
        gpu.sync();

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
