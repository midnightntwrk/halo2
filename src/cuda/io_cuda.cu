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

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

extern "C"
RustError::by_value sppark_msm(point_t* out, const affine_t points[],
                                size_t npoints, const scalar_t scalars[],
                                size_t ffi_affine_sz)
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars, true, ffi_affine_sz);
}

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
