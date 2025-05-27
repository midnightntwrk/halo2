//! # halo2_proofs

#![cfg_attr(docsrs, feature(doc_cfg))]
// The actual lints we want to disable.
#![allow(clippy::op_ref, clippy::many_single_char_names)]
#![deny(rustdoc::broken_intra_doc_links)] // remove it
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)] // remove it

pub mod circuit;
pub use halo2curves;
pub mod plonk;
pub mod poly;
pub mod transcript;

pub mod dev;
pub mod utils;

//////////////////////////////////////////////
 
/// Timing Start
#[macro_export]
macro_rules! start_timer {
    ($name:ident) => {
        let $name = std::time::Instant::now();
    };
}

/// Timing End
#[macro_export]
macro_rules! end_timer {
    ($name:ident, $label:expr) => {
        {
            let duration = $name.elapsed();
            println!("{} done: {:.3?}", $label, duration);
        }
    };
}

//////////////////////////////////////////////
use sppark::{NTTInputOutputOrder, NTTDirection, NTTType};
use core::ffi::c_void;
use group::Group;
use halo2curves::CurveAffine;

extern "C" {
    fn gpu_msm(
        out1: *mut c_void,
        out2: *mut c_void,
        points_with_infinity: *const c_void,
        npoints: usize,
        scalars: *const c_void,
        ffi_affine_sz: usize,
    ) -> sppark::Error;
}

extern "C" {
    fn gpu_msm_lagrange(
        out1: *mut c_void,
        out2: *mut c_void,
        points_with_infinity: *const c_void,
        npoints: usize,
        scalars: *const c_void,
        ffi_affine_sz: usize,
    ) -> sppark::Error;
}

#[allow(unsafe_code)]
/// Perform MSM GPU
pub fn msm_gpu<C: CurveAffine>(
    points: &[C],
    scalars: &[C::Scalar],
) ->  C::Curve {

    let npoints = points.len();

    if npoints != scalars.len() {
        panic!("length mismatch")
    }

    let mut ret = C::Curve::identity();
    let mut ret2 = C::Curve::identity();
    let err = unsafe {
        gpu_msm(
            &mut ret as *mut _ as *mut _,
            &mut ret2 as *mut _ as *mut _,
            points.as_ptr() as *const _,
            npoints,
            scalars.as_ptr() as *const _,
            std::mem::size_of::<C>(),
        )
    };

    if err.code != 0 {
        panic!("MSM GPU error: {}", String::from(err));
    }

    ret + ret2
}

#[allow(unsafe_code)]
/// Perform MSM GPU
pub fn msm_gpu_lagrange<C: CurveAffine>(
    points: &[C],
    scalars: &[C::Scalar],
) ->  C::Curve {

    let npoints = points.len();

    if npoints != scalars.len() {
        panic!("length mismatch")
    }

    let mut ret = C::Curve::identity();
    let mut ret2 = C::Curve::identity();
    let err = unsafe {
        gpu_msm_lagrange(
            &mut ret as *mut _ as *mut _,
            &mut ret2 as *mut _ as *mut _,
            points.as_ptr() as *const _,
            npoints,
            scalars.as_ptr() as *const _,
            std::mem::size_of::<C>(),
        )
    };

    if err.code != 0 {
        panic!("MSM GPU error: {}", String::from(err));
    }

    ret + ret2
}

extern "C" {
    fn sppark_ntt(
        device_id: usize,
        inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        ntt_order: NTTInputOutputOrder,
        ntt_direction: NTTDirection,
        ntt_type: NTTType,
    ) -> sppark::Error;
}

/// Compute an in-place forward NTT on the input data.
#[allow(unsafe_code)]
pub fn ntt_gpu<T>(device_id: usize, inout: &mut [T], order: NTTInputOutputOrder) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        sppark_ntt(
            device_id,
            inout.as_mut_ptr() as *mut _,
            len.trailing_zeros(),
            order,
            NTTDirection::Forward,
            NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

/// Compute an in-place inverse NTT on the input data.
#[allow(unsafe_code)]
pub fn intt_gpu<T>(device_id: usize, inout: &mut [T], order: NTTInputOutputOrder) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        sppark_ntt(
            device_id,
            inout.as_mut_ptr() as *mut _,
            len.trailing_zeros(),
            order,
            NTTDirection::Inverse,
            NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}