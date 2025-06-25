//! # halo2_proofs

#![cfg_attr(docsrs, feature(doc_cfg))]
// The actual lints we want to disable.
#![allow(clippy::op_ref, clippy::many_single_char_names)]
#![deny(rustdoc::broken_intra_doc_links)] // remove it
#![deny(missing_debug_implementations)]
//#![deny(missing_docs)]
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
use core::ffi::{c_void, c_int};
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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum ValueSourceKind {
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
    PreviousValue,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ValueSourceFFI {
    pub kind: ValueSourceKind,
    pub param0: usize,
    pub param1: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum CalculationKind {
    Add,
    Sub,
    Mul,
    Square,
    Double,
    Negate,
    Store,
    Horner,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct CalculationFFI {
    pub kind: CalculationKind,
    pub a: ValueSourceFFI,
    pub b: ValueSourceFFI,
    pub extra: ValueSourceFFI,
    pub horner_parts_ptr: *const ValueSourceFFI,
    pub horner_parts_len: usize,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct CalculationInfoFFI {
    pub calculation: CalculationFFI,
    pub target: usize,
}

extern "C" {
    fn custom_gates_evaluation(
        calculations: *const CalculationInfoFFI,
        calculations_count: usize,

        fixed_ptrs: *const *const c_void,
        fixed_ptr_len: usize,

        advice_ptrs: *const *const c_void,
        advice_ptr_len: usize,

        instance_ptrs: *const *const c_void,
        instance_ptr_len: usize,

        challenges: *const c_void,
        challenges_ptr_len: usize,

        beta: *const c_void,
        gamma: *const c_void,
        theta: *const c_void,
        y: *const c_void,

        output: *mut c_void,

        constants: *const c_void,
        constants_ptr_len: usize,

        rotation_value: *const c_int,
        rotation_ptr_len: usize,

        rot_scale: c_int,
        isize: c_int,
    ) -> sppark::Error;
}

#[allow(unsafe_code)]
pub fn custom_gates_evaluation_r<T: std::clone::Clone>(
    calculation: &[CalculationInfoFFI],
    fixed_ptrs: &[*const T],
    advice_ptrs: &[*const T],
    instance_ptrs: &[*const T],
    challenges: &[T],
    beta: &T, gamma: &T, theta: &T, y: &T, 
    output: &mut [T],
    constants: &[T],
    rotation_value: &Vec<i32>,
    rot_scale: &i32,
    isize: &i32
) 
{ 
    let beta_p = &[ beta.clone() ];
    let gamma_p = &[ gamma.clone() ];
    let theta_p = &[ theta.clone() ];
    let y_p = &[ y.clone() ];

    unsafe {
        custom_gates_evaluation(calculation.as_ptr(), calculation.len(),
        fixed_ptrs.as_ptr() as *const *const c_void, fixed_ptrs.len(),
        advice_ptrs.as_ptr() as *const *const c_void, advice_ptrs.len(),
        instance_ptrs.as_ptr() as *const *const c_void, instance_ptrs.len(),
        challenges.as_ptr() as *const c_void, challenges.len(),

        beta_p.as_ptr() as *const c_void,
        gamma_p.as_ptr() as *const c_void,
        theta_p.as_ptr() as *const c_void,
        y_p.as_ptr() as *const c_void,

        output.as_mut_ptr() as *mut _, 

        constants.as_ptr() as *const c_void, constants.len(),

        rotation_value.as_ptr(), rotation_value.len(),

        *rot_scale, *isize
        );
    }

}