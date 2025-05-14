use ff::Field;
use std::ops::{Add, Mul};

/// Given a vector v, computes a vector of length 2^|v| whose i-th element
/// is the product of {v_j : bin(i)_j = 1}. Where bin(i)_j is the
/// j-th (little-endian) bit of i.
pub(crate) fn pow_vec<F: Field>(vector: &[F]) -> Vec<F> {
    let mut res = vec![F::ONE];
    for x in vector {
        res.extend(res.clone().iter().map(|v| *v * x));
    }
    res
}

/// Computes a linear combination between the elements and the scalars.
/// Namely, `sum_i scalars[i] * elements[i]`.
///
/// # Panics
///
/// If the |elements| != |scalars|.
///
/// # Notes
///
/// This function works for any elements of type `T` and scalars of type
/// `F: Field` such that `T: Add<&T, Output = T> + Mul<F, Output = T>`, even
/// if these `Add` and `Mul` traits are implemented in-place (mutating self).
/// Type `T` does not need to implement `Copy` or `Clone`.
///
/// Note however that this function does not mutate the inputs, it only
/// mutates the buffer, in which the result is stored.
///
// We achieve this by computing a different linear combination using
// Horner's method. For that, first filter out all the elements whose
// corresponding scalar is zero, then we need to convert the scalars into a
// different form.
// Concretely, let k = |elements| = |scalars|. Let c_0 = 0, and scalar[k] = 1. Then,
// for every i ∈ [1, k] let c_i = scalars[i-1] / scalars[i].
//
// Then we compute the linear combination as:
// ```
// for i = 0 to k-1:
//   buffer *= c_i
//   buffer += elements[i]
// return buffer * c_k
// ```
//
// Note that at the end of this execution, the buffer contains:
//    c_1 * c_2 * ... * c_k * elements[0]
//  +       c_2 * ... * c_k * elements[1]
//  + ...
//  +                   c_k * elements[k-1]
//
// Finally, note that, given how we defined c_i, we have:
//   c_1 * c_2 * ... * c_k = scalars[0],
//         c_2 * ... * c_k = scalars[1],
//   ...
//                     c_k = scalars[k-1]
// as desired.
pub(crate) fn linear_combination<F, T>(mut buffer: T, elements: &[T], scalars: &[F]) -> T
where
    F: Field,
    T: for<'a> Add<&'a T, Output = T> + Mul<F, Output = T>,
{
    assert_eq!(elements.len(), scalars.len());

    // Filter out elements whose scalar is zero.
    let (elements, scalars): (Vec<&T>, Vec<&F>) = (elements.iter())
        .zip(scalars)
        .filter(|(_, s)| !s.is_zero_vartime())
        .unzip();

    let k = elements.len();
    let mut scalars = scalars.into_iter().cloned().collect::<Vec<_>>();
    scalars.push(F::ONE);
    let mut c = F::ZERO;

    for i in 0..k {
        buffer = buffer * c;
        buffer = buffer + elements[i];
        c = scalars[i] * scalars[i + 1].invert().unwrap();
    }

    buffer * c
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::Scalar as F;

    #[test]
    fn test_linear_combination() {
        let to_field = |v: &[u64]| -> Vec<F> { v.iter().cloned().map(F::from).collect() };
        let elements = vec![1, 2, 3, 4];
        let scalars = vec![1, 10, 100, 1000];
        let expected = 4321;
        let buffer = F::default();
        let result = linear_combination(buffer, &to_field(&elements), &to_field(&scalars));
        assert_eq!(result, F::from(expected));
    }

    #[test]
    fn test_pow_vec() {
        let to_field = |v: &[u64]| -> Vec<F> { v.iter().cloned().map(F::from).collect() };
        let input = vec![2, 3, 5];
        let expected = vec![1, 2, 3, 6, 5, 10, 15, 30];
        assert_eq!(pow_vec(&to_field(&input)), to_field(&expected));
    }
}
