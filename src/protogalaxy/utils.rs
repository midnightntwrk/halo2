use ff::PrimeField;

/// Given a vector v, computes a vector of length 2^|v| whose i-th element
/// is the product of {v_j : bin(i)_j = 1}. Where bin(i)_j is the
/// j-th (little-endian) bit of i.
pub(crate) fn pow_vec<F: PrimeField>(vector: &[F]) -> Vec<F> {
    let mut res = vec![F::ONE];
    for x in vector {
        res.extend(res.clone().iter().map(|v| *v * x));
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::Scalar as F;
    use ff::Field;

    #[test]
    fn test_pow_vec() {
        let vector = vec![F::from(2), F::from(3)];
        let result = pow_vec(&vector);
        assert_eq!(result, vec![F::ONE, F::from(2), F::from(3), F::from(6)]);
    }
}
