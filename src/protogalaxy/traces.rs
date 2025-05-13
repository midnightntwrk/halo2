//! TODO
//!

use std::ops::{Add, Mul};

use ff::{PrimeField, WithSmallOrderMulGroup};

use crate::plonk::permutation;
use crate::{
    plonk::Evaluator,
    poly::{EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial},
};

/// ω in the protogalaxy paper.
pub(crate) struct FoldingTrace<F: PrimeField> {
    fixed_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    advice_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    instance_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    lookups: Vec<crate::plonk::lookup::prover::Committed<F>>,
    permutations: permutation::prover::Committed<F>,
    challenges: Vec<F>,
    beta: F,
    gamma: F,
    theta: F,
    y: F,
}

impl<F: PrimeField> Add<FoldingTrace<F>> for FoldingTrace<F> {
    type Output = Self;

    fn add(self, rhs: FoldingTrace<F>) -> Self {
        assert_eq!(self.fixed_polys.len(), rhs.fixed_polys.len());
        assert_eq!(self.advice_polys.len(), rhs.advice_polys.len());
        assert_eq!(self.instance_polys.len(), rhs.instance_polys.len());
        assert_eq!(self.challenges.len(), rhs.challenges.len());

        let mut result = self.clone();

        for (lhs, rhs) in result.fixed_polys.iter_mut().zip(rhs.fixed_polys.iter()) {
            *lhs = *lhs + rhs;
        }
        for (lhs, rhs) in result.advice_polys.iter_mut().zip(rhs.advice_polys.iter()) {
            *lhs = *lhs + rhs;
        }
        for (lhs, rhs) in (result.instance_polys.iter_mut()).zip(rhs.instance_polys.iter()) {
            *lhs = *lhs + rhs;
        }
        for (lhs, rhs) in result.challenges.iter_mut().zip(rhs.challenges.iter()) {
            *lhs += *rhs;
        }
        result.beta += rhs.beta;
        result.gamma += rhs.gamma;
        result.theta += rhs.theta;
        result.y += rhs.y;

        result
    }
}

impl<F: PrimeField> Mul<F> for FoldingTrace<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self {
        let mut result = self.clone();

        for (lhs, rhs) in result.fixed_polys.iter_mut().zip(rhs.fixed_polys.iter()) {
            *lhs = *lhs * rhs;
        }
        for (lhs, rhs) in result.advice_polys.iter_mut().zip(rhs.advice_polys.iter()) {
            *lhs = *lhs * rhs;
        }
        for (lhs, rhs) in (result.instance_polys.iter_mut()).zip(rhs.instance_polys.iter()) {
            *lhs = *lhs * rhs;
        }
        for (lhs, rhs) in result.challenges.iter_mut().zip(rhs.challenges.iter()) {
            *lhs *= rhs;
        }
        result.beta *= rhs;
        result.gamma *= rhs;
        result.theta *= rhs;
        result.y *= rhs;

        result
    }
}

/// A folding trace where, instead of field elements, we have polynomials.
/// It is represented as a vector of folding traces, where the i-th folding trace
/// represents the evaluation of the polynomial at the i-th domain point.
pub type LiftedFoldingTrace<F> = Vec<FoldingTrace<F>>;

/// Computes \sum_{j = 0}^k L_j(X) ω_j, where ω_j is the j-th trace,
/// for j = 0, ..., k and the `lagrange_polys` L_j(X) are given in evaluations
/// form (in an extended `domain`).
///
/// # Panics
///
/// If |lagrange_polys| != |traces|.
/// If the number of coefficients in each `lagrange_polys` is not equal.
fn batch_traces<F: PrimeField>(
    lagrange_polys: &[Polynomial<F, ExtendedLagrangeCoeff>],
    traces: &[&FoldingTrace<F>],
) -> LiftedFoldingTrace<F> {
    let domain_size = lagrange_polys[0].num_coeffs();

    assert!(lagrange_polys.len() == traces.len());
    assert!(lagrange_polys
        .iter()
        .all(|poly| poly.num_coeffs() == domain_size));

    (0..domain_size)
        .map(|i| {
            (lagrange_polys.iter())
                .zip(traces.iter())
                .map(|(lagrange_poly, trace)| trace * lagrange_poly[i])
                .sum()
        })
        .collect()
}
