//! TODO
//!

use std::ops::{Add, Mul};

use ff::{PrimeField, WithSmallOrderMulGroup};

use crate::{
    plonk::Evaluator,
    poly::{EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial},
};

/// ω in the protogalaxy paper.
struct FoldingTrace<F> {
    fixed_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    advice_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    instance_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    challenges: Vec<F>,
    beta: F,
    gamma: F,
    theta: F,
    y: F,
}

impl<F: PrimeField> Add<FoldingTrace<F>> for FoldingTrace<F> {
    type Output = Self;

    fn add(mut self, rhs: FoldingTrace<F>) -> Self {
        assert_eq!(self.fixed_polys.len(), rhs.fixed_polys.len());
        assert_eq!(self.advice_polys.len(), rhs.advice_polys.len());
        assert_eq!(self.instance_polys.len(), rhs.instance_polys.len());
        assert_eq!(self.challenges.len(), rhs.challenges.len());

        for (lhs, rhs) in self.fixed_polys.iter_mut().zip(rhs.fixed_polys.iter()) {
            *lhs = *lhs + rhs;
        }
        for (lhs, rhs) in self.advice_polys.iter_mut().zip(rhs.advice_polys.iter()) {
            *lhs = *lhs + rhs;
        }
        for (lhs, rhs) in (self.instance_polys.iter_mut()).zip(rhs.instance_polys.iter()) {
            *lhs = *lhs + rhs;
        }
        for (lhs, rhs) in self.challenges.iter_mut().zip(rhs.challenges.iter()) {
            *lhs += *rhs;
        }
        self.beta += rhs.beta;
        self.gamma += rhs.gamma;
        self.theta += rhs.theta;
        self.y += rhs.y;

        self
    }
}

impl<F: PrimeField> Mul<F> for FoldingTrace<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self {
        for (lhs, rhs) in self.fixed_polys.iter_mut().zip(rhs.fixed_polys.iter()) {
            *lhs = *lhs * rhs;
        }
        for (lhs, rhs) in self.advice_polys.iter_mut().zip(rhs.advice_polys.iter()) {
            *lhs = *lhs * rhs;
        }
        for (lhs, rhs) in (self.instance_polys.iter_mut()).zip(rhs.instance_polys.iter()) {
            *lhs = *lhs * rhs;
        }
        for (lhs, rhs) in self.challenges.iter_mut().zip(rhs.challenges.iter()) {
            *lhs *= rhs;
        }
        self.beta *= rhs;
        self.gamma *= rhs;
        self.theta *= rhs;
        self.y *= rhs;

        self
    }
}

/// A folding trace where instead of field elements, we have polynomials.
/// It is represented as a vector of folding traces, where the i-th folding trace
/// represents the evaluation of the polynomial at the i-th domain point.
type LiftedFoldingTrace<F> = Vec<FoldingTrace<F>>;

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

    let mut batched_traces = Vec::with_capacity(domain_size);
    for i in 0..domain_size {
        let mut trace = FoldingTrace {
            fixed_polys: Vec::new(),
            advice_polys: Vec::new(),
            instance_polys: Vec::new(),
            challenges: Vec::new(),
            beta: F::zero(),
            gamma: F::zero(),
            theta: F::zero(),
            y: F::zero(),
        };
        for (lagrange_poly, folding_trace) in lagrange_polys.iter().zip(traces.iter()) {
            trace
                .fixed_polys
                .push(lagrange_poly[i] * folding_trace.fixed_polys[i]);
            trace
                .advice_polys
                .push(lagrange_poly[i] * folding_trace.advice_polys[i]);
            trace
                .instance_polys
                .push(lagrange_poly[i] * folding_trace.instance_polys[i]);
            trace
                .challenges
                .push(lagrange_poly[i] * folding_trace.challenges[i]);
            trace.beta += lagrange_poly[i] * folding_trace.beta;
            trace.gamma += lagrange_poly[i] * folding_trace.gamma;
            trace.theta += lagrange_poly[i] * folding_trace.theta;
            trace.y += lagrange_poly[i] * folding_trace.y;
        }
        batched_traces.push(trace);
    }
    // (lagrange_polys.iter())
    //     .zip(traces.iter())
    //     .map(|(lagrange_poly, trace)| {
    //         let mut result = vec![F::zero(); domain_size];
    //         for (i, value) in trace.iter().enumerate() {
    //             for j in 0..lagrange_poly.num_coeffs() {
    //                 result[j] += lagrange_poly[i] * value[j];
    //             }
    //         }
    //         result
    //     })
    //     .collect()
}
