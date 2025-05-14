//! TODO
#![allow(dead_code)]

use std::ops::{Add, Mul};

use ff::PrimeField;
use ff::WithSmallOrderMulGroup;

use crate::plonk::lookup;
use crate::plonk::permutation;
use crate::poly::EvaluationDomain;
use crate::poly::{LagrangeCoeff, Polynomial};

use super::utils::linear_combination;

/// ω in the protogalaxy paper.
pub(crate) struct FoldingTrace<F: PrimeField> {
    pub(crate) fixed_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    pub(crate) advice_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    pub(crate) instance_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    pub(crate) lookups: Vec<lookup::prover::Committed<F>>,
    pub(crate) permutation: permutation::prover::Committed<F>,
    pub(crate) challenges: Vec<F>,
    pub(crate) beta: F,
    pub(crate) gamma: F,
    pub(crate) theta: F,
    pub(crate) y: F,
}

impl<F: PrimeField> FoldingTrace<F> {
    pub fn init(
        domain_size: usize,
        num_fixed_polys: usize,
        num_advice_polys: usize,
        num_instance_polys: usize,
        num_lookups: usize,
        num_permutation_sets: usize,
        num_challenges: usize,
    ) -> Self {
        let mut lookups = Vec::with_capacity(num_lookups);
        for _ in 0..num_lookups {
            lookups.push(lookup::prover::Committed {
                permuted_input_poly: Polynomial::init(domain_size),
                permuted_table_poly: Polynomial::init(domain_size),
                product_poly: Polynomial::init(domain_size),
            });
        }
        let mut permutation_sets = Vec::with_capacity(num_permutation_sets);
        for _ in 0..num_permutation_sets {
            permutation_sets.push(permutation::prover::CommittedSet {
                permutation_product_poly: Polynomial::init(domain_size),
            });
        }
        FoldingTrace {
            fixed_polys: vec![Polynomial::init(domain_size); num_fixed_polys],
            advice_polys: vec![Polynomial::init(domain_size); num_advice_polys],
            instance_polys: vec![Polynomial::init(domain_size); num_instance_polys],
            lookups,
            permutation: permutation::prover::Committed {
                sets: permutation_sets,
            },
            challenges: vec![F::ZERO; num_challenges],
            beta: F::ZERO,
            gamma: F::ZERO,
            theta: F::ZERO,
            y: F::ZERO,
        }
    }
}

impl<'a, F: PrimeField> Add<&'a FoldingTrace<F>> for FoldingTrace<F> {
    type Output = Self;

    /// TODO: parallelize.
    fn add(mut self, rhs: &FoldingTrace<F>) -> Self {
        assert_eq!(self.fixed_polys.len(), rhs.fixed_polys.len());
        assert_eq!(self.advice_polys.len(), rhs.advice_polys.len());
        assert_eq!(self.instance_polys.len(), rhs.instance_polys.len());
        assert_eq!(self.challenges.len(), rhs.challenges.len());
        assert_eq!(self.lookups.len(), rhs.lookups.len());
        assert_eq!(self.permutation.sets.len(), rhs.permutation.sets.len());

        for (lhs, rhs) in self.fixed_polys.iter_mut().zip(rhs.fixed_polys.iter()) {
            *lhs += rhs;
        }
        for (lhs, rhs) in self.advice_polys.iter_mut().zip(rhs.advice_polys.iter()) {
            *lhs += rhs;
        }
        for (lhs, rhs) in (self.instance_polys.iter_mut()).zip(rhs.instance_polys.iter()) {
            *lhs += rhs;
        }
        for (lhs, rhs) in self.challenges.iter_mut().zip(rhs.challenges.iter()) {
            *lhs += *rhs;
        }
        for (lhs, rhs) in self.lookups.iter_mut().zip(rhs.lookups.iter()) {
            lhs.permuted_input_poly += &rhs.permuted_input_poly;
            lhs.permuted_table_poly += &rhs.permuted_table_poly;
            lhs.product_poly += &rhs.product_poly;
        }
        for (lhs, rhs) in (self.permutation.sets.iter_mut()).zip(rhs.permutation.sets.iter()) {
            lhs.permutation_product_poly += &rhs.permutation_product_poly;
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

    /// TODO: parallelize.
    fn mul(mut self, scalar: F) -> Self {
        for p in self.fixed_polys.iter_mut() {
            *p *= scalar;
        }
        for p in self.advice_polys.iter_mut() {
            *p *= scalar;
        }
        for p in self.instance_polys.iter_mut() {
            *p *= scalar;
        }
        for p in self.challenges.iter_mut() {
            *p *= scalar;
        }
        self.beta *= scalar;
        self.gamma *= scalar;
        self.theta *= scalar;
        self.y *= scalar;

        self
    }
}

/// A folding trace where, instead of field elements, we have polynomials.
/// It is represented as a vector of folding traces, where the i-th folding trace
/// represents the evaluation of the polynomial at the i-th domain point.
pub type LiftedFoldingTrace<F> = Vec<FoldingTrace<F>>;

/// Computes \sum_{j = 0}^k L_j(X) ω_j, where ω_j is the j-th trace,
/// for j = 0, ..., k. The `degree` is the maximum degree of the
/// constraint system.
///
/// TODO: Improve the memory peak that this function leads to.
/// We could handle each output folding trace one by one instead.
pub fn batch_traces<F: PrimeField + WithSmallOrderMulGroup<3>>(
    dk_domain: &EvaluationDomain<F>,
    traces: &[FoldingTrace<F>],
) -> LiftedFoldingTrace<F> {
    println!("Domain: {:?}", dk_domain);
    let lagrange_polys = (0..traces.len())
        .map(|i| {
            let mut l = dk_domain.empty_lagrange();
            l[i] = F::ONE;
            l
        })
        .map(|p| dk_domain.lagrange_to_coeff(p))
        .map(|p| dk_domain.coeff_to_extended(p))
        .collect::<Vec<_>>();

    let domain_size = lagrange_polys[0].num_coeffs();

    (0..domain_size)
        .map(|i| {
            let buffer = FoldingTrace::init(
                domain_size,
                traces[0].fixed_polys.len(),
                traces[0].advice_polys.len(),
                traces[0].instance_polys.len(),
                traces[0].lookups.len(),
                traces[0].permutation.sets.len(),
                traces[0].challenges.len(),
            );
            let coordinate_i_lagrange = lagrange_polys
                .iter()
                .map(|poly| poly[i])
                .collect::<Vec<_>>();

            linear_combination(buffer, traces, &coordinate_i_lagrange)
        })
        .collect()
}
