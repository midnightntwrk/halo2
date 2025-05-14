//! TODO
//!
#![allow(dead_code)]

use crate::circuit::Value;
use crate::plonk::{
    lookup, permutation, sealed, Advice, Any, Assignment, Challenge, Circuit, Column,
    ConstraintSystem, Error, Evaluator, Fixed, FloorPlanner, Instance, ProvingKey, Selector,
};
use crate::poly::commitment::PolynomialCommitmentScheme;
use crate::poly::{batch_invert_rational, Basis, EvaluationDomain, LagrangeCoeff, Polynomial, Rotation, ExtendedLagrangeCoeff};
use crate::protogalaxy::traces::{FoldingTrace, LiftedFoldingTrace};
use crate::transcript::Hashable;
use crate::transcript::Sampleable;
use crate::transcript::Transcript;
use crate::utils::arithmetic::parallelize;
use crate::utils::rational::Rational;
use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use rand_core::{CryptoRng, RngCore};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::RangeTo;
use crate::protogalaxy::utils::pow_vec;

/// PK used for folding. All traces being folded need to be valid for the same FoldingPk.
pub(crate) struct FoldingPk<F: PrimeField> {
    domain: EvaluationDomain<F>,
    cs: ConstraintSystem<F>,
    l0: Polynomial<F, LagrangeCoeff>,
    l_last: Polynomial<F, LagrangeCoeff>,
    l_active_row: Polynomial<F, LagrangeCoeff>,
    permutation: permutation::ProvingKey<F>,
    ev: Evaluator<F>,
}

impl<F: PrimeField + WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>>
    From<ProvingKey<F, CS>> for FoldingPk<F>
{
    fn from(pk: ProvingKey<F, CS>) -> Self {
        let domain = pk.vk.get_domain().clone();
        let cs = pk.vk.cs;

        let mut l0 = domain.empty_lagrange();
        l0[0] = F::ONE;

        // Compute l_last(X) which evaluates to 1 on the first inactive row (just
        // before the blinding factors) and 0 otherwise over the domain
        let mut l_last = domain.empty_lagrange();
        let n = domain.n as usize;
        l_last[n - cs.blinding_factors() - 1] = F::ONE;

        // Compute l_blind(X) which evaluates to 1 for each blinding factor row
        // and 0 otherwise over the domain.
        let mut l_blind = domain.empty_lagrange();
        for evaluation in l_blind[..].iter_mut().rev().take(cs.blinding_factors()) {
            *evaluation = F::ONE;
        }

        let mut l_active_row = domain.empty_lagrange();
        parallelize(&mut l_active_row, |values, start| {
            for (i, value) in values.iter_mut().enumerate() {
                let idx = i + start;
                *value = F::ONE - (l_last[idx] + l_blind[idx]);
            }
        });

        Self {
            cs,
            l0,
            l_last,
            l_active_row,
            permutation: pk.permutation,
            ev: pk.ev,
            domain,
        }
    }
}

/// Computes f_{row_idx}(\sum_{j = 0}^k L_j(X) ω_j) for the `evaluator` polynomial
/// `f_{row_idx}` at row `row_idx`. The `evaluator` polynomial `f_{row_idx}` is the aggregation
/// (with `y`) of all custom gates, permutation and lookup identities.
///
/// The function receives `[FoldingTrace<F>]`, which contains the 'lifted' folding trace, i.e.,
/// `\sum_{j = 0}^k L_j(X) ω_j`. The size of `traces` corresponds to the extended domain `d*k`.
///
/// `eval_f` evaluates `f_{row_idx}` over each of the folding traces.
///
/// Returns a vector of size `d * n`
// TODO: We may have a problem with identities that depend on more than one row.
// TODO: We should create a `FoldingCommonPk`, which is a structure that contains all
// necessary data from PKs, and one can generate it from several PKs:
pub(crate) fn eval_f_i<F>(
    pk: &FoldingPk<F>,
    row_idx: usize,
    traces: &LiftedFoldingTrace<F>,
) -> Polynomial<F, ExtendedLagrangeCoeff>
where
    F: PrimeField + WithSmallOrderMulGroup<3>,
{
    let mut res = Vec::with_capacity(traces.len());
    let domain = pk.domain.clone();
    let size = domain.n;
    // let rot_scale = 1 << (domain.extended_k() - domain.k());
    let rot_scale = 1 << 0;
    let omega = domain.get_omega();
    let isize = size as i32;
    let l0 = &pk.l0;
    let l_last = &pk.l_last;
    let l_active_row = &pk.l_active_row;
    let one = F::ONE;
    let p = &pk.cs.permutation;
    let mut eval_data = pk.ev.custom_gates.instance();
    for trace in traces {
        let FoldingTrace {
            fixed_polys,
            advice_polys,
            instance_polys,
            lookups,
            permutation,
            challenges,
            beta,
            gamma,
            theta,
            y,
        } = trace;
        let mut value = pk.ev.custom_gates.evaluate(
            &mut eval_data,
            fixed_polys,
            advice_polys,
            instance_polys,
            challenges,
            &beta,
            &gamma,
            &theta,
            &y,
            &F::ZERO,
            row_idx,
            rot_scale,
            isize,
        );

        // Permutations
        let sets = &permutation.sets;
        if !sets.is_empty() {
            let blinding_factors = pk.cs.blinding_factors();
            let last_rotation = Rotation(-((blinding_factors + 1) as i32));
            let chunk_len = pk.cs.degree() - 2;
            let delta_start = domain.g_coset * beta;

            let permutation_product_cosets: Vec<Polynomial<F, LagrangeCoeff>> = sets
                .iter()
                .map(|set| domain.coeff_to_lagrange(set.permutation_product_poly.clone()))
                .collect();

            let first_set_permutation_product_coset = permutation_product_cosets.first().unwrap();
            let last_set_permutation_product_coset = permutation_product_cosets.last().unwrap();

            // Permutation constraints
            // TODO: careful with this term - might be introducing a bug
            let beta_term = omega.pow_vartime([row_idx as u64, 0, 0, 0]);
            let r_next = crate::plonk::evaluation::get_rotation_idx(row_idx, 1, rot_scale, isize);
            let r_last = crate::plonk::evaluation::get_rotation_idx(
                row_idx,
                last_rotation.0,
                rot_scale,
                isize,
            );

            // Enforce only for the first set.
            // l_0(X) * (1 - z_0(X)) = 0
            value =
                value * y + ((one - first_set_permutation_product_coset[row_idx]) * l0[row_idx]);
            // Enforce only for the last set.
            // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
            value = value * y
                + ((last_set_permutation_product_coset[row_idx]
                    * last_set_permutation_product_coset[row_idx]
                    - last_set_permutation_product_coset[row_idx])
                    * l_last[row_idx]);
            // Except for the first set, enforce.
            // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
            for set_idx in 0..sets.len() {
                if set_idx != 0 {
                    value = value * y
                        + ((permutation_product_cosets[set_idx][row_idx]
                            - permutation_product_cosets[set_idx - 1][r_last])
                            * l0[row_idx]);
                }
            }
            // And for all the sets we enforce:
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
            // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
            // )
            let mut current_delta = delta_start * beta_term;
            for ((permutation_product_coset, columns), cosets) in permutation_product_cosets
                .iter()
                .zip(p.columns.chunks(chunk_len))
                .zip(pk.permutation.cosets.chunks(chunk_len))
            {
                let mut left = permutation_product_coset[r_next];
                for (values, permutation) in columns
                    .iter()
                    .map(|&column| match column.column_type() {
                        Any::Advice(_) => &advice_polys[column.index()],
                        Any::Fixed => &fixed_polys[column.index()],
                        Any::Instance => &instance_polys[column.index()],
                    })
                    .zip(cosets.iter())
                {
                    left *= values[row_idx] + permutation[row_idx] * beta + gamma;
                }

                let mut right = permutation_product_coset[row_idx];
                for values in columns.iter().map(|&column| match column.column_type() {
                    Any::Advice(_) => &advice_polys[column.index()],
                    Any::Fixed => &fixed_polys[column.index()],
                    Any::Instance => &instance_polys[column.index()],
                }) {
                    right *= values[row_idx] + current_delta + gamma;
                    current_delta *= &F::DELTA;
                }

                value = value * y + ((left - right) * l_active_row[row_idx]);
            }
        }

        // Lookups
        for (n, lookup) in lookups.iter().enumerate() {
            // Polynomials required for this lookup.
            // Calculated here so these only have to be kept in memory for the short time
            // they are actually needed.
            let product_coset = domain.coeff_to_extended(lookup.product_poly.clone());
            let permuted_input_coset = domain.coeff_to_extended(lookup.permuted_input_poly.clone());
            let permuted_table_coset = domain.coeff_to_extended(lookup.permuted_table_poly.clone());

            // Lookup constraints
            let lookup_evaluator = &pk.ev.lookups[n];
            let mut eval_data = lookup_evaluator.instance();
            let table_value = lookup_evaluator.evaluate(
                &mut eval_data,
                fixed_polys,
                advice_polys,
                instance_polys,
                challenges,
                &beta,
                &gamma,
                &theta,
                &y,
                &F::ZERO,
                row_idx,
                rot_scale,
                isize,
            );

            let r_next = crate::plonk::evaluation::get_rotation_idx(row_idx, 1, rot_scale, isize);
            let r_prev = crate::plonk::evaluation::get_rotation_idx(row_idx, -1, rot_scale, isize);

            let a_minus_s = permuted_input_coset[row_idx] - permuted_table_coset[row_idx];
            // l_0(X) * (1 - z(X)) = 0
            value = value * y + ((one - product_coset[row_idx]) * l0[row_idx]);
            // l_last(X) * (z(X)^2 - z(X)) = 0
            value = value * y
                + ((product_coset[row_idx] * product_coset[row_idx] - product_coset[row_idx])
                    * l_last[row_idx]);
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
            //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            // ) = 0
            value = value * y
                + ((product_coset[r_next]
                    * (permuted_input_coset[row_idx] + beta)
                    * (permuted_table_coset[row_idx] + gamma)
                    - product_coset[row_idx] * table_value)
                    * l_active_row[row_idx]);
            // Check that the first values in the permuted input expression and permuted
            // fixed expression are the same.
            // l_0(X) * (a'(X) - s'(X)) = 0
            value = value * y + (a_minus_s * l0[row_idx]);
            // Check that each value in the permuted lookup input expression is either
            // equal to the value above it, or the value at the same index in the
            // permuted table expression.
            // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
            value = value * y
                + (a_minus_s
                    * (permuted_input_coset[row_idx] - permuted_input_coset[r_prev])
                    * l_active_row[row_idx]);
        }

        res.push(value);
    }

    Polynomial {
        values: res,
        _marker: Default::default(),
    }
}

/// This creates part of a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The partial result is used to fold
/// several proofs together.
pub(crate) fn create_folding_trace<
    F,
    CS: PolynomialCommitmentScheme<F>,
    T: Transcript,
    ConcreteCircuit: Circuit<F>,
>(
    params: &CS::Parameters,
    pk: &ProvingKey<F, CS>,
    circuit: &ConcreteCircuit,
    instances: &[&[F]],
    mut rng: impl RngCore + CryptoRng,
    transcript: &mut T,
) -> Result<FoldingTrace<F>, Error>
where
    CS::Commitment: Hashable<T::Hash>,
    F: WithSmallOrderMulGroup<3>
        + Sampleable<T::Hash>
        + Hashable<T::Hash>
        + Ord
        + FromUniformBytes<64>,
{
    if instances.len() != pk.vk.cs.num_instance_columns {
        return Err(Error::InvalidInstances);
    }

    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;

    let domain = &pk.vk.domain;
    let mut meta = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut meta, circuit.params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut meta);

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &pk.vk.cs;

    struct InstanceSingle<F: PrimeField> {
        pub instance_values: Vec<Polynomial<F, LagrangeCoeff>>,
    }

    let instance: InstanceSingle<F> = {
        let instance_values = instances
            .iter()
            .map(|values| {
                let mut poly = domain.empty_lagrange();
                assert_eq!(poly.len(), domain.n as usize);
                if values.len() > (poly.len() - (meta.blinding_factors() + 1)) {
                    return Err(Error::InstanceTooLarge);
                }
                for (poly, value) in poly.iter_mut().zip(values.iter()) {
                    transcript.common(value)?;
                    *poly = *value;
                }
                Ok(poly)
            })
            .collect::<Result<Vec<_>, _>>()?;

        InstanceSingle {
            instance_values,
        }
    };

    #[derive(Clone)]
    struct AdviceSingle<F: PrimeField, B: Basis> {
        pub advice_polys: Vec<Polynomial<F, B>>,
    }

    struct WitnessCollection<'a, F: Field> {
        k: u32,
        current_phase: sealed::Phase,
        advice: Vec<Polynomial<Rational<F>, LagrangeCoeff>>,
        unblinded_advice: HashSet<usize>,
        challenges: &'a HashMap<usize, F>,
        instances: &'a [&'a [F]],
        usable_rows: RangeTo<usize>,
        _marker: std::marker::PhantomData<F>,
    }

    impl<'a, F: Field> Assignment<F> for WitnessCollection<'a, F> {
        fn enter_region<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about regions in this context.
        }

        fn exit_region(&mut self) {
            // Do nothing; we don't care about regions in this context.
        }

        fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Do nothing
        }

        fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            self.instances
                .get(column.index())
                .and_then(|column| column.get(row))
                .map(|v| Value::known(*v))
                .ok_or(Error::BoundsFailure)
        }

        fn assign_advice<V, VR, A, AR>(
            &mut self,
            _: A,
            column: Column<Advice>,
            row: usize,
            to: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Rational<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Ignore assignment of advice column in different phase than current one.
            if self.current_phase != column.column_type().phase {
                return Ok(());
            }

            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            *self
                .advice
                .get_mut(column.index())
                .and_then(|v| v.get_mut(row))
                .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

            Ok(())
        }

        fn assign_fixed<V, VR, A, AR>(
            &mut self,
            _: A,
            _: Column<Fixed>,
            _: usize,
            _: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Rational<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn copy(
            &mut self,
            _: Column<Any>,
            _: usize,
            _: Column<Any>,
            _: usize,
        ) -> Result<(), Error> {
            // We only care about advice columns here

            Ok(())
        }

        fn fill_from_row(
            &mut self,
            _: Column<Fixed>,
            _: usize,
            _: Value<Rational<F>>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn get_challenge(&self, challenge: Challenge) -> Value<F> {
            self.challenges
                .get(&challenge.index())
                .cloned()
                .map(Value::known)
                .unwrap_or_else(Value::unknown)
        }

        fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about namespaces in this context.
        }

        fn pop_namespace(&mut self, _: Option<String>) {
            // Do nothing; we don't care about namespaces in this context.
        }
    }

    let (advice, challenges) = {
        let mut advice = AdviceSingle::<F, LagrangeCoeff> {
            advice_polys: vec![domain.empty_lagrange(); meta.num_advice_columns],
        };
        let mut challenges = HashMap::<usize, F>::with_capacity(meta.num_challenges);

        let unusable_rows_start = domain.n as usize - (meta.blinding_factors() + 1);
        for current_phase in pk.vk.cs.phases() {
            let column_indices = meta
                .advice_column_phase
                .iter()
                .enumerate()
                .filter_map(|(column_index, phase)| {
                    if current_phase == *phase {
                        Some(column_index)
                    } else {
                        None
                    }
                })
                .collect::<BTreeSet<_>>();

            let mut witness = WitnessCollection {
                k: domain.k(),
                current_phase,
                advice: vec![domain.empty_lagrange_rational(); meta.num_advice_columns],
                unblinded_advice: HashSet::from_iter(meta.unblinded_advice_columns.clone()),
                instances,
                challenges: &challenges,
                // The prover will not be allowed to assign values to advice
                // cells that exist within inactive rows, which include some
                // number of blinding factors and an extra row for use in the
                // permutation argument.
                usable_rows: ..unusable_rows_start,
                _marker: std::marker::PhantomData,
            };

            // Synthesize the circuit to obtain the witness and other information.
            ConcreteCircuit::FloorPlanner::synthesize(
                &mut witness,
                circuit,
                config.clone(),
                meta.constants.clone(),
            )?;

            let mut advice_values = batch_invert_rational::<F>(
                witness
                    .advice
                    .into_iter()
                    .enumerate()
                    .filter_map(|(column_index, advice)| {
                        if column_indices.contains(&column_index) {
                            Some(advice)
                        } else {
                            None
                        }
                    })
                    .collect(),
            );

            for (column_index, advice_values) in column_indices.iter().zip(&mut advice_values) {
                if !witness.unblinded_advice.contains(column_index) {
                    for cell in &mut advice_values[unusable_rows_start..] {
                        *cell = F::random(&mut rng);
                    }
                } else {
                    #[cfg(feature = "sanity-checks")]
                    for cell in &advice_values[unusable_rows_start..] {
                        assert_eq!(*cell, F::ZERO);
                    }
                }
            }

            let advice_commitments: Vec<_> = advice_values
                .iter()
                .map(|poly| CS::commit_lagrange(params, poly))
                .collect();

            for commitment in &advice_commitments {
                transcript.write(commitment)?;
            }
            for (column_index, advice_values) in column_indices.iter().zip(advice_values) {
                advice.advice_polys[*column_index] = advice_values;
            }

            for (index, phase) in meta.challenge_phase.iter().enumerate() {
                if current_phase == *phase {
                    let existing = challenges.insert(index, transcript.squeeze_challenge());
                    assert!(existing.is_none());
                }
            }
        }

        assert_eq!(challenges.len(), meta.num_challenges);
        let challenges = (0..meta.num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice, challenges)
    };

    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: F = transcript.squeeze_challenge();

    let lookups: Vec<lookup::prover::Permuted<F>> =
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.commit_permuted(
                        pk,
                        params,
                        domain,
                        theta,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect::<Result<Vec<_>, Error>>()?;

    // Sample beta challenge
    let beta: F = transcript.squeeze_challenge();

    // Sample gamma challenge
    let gamma: F = transcript.squeeze_challenge();

    // Commit to permutations.
    let permutation: permutation::prover::Committed<F> = pk.vk.cs.permutation.commit(
        params,
        pk,
        &pk.permutation,
        &advice.advice_polys,
        &pk.fixed_values,
        &instance.instance_values,
        beta,
        gamma,
        &mut rng,
        transcript,
    )?;

    let lookups: Vec<lookup::prover::Committed<F>> =
            // Construct and commit to products for each lookup
            lookups
                .into_iter()
                .map(|lookup| lookup.commit_product(pk, params, beta, gamma, &mut rng, transcript))
                .collect::<Result<Vec<_>, _>>()?;

    // Obtain challenge for keeping all separate gates linearly independent
    let y: F = transcript.squeeze_challenge();

    Ok(FoldingTrace {
        fixed_polys: pk.fixed_values.clone(),
        advice_polys: advice.advice_polys,
        instance_polys: instance.instance_values,
        lookups,
        permutation,
        challenges,
        beta,
        gamma,
        theta,
        y,
    })
}

fn compute_poly_g<F: PrimeField + WithSmallOrderMulGroup<3>>(pk: &FoldingPk<F>, dk_domain: &EvaluationDomain<F>, beta: &[F], lifted_folding_trace: &LiftedFoldingTrace<F>) -> Polynomial<F, ExtendedLagrangeCoeff> {
    let beta_pows = pow_vec(beta);

    let mut g_poly = Polynomial::init(dk_domain.extended_len());

    beta_pows.iter().enumerate().for_each(|(i, beta_pow_i)| {
        g_poly += &(eval_f_i(pk, i, lifted_folding_trace) * *beta_pow_i)
    });

    g_poly
}

#[cfg(test)]
mod tests {
    use blstrs::{Bls12, Scalar as Fp};
    use ff::Field;
    use rand_core::{OsRng, RngCore};

    use crate::circuit::{Layouter, SimpleFloorPlanner, Value};
    use crate::dev::MockProver;
    use crate::plonk::{
        keygen_pk, keygen_vk_with_k, Advice, Circuit, Column, ConstraintSystem, Error, Expression,
        Selector, TableColumn,
    };
    use crate::poly::kzg::params::ParamsKZG;
    use crate::poly::kzg::KZGCommitmentScheme;
    use crate::poly::{EvaluationDomain, Rotation};
    use crate::protogalaxy::prover::{compute_poly_g, create_folding_trace, FoldingPk};
    use crate::protogalaxy::traces::batch_traces;
    use crate::transcript::{CircuitTranscript, Transcript};
    use crate::utils::arithmetic::eval_polynomial;

    #[derive(Clone, Copy)]
    struct TestCircuit {
        witness: [Value<Fp>; 1 << 10],
    }

    #[derive(Debug, Clone)]
    struct MyConfig {
        selector: Selector,
        table: TableColumn,
        advice: Column<Advice>,
    }

    impl Circuit<Fp> for TestCircuit {
        type Config = MyConfig;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self {
                witness: [Value::unknown(); 1 << 10],
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fp>) -> MyConfig {
            let config = MyConfig {
                selector: meta.complex_selector(),
                table: meta.lookup_table_column(),
                advice: meta.advice_column(),
            };

            meta.lookup("lookup", |meta| {
                let selector = meta.query_selector(config.selector);
                let not_selector = Expression::Constant(Fp::ONE) - selector.clone();
                let advice = meta.query_advice(config.advice, Rotation::cur());
                vec![(selector * advice + not_selector, config.table)]
            });

            config
        }

        fn synthesize(
            &self,
            config: MyConfig,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            layouter.assign_table(
                || "8-bit table",
                |mut table| {
                    for row in 0u64..(1 << 8) {
                        table.assign_cell(
                            || format!("row {row}"),
                            config.table,
                            row as usize,
                            || Value::known(Fp::from(row + 1)),
                        )?;
                    }

                    Ok(())
                },
            )?;

            layouter.assign_region(
                || "assign values",
                |mut region| {
                    for (offset, val) in self.witness.into_iter().enumerate() {
                        config.selector.enable(&mut region, offset as usize)?;
                        region.assign_advice(
                            || format!("offset {offset}"),
                            config.advice,
                            offset,
                            || val,
                        )?;
                    }

                    Ok(())
                },
            )
        }
    }

    #[test]
    fn folding_test() {
        const K: u32 = 11;
        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);

        let mut rand_bytes = [0u8; 1 << 10];
        OsRng.fill_bytes(&mut rand_bytes);

        let witness_1 = rand_bytes
            .into_iter()
            .map(|byte| Value::known(Fp::from((byte as u64) + 1)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let circuit1 = TestCircuit { witness: witness_1 };

        OsRng.fill_bytes(&mut rand_bytes);

        let witness_2 = rand_bytes
            .into_iter()
            .map(|byte| Value::known(Fp::from((byte as u64) + 1)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let circuit2 = TestCircuit { witness: witness_2 };

        OsRng.fill_bytes(&mut rand_bytes);

        let witness_3 = rand_bytes
            .into_iter()
            .map(|byte| Value::known(Fp::from((byte as u64) + 1)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let circuit3 = TestCircuit { witness: witness_3 };

        let vk = keygen_vk_with_k::<_, KZGCommitmentScheme<Bls12>, _>(&params, &circuit1, K)
            .expect("keygen_vk should not fail");
        let pk = keygen_pk(vk, &circuit1).expect("keygen_pk should not fail");

        MockProver::run(K, &circuit1, vec![]).unwrap().assert_satisfied();
        let mut transcript_1 = CircuitTranscript::init();
        let folding_trace_1 =
            create_folding_trace(&params, &pk, &circuit1, &[], OsRng, &mut transcript_1)
                .expect("Failed to compute the folding trace");

        let mut transcript_2 = CircuitTranscript::init();
        let folding_trace_2 =
            create_folding_trace(&params, &pk, &circuit2, &[], OsRng, &mut transcript_2)
                .expect("Failed to compute the folding trace");

        let mut transcript_3 = CircuitTranscript::init();
        let folding_trace_3 =
            create_folding_trace(&params, &pk, &circuit3, &[], OsRng, &mut transcript_3)
                .expect("Failed to compute the folding trace");

        let degree = pk.vk.cs.degree() as u32;
        let dk_domain = EvaluationDomain::new(degree, 3);
        let folding_pk = FoldingPk::from(pk);

        let lifted_trace = batch_traces(
            &dk_domain,
            &[folding_trace_1, folding_trace_2, folding_trace_3],
        );

        let betas = [Fp::ONE; K as usize];
        let poly_g = compute_poly_g(&folding_pk, &dk_domain, &betas, &lifted_trace);

        let poly_g_coeff = dk_domain.extended_to_coeff(poly_g);

        for exponent in 0..degree * 3 {
            let res = eval_polynomial(&poly_g_coeff, dk_domain.get_omega().pow_vartime(&[exponent as u64]));
            assert_eq!(res, Fp::ZERO);
        }
        // let poly_k = domain.divide_by_vanishing_poly(poly_g);
        //
        // domain
    }
}
