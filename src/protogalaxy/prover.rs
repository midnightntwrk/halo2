//! TODO
//!
#![allow(dead_code)]

use std::marker::PhantomData;

use crate::plonk::permutation;
use crate::utils::arithmetic::eval_polynomial;
use ff::{PrimeField, WithSmallOrderMulGroup};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

use crate::plonk::{
    traces::{LiftedFoldingTrace, Trace},
    Any, ConstraintSystem, Evaluator, ProvingKey, VerifyingKey,
};
use crate::poly::commitment::PolynomialCommitmentScheme;
use crate::poly::{
    Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
};
use crate::protogalaxy::utils::{linear_combination, pow_vec};
use crate::utils::arithmetic::parallelize;

/// PK used for folding. All traces being folded need to be valid for the same FoldingPk.
#[derive(Clone)]
pub(crate) struct FoldingPk<F: PrimeField> {
    domain: EvaluationDomain<F>,
    cs: ConstraintSystem<F>,
    l0: Polynomial<F, LagrangeCoeff>,
    l_last: Polynomial<F, LagrangeCoeff>,
    l_active_row: Polynomial<F, LagrangeCoeff>,
    // The following three were groupped in a type called permutation::ProverKey.
    // We prefix them here to avoid creating a new type.
    permutation_pk_permutations: Vec<Polynomial<F, LagrangeCoeff>>,
    permutation_pk_polys: Vec<Polynomial<F, Coeff>>,
    permutation_pk_cosets: Vec<Polynomial<F, LagrangeCoeff>>,
    ev: Evaluator<F>,
}

impl<F: PrimeField + WithSmallOrderMulGroup<3>> FoldingPk<F> {
    /// Given a FoldingPk, it takes the folded trace and returns a proving key. Concretely,
    /// it uses the folded fixed polys as the fixed polys for the proving key. The verifier
    /// should perform the same operation to compute its new verifying key.
    pub fn to_proving_key<CS: PolynomialCommitmentScheme<F>>(
        self,
        folded_trace: &Trace<F>,
        vk: &VerifyingKey<F, CS>,
    ) -> ProvingKey<F, CS> {
        let FoldingPk {
            domain,
            l0,
            l_last,
            l_active_row,
            permutation_pk_permutations,
            permutation_pk_polys,
            permutation_pk_cosets,
            ev,
            ..
        } = self;
        let lagrange_to_extended =
            |poly: Polynomial<F, LagrangeCoeff>| -> Polynomial<F, ExtendedLagrangeCoeff> {
                let tmp = domain.lagrange_to_coeff(poly);
                domain.coeff_to_extended(tmp)
            };

        let fixed_values = folded_trace.fixed_polys.clone();
        let fixed_polys = fixed_values
            .iter()
            .cloned()
            .map(|p| domain.lagrange_to_coeff(p))
            .collect::<Vec<_>>();
        let fixed_cosets = fixed_polys
            .iter()
            .cloned()
            .map(|p| domain.coeff_to_extended(p))
            .collect::<Vec<_>>();

        ProvingKey {
            vk: vk.clone(),
            l0: lagrange_to_extended(l0),
            l_last: lagrange_to_extended(l_last),
            l_active_row: lagrange_to_extended(l_active_row),
            fixed_values,
            fixed_polys,
            fixed_cosets,
            permutation: permutation::ProvingKey {
                permutations: permutation_pk_permutations,
                polys: permutation_pk_polys,
                cosets: permutation_pk_cosets
                    .into_iter()
                    .map(|poly| {
                        let poly = vk.domain.lagrange_to_coeff(poly);
                        vk.domain.coeff_to_extended(poly)
                    })
                    .collect(),
            },
            ev,
        }
    }
}

impl<F: PrimeField + WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>>
    From<ProvingKey<F, CS>> for FoldingPk<F>
{
    fn from(pk: ProvingKey<F, CS>) -> Self {
        let domain = pk.vk.get_domain().clone();
        let cs = pk.vk.cs;

        let mut l0 = domain.empty_lagrange();
        l0[0] = F::ONE;

        // Compute l_blind(X) which evaluates to 1 for each blinding factor row
        // and 0 otherwise over the domain.
        let mut l_blind = domain.empty_lagrange();
        for evaluation in l_blind[..].iter_mut().rev().take(cs.blinding_factors()) {
            *evaluation = F::ONE;
        }

        // Compute l_last(X) which evaluates to 1 on the first inactive row (just
        // before the blinding factors) and 0 otherwise over the domain
        let mut l_last = domain.empty_lagrange();
        let n = domain.n as usize;
        l_last[n - cs.blinding_factors() - 1] = F::ONE;

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
            permutation_pk_permutations: pk.permutation.permutations.clone(),
            permutation_pk_polys: pk.permutation.polys.clone(),
            permutation_pk_cosets: (pk.permutation.cosets.into_iter())
                .map(|poly| domain.extended_to_lagrange(poly))
                .collect(),
            ev: pk.ev,
            domain,
        }
    }
}

/// Computes f_{row_idx}(\sum_{j = 0}^k L_j(X) ω_j) for the `evaluator` polynomial
/// `f_{row_idx}` at row `row_idx`. The `evaluator` polynomial `f_{row_idx}` is the aggregation
/// (with `y`) of all custom gates, permutation and lookup identities.
///
/// The function receives `[Trace<F>]`, which contains the 'lifted' folding trace, i.e.,
/// `\sum_{j = 0}^k L_j(X) ω_j`. The size of `traces` corresponds to the extended domain `d * k`.
///
/// `eval_f` evaluates `f_{row_idx}` over the given folding trace.
///
// TODO: We should create a `FoldingCommonPk`, which is a structure that contains all
// necessary data from PKs, and one can generate it from several PKs:
pub(crate) fn eval_f_i<F>(pk: &FoldingPk<F>, row_idx: usize, trace: &Trace<F>) -> F
where
    F: PrimeField + WithSmallOrderMulGroup<3>,
{
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

    let Trace {
        fixed_polys,
        advice_polys,
        instance_values,
        lookups,
        permutation,
        challenges,
        beta,
        gamma,
        theta,
        y,
        ..
    } = trace;
    let mut value = pk.ev.custom_gates.evaluate(
        &mut eval_data,
        fixed_polys,
        advice_polys,
        instance_values,
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
        let delta_start = *beta; //* domain.g_coset;

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
        let r_last =
            crate::plonk::evaluation::get_rotation_idx(row_idx, last_rotation.0, rot_scale, isize);

        // Enforce only for the first set.
        // l_0(X) * (1 - z_0(X)) = 0
        value = value * y + ((one - first_set_permutation_product_coset[row_idx]) * l0[row_idx]);
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
            .zip(pk.permutation_pk_cosets.chunks(chunk_len))
        {
            let mut left = permutation_product_coset[r_next];
            for (values, permutation) in columns
                .iter()
                .map(|&column| match column.column_type() {
                    Any::Advice(_) => &advice_polys[column.index()],
                    Any::Fixed => &fixed_polys[column.index()],
                    Any::Instance => &instance_values[column.index()],
                })
                .zip(cosets.iter())
            {
                left *= values[row_idx] + *beta * permutation[row_idx] + gamma;
            }

            let mut right = permutation_product_coset[row_idx];
            for values in columns.iter().map(|&column| match column.column_type() {
                Any::Advice(_) => &advice_polys[column.index()],
                Any::Fixed => &fixed_polys[column.index()],
                Any::Instance => &instance_values[column.index()],
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
        let product = domain.coeff_to_lagrange(lookup.product_poly.clone());
        let permuted_input = domain.coeff_to_lagrange(lookup.permuted_input_poly.clone());
        let permuted_table = domain.coeff_to_lagrange(lookup.permuted_table_poly.clone());

        // Lookup constraints
        let lookup_evaluator = &pk.ev.lookups[n];
        let mut eval_data = lookup_evaluator.instance();
        let table_value = lookup_evaluator.evaluate(
            &mut eval_data,
            fixed_polys,
            advice_polys,
            instance_values,
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

        let a_minus_s = permuted_input[row_idx] - permuted_table[row_idx];
        // l_0(X) * (1 - z(X)) = 0
        value = value * y + ((one - product[row_idx]) * l0[row_idx]);
        // l_last(X) * (z(X)^2 - z(X)) = 0
        value = value * y
            + ((product[row_idx] * product[row_idx] - product[row_idx]) * l_last[row_idx]);
        // (1 - (l_last(X) + l_blind(X))) * (
        //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
        //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
        //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
        // ) = 0
        value = value * y
            + ((product[r_next]
                * (permuted_input[row_idx] + beta)
                * (permuted_table[row_idx] + gamma)
                - product[row_idx] * table_value)
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
                * (permuted_input[row_idx] - permuted_input[r_prev])
                * l_active_row[row_idx]);
    }

    value
}

/// Tasks to clean this up a bit:
/// We should have the function `create_proof` that is split into two:
/// * The prepare proof (which would give you the proof up to the point that folding requires it)
/// * The finalise proof (which takes a trace, and finalises the proof
///
/// We should also start a proper interface for folding. In this way, all instances (meaning all
/// challenges included) can be verified.
///
/// Then the proof verifier would:
/// * Parse the transcript and compute the different challenges.
/// * Perform the linear combination of challenges
/// * Perform the linear combinations of commitments of advice
/// * Perform the linear combination of fixed columns (from the VK)
/// * Verify the plonk proof with these new commitments
/// * Think about what we need to do with the error - unclear where we'll put it.

fn compute_poly_g<F: PrimeField + WithSmallOrderMulGroup<3>>(
    pk: &FoldingPk<F>,
    dk_domain: &EvaluationDomain<F>,
    beta: &[F],
    lifted_folding_trace: &LiftedFoldingTrace<F>,
) -> Polynomial<F, ExtendedLagrangeCoeff> {
    let beta_pows = pow_vec(beta);
    println!("beta_pows len: {:?}", beta_pows.len());
    println!("Lifted folding trace len: {:?}", lifted_folding_trace.len());

    let init = Polynomial::init(dk_domain.extended_len());

    let g_poly = beta_pows
        .par_iter()
        .enumerate()
        .fold(
            || init.clone(),
            |g, (i, beta_pow_i)| {
                let evals: Vec<F> = lifted_folding_trace
                    .par_iter()
                    .map(|trace| eval_f_i(pk, i, trace))
                    .collect();
                g + &(Polynomial {
                    values: evals,
                    _marker: PhantomData,
                } * *beta_pow_i)
            },
        )
        .reduce(|| init.clone(), |a, b| a + b);

    g_poly
}

/// Function to fold traces over an evaluation `\gamma`
fn fold_traces<F: PrimeField + WithSmallOrderMulGroup<3>>(
    dk_domain: &EvaluationDomain<F>,
    traces: &[&Trace<F>],
    gamma: &F,
) -> Trace<F> {
    let lagrange_polys = (0..traces.len())
        .map(|i| {
            let mut l = dk_domain.empty_lagrange();
            l[i] = F::ONE;
            l
        })
        .map(|p| dk_domain.lagrange_to_coeff(p))
        .collect::<Vec<_>>();

    let trace_domain_size = traces[0].fixed_polys[0].num_coeffs();

    let buffer = Trace::init(
        trace_domain_size,
        traces[0].fixed_polys.len(),
        traces[0].advice_polys.len(),
        traces[0].instance_polys.len(),
        traces[0].lookups.len(),
        traces[0].permutation.sets.len(),
        traces[0].challenges.len(),
    );
    let lagranges_in_gamma = lagrange_polys
        .iter()
        .map(|poly| eval_polynomial(poly, *gamma))
        .collect::<Vec<_>>();

    linear_combination(buffer, traces, &lagranges_in_gamma)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use blstrs::{Bls12, Scalar as Fp};
    use ff::Field;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_core::RngCore;

    use crate::circuit::{Layouter, SimpleFloorPlanner, Value};
    use crate::dev::MockProver;
    use crate::plonk::traces::batch_traces;
    use crate::plonk::{compute_trace, keygen_pk, keygen_vk_with_k, Advice, Circuit, Column, ConstraintSystem, Error, Expression, Selector, TableColumn, finalise_proof};
    use crate::poly::kzg::params::ParamsKZG;
    use crate::poly::kzg::KZGCommitmentScheme;
    use crate::poly::{EvaluationDomain, Rotation};
    use crate::protogalaxy::prover::{compute_poly_g, fold_traces, FoldingPk};
    use crate::transcript::{CircuitTranscript, Transcript};
    use crate::utils::arithmetic::eval_polynomial;

    #[derive(Clone, Copy)]
    struct TestCircuit {
        witness: [Value<Fp>; 1 << 8],
    }

    #[derive(Debug, Clone)]
    struct MyConfig {
        mul_selector: Selector,
        table_selector: Selector,
        table: TableColumn,
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
    }

    impl Circuit<Fp> for TestCircuit {
        type Config = MyConfig;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self {
                witness: [Value::unknown(); 1 << 8],
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fp>) -> MyConfig {
            let config = MyConfig {
                mul_selector: meta.complex_selector(),
                table_selector: meta.complex_selector(),
                table: meta.lookup_table_column(),
                a: meta.advice_column(),
                b: meta.advice_column(),
                c: meta.advice_column(),
            };

            meta.enable_equality(config.a);
            meta.enable_equality(config.b);

            meta.create_gate("mul", |meta| {
                let a = meta.query_advice(config.a, Rotation::cur());
                let b = meta.query_advice(config.b, Rotation::cur());
                let c = meta.query_advice(config.c, Rotation::cur());
                let q = meta.query_selector(config.mul_selector);
                vec![q * (a * b - c)]
            });

            meta.lookup("lookup", |meta| {
                let selector = meta.query_selector(config.table_selector);
                let not_selector = Expression::Constant(Fp::ONE) - selector.clone();

                let a = meta.query_advice(config.a, Rotation::cur());
                vec![(selector * a + not_selector, config.table)]
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
                        config.table_selector.enable(&mut region, offset as usize)?;
                        config.mul_selector.enable(&mut region, offset as usize)?;
                        let a = region.assign_advice(|| "a", config.a, offset, || val)?;
                        a.copy_advice(|| "copy a to b", &mut region, config.b, offset)?;
                        // region.assign_advice(|| "b", config.b, offset, || val)?;
                        region.assign_advice(|| "c", config.c, offset, || val.map(|v| v * v))?;
                    }

                    Ok(())
                },
            )?;

            Ok(())
        }
    }

    #[test]
    fn folding_test() {
        const K: u32 = 9;
        let k = 4;

        let rng = ChaCha8Rng::from_seed([0u8; 32]);
        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, rng);

        let mut rng = ChaCha8Rng::from_seed([0u8; 32]);
        let mut rand_bytes = [0u8; 1 << 8];
        rng.fill_bytes(&mut rand_bytes);

        let witness_1 = rand_bytes
            .into_iter()
            .map(|byte| Value::known(Fp::from((byte as u64) + 1)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let circuit1 = TestCircuit { witness: witness_1 };

        MockProver::run(10, &circuit1, vec![])
            .unwrap()
            .assert_satisfied();
        rng.fill_bytes(&mut rand_bytes);

        let witness_2 = rand_bytes
            .into_iter()
            .map(|byte| Value::known(Fp::from((byte as u64) + 1)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let circuit2 = TestCircuit { witness: witness_2 };

        rng.fill_bytes(&mut rand_bytes);

        let witness_3 = rand_bytes
            .into_iter()
            .map(|byte| Value::known(Fp::from((byte as u64) + 1)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let circuit3 = TestCircuit { witness: witness_3 };

        rng.fill_bytes(&mut rand_bytes);

        let witness_4 = rand_bytes
            .into_iter()
            .map(|byte| Value::known(Fp::from((byte as u64) + 1)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let circuit4 = TestCircuit { witness: witness_4 };

        let vk = keygen_vk_with_k::<_, KZGCommitmentScheme<Bls12>, _>(&params, &circuit1, K)
            .expect("keygen_vk should not fail");
        let pk = keygen_pk(vk.clone(), &circuit1).expect("keygen_pk should not fail");

        // Compute folding traces
        let now = Instant::now();
        let mut rng = ChaCha8Rng::from_seed([0u8; 32]);
        let mut transcript = CircuitTranscript::init();
        let folding_trace_1 =
            compute_trace(&params, &pk, &[circuit1], &[&[]], &mut rng, &mut transcript)
                .expect("Failed to compute the folding trace");

        let folding_trace_2 =
            compute_trace(&params, &pk, &[circuit2], &[&[]], &mut rng, &mut transcript)
                .expect("Failed to compute the folding trace");

        let folding_trace_3 =
            compute_trace(&params, &pk, &[circuit3], &[&[]], &mut rng, &mut transcript)
                .expect("Failed to compute the folding trace");

        let folding_trace_4 =
            compute_trace(&params, &pk, &[circuit3], &[&[]], &mut rng, &mut transcript)
                .expect("Failed to compute the folding trace");

        println!("Compute three traces: {:?}", now.elapsed().as_millis());

        let now = Instant::now();
        let degree = pk.vk.cs.degree() as u32;
        let k_log2_ceil = (k as f64 - 1.).log2() as u32 + 1;
        // We must increase the degree, since we need to count y as a variable.
        // Computing the real degree seems hard.
        let dk_domain = EvaluationDomain::new(degree + 3, k_log2_ceil);
        let folding_pk = FoldingPk::from(pk);

        let lifted_trace = batch_traces(
            &dk_domain,
            &[
                &folding_trace_1[0],
                &folding_trace_2[0],
                &folding_trace_3[0],
                &folding_trace_4[0],
            ],
        );
        println!("Batch three traces: {:?}", now.elapsed().as_millis());
        let now = Instant::now();

        let mut rng = ChaCha8Rng::from_seed([0u8; 32]);
        let mut betas = [Fp::ONE; K as usize];
        let mut beta_pow = Fp::random(&mut rng);
        for beta in betas.iter_mut() {
            *beta = beta_pow;
            beta_pow *= beta_pow
        }

        let poly_g = compute_poly_g(&folding_pk, &dk_domain, &betas, &lifted_trace);

        dbg!(&poly_g);

        println!("G poly: {:?}", now.elapsed().as_millis());

        let poly_k = dk_domain.divide_by_vanishing_poly(poly_g.clone());

        let gamma = Fp::random(&mut rng);

        let poly_k_coeff = dk_domain.extended_to_coeff(poly_k);

        // Final check. Eval G(X), K(X) and Z(X) in \gamma
        let g_in_gamma = dk_domain.eval_extended_lagrange(poly_g, gamma);
        let k_in_gamma = eval_polynomial(&poly_k_coeff, gamma);
        let z_in_gamma = gamma.pow_vartime(&[dk_domain.n]) - Fp::ONE;

        assert_eq!(g_in_gamma, k_in_gamma * z_in_gamma);

        let folded_trace = fold_traces(
            &dk_domain,
            &[
                &folding_trace_1[0],
                &folding_trace_2[0],
                &folding_trace_3[0],
                &folding_trace_4[0],
            ],
            &gamma,
        );

        let final_pk = folding_pk.to_proving_key(&folded_trace, &vk);
        finalise_proof(&params, &final_pk, &[folded_trace], &mut transcript).expect("Failed to finalise proof");


    }
}
