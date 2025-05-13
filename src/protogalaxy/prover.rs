//! TODO
//!

use crate::plonk::{permutation, Any, ConstraintSystem, Evaluator, ProvingKey};
use crate::poly::{
    Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
};
use crate::protogalaxy::traces::{FoldingTrace, LiftedFoldingTrace};
use crate::utils::arithmetic::parallelize;
use ff::{PrimeField, WithSmallOrderMulGroup};

/// PK used for folding. All traces being folded need to be valid for the same FoldingPk.
struct FoldingPk<F: PrimeField> {
    domain: EvaluationDomain<F>,
    cs: ConstraintSystem<F>,
    l0: Polynomial<F, LagrangeCoeff>,
    l_last: Polynomial<F, LagrangeCoeff>,
    l_active_row: Polynomial<F, LagrangeCoeff>,
    permutation: permutation::ProvingKey<F>,
    ev: Evaluator<F>,
}

/// Computes f_{row_idx}(\sum_{j = 0}^k L_j(X) ω_j) for the `evaluator` polynomial
/// `f_{row_idx}` at row `row_idx`. The `evaluator` polynomial `f_{row_idx}` is the aggregation
/// (with `y`) of all custom gates, permutation and lookup identities.
///
/// The function receives `[FoldingTrace<F>]`, which contains the 'lifted' folding trace, i.e.,
/// `\sum_{j = 0}^k L_j(X) ω_j`. The size of `traces` corresponds to the extended domain `d*k`.
///
/// `eval_f` evaluates `f_{row_idx}` over each of the folding traces.
/// `domain` is the evaluation domain over which the advice, instance, and fixed columns are sent.
///
/// Returns a vector of size `d * n`
// TODO: We may have a problem with identities that depend on more than one row.
// TODO: We should create a `FoldingCommonPk`, which is a structure that contains all
// necessary data from PKs, and one can generate it from several PKs:
fn eval_f_i<F>(
    pk: FoldingPk<F>,
    row_idx: usize,
    traces: &LiftedFoldingTrace<F>,
    domain: &EvaluationDomain<F>,
) -> Vec<F>
where
    F: PrimeField + WithSmallOrderMulGroup<3>,
{
    let mut res = Vec::with_capacity(traces.len());
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
            permutations,
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
        let sets = &permutations.sets;
        if !sets.is_empty() {
            let blinding_factors = pk.cs.blinding_factors();
            let last_rotation = Rotation(-((blinding_factors + 1) as i32));
            let chunk_len = pk.cs.degree() - 2;
            let delta_start = pk.domain.g_coset * beta;

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
            let product_coset = pk.domain.coeff_to_extended(lookup.product_poly.clone());
            let permuted_input_coset = pk
                .domain
                .coeff_to_extended(lookup.permuted_input_poly.clone());
            let permuted_table_coset = pk
                .domain
                .coeff_to_extended(lookup.permuted_table_poly.clone());

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
    res
}
