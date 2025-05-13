//! TODO
//!

// Computes f_i(\sum_{j = 0}^k L_j(X) ω_j) for all the `evaluator` polynomials
// `f_i` at row `row_idx`. Here, ω_j is the j-th trace, for j = 0, ..., k and
// the lagrange polynomials `L_j(X)` are given in evaluations form (in an
// extended `domain` that is compatible with the degree introduced by the `evaluator`).
//
// TODO: We may have a problem with identities that depend on more than one row.
// fn eval_f_i<F>(
//     row_idx: usize,
//     evaluator: &Evaluator<F>,
//     lagrange_polys: &[Polynomial<F, ExtendedLagrangeCoeff>],
//     traces: &[&FoldingTrace<F>],
//     domain: &EvaluationDomain<F>,
// ) -> F
// where
//     F: PrimeField + WithSmallOrderMulGroup<3>,
// {
//     let size = domain.extended_len();
//     let rot_scale = 1 << (domain.extended_k() - domain.k());
//     let isize = size as i32;
//     let mut eval_data = evaluator.custom_gates.instance();
//     let mut values = domain.empty_extended();
//     for (i, value) in values.iter_mut().enumerate() {
//         *value = evaluator.custom_gates.evaluate(
//             &mut eval_data,
//             fixed,
//             advice,
//             instance,
//             challenges,
//             &beta,
//             &gamma,
//             &theta,
//             &y,
//             value,
//             i,
//             rot_scale,
//             isize,
//         );
//     }
//     todo!()
// }

// Computes L_i(X) * trace,
// fn mul_trace_by_ith_lagrange(
//     i: usize,
// )

// for i in rows {
//     f_i(w_row_i * L_0(X) + sum_j wj_row_i * L_j(X))
// }
