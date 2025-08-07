use crate::plonk::{lookup, permutation, Any, ProvingKey};
use crate::plonk::permutation::prover::CommittedSet;
use crate::poly::commitment::PolynomialCommitmentScheme;
use crate::poly::Basis;
use crate::{
    poly::{Coeff, ExtendedLagrangeCoeff, Polynomial, Rotation},
    utils::arithmetic::parallelize,
};

use ff::{PrimeField, WithSmallOrderMulGroup};
use group::ff::Field;


use super::{ConstraintSystem, Expression};
use crate::{start_timer, end_timer};

/// Return the index in the polynomial of size `isize` after rotation `rot`.
fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}

/// Value used in a calculation
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub enum ValueSource {
    /// This is a constant value
    Constant(usize),
    /// This is an intermediate value
    Intermediate(usize),
    /// This is a fixed column
    Fixed(usize, usize),
    /// This is an advice (witness) column
    Advice(usize, usize),
    /// This is an instance (external) column
    Instance(usize, usize),
    /// This is a challenge
    Challenge(usize),
    /// beta
    Beta(),
    /// gamma
    Gamma(),
    /// theta
    Theta(),
    /// y
    Y(),
    /// Previous value
    PreviousValue(),
}

impl Default for ValueSource {
    fn default() -> Self {
        ValueSource::Constant(0)
    }
}

impl ValueSource {
    /// Get the value for this source
    #[allow(clippy::too_many_arguments)]
    pub fn get<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        match self {
            ValueSource::Constant(idx) => constants[*idx],
            ValueSource::Intermediate(idx) => intermediates[*idx],
            ValueSource::Fixed(column_index, rotation) => {
                fixed_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Advice(column_index, rotation) => {
                advice_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Instance(column_index, rotation) => {
                instance_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Challenge(index) => challenges[*index],
            ValueSource::Beta() => *beta,
            ValueSource::Gamma() => *gamma,
            ValueSource::Theta() => *theta,
            ValueSource::Y() => *y,
            ValueSource::PreviousValue() => *previous_value,
        }
    }

    pub fn to_ffi(&self) -> crate::ValueSourceFFI {
        match self {
            ValueSource::Constant(i) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Constant,
                param0: *i,
                param1: 0,
            },
            ValueSource::Intermediate(i) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Intermediate,
                param0: *i,
                param1: 0,
            },
            ValueSource::Fixed(col, rot) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Fixed,
                param0: *col,
                param1: *rot,
            },
            ValueSource::Advice(col, rot) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Advice,
                param0: *col,
                param1: *rot,
            },
            ValueSource::Instance(col, rot) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Instance,
                param0: *col,
                param1: *rot,
            },
            ValueSource::Challenge(i) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Challenge,
                param0: *i,
                param1: 0,
            },
            ValueSource::Beta() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Beta,
                param0: 0,
                param1: 0,
            },
            ValueSource::Gamma() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Gamma,
                param0: 0,
                param1: 0,
            },
            ValueSource::Theta() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Theta,
                param0: 0,
                param1: 0,
            },
            ValueSource::Y() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Y,
                param0: 0,
                param1: 0,
            },
            ValueSource::PreviousValue() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::PreviousValue,
                param0: 0,
                param1: 0,
            },
        }
    }

}

/// Calculation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Calculation {
    /// This is an addition
    Add(ValueSource, ValueSource),
    /// This is a subtraction
    Sub(ValueSource, ValueSource),
    /// This is a product
    Mul(ValueSource, ValueSource),
    /// This is a square
    Square(ValueSource),
    /// This is a double
    Double(ValueSource),
    /// This is a negation
    Negate(ValueSource),
    /// This is Horner's rule: `val = a; val = val * c + b[]`
    Horner(ValueSource, Vec<ValueSource>, ValueSource),
    /// This is a simple assignment
    Store(ValueSource),
}

impl Calculation {
    /// Get the resulting value of this calculation
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        let get_value = |value: &ValueSource| {
            value.get(
                rotations,
                constants,
                intermediates,
                fixed_values,
                advice_values,
                instance_values,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            )
        };
        match self {
            Calculation::Add(a, b) => get_value(a) + get_value(b),
            Calculation::Sub(a, b) => get_value(a) - get_value(b),
            Calculation::Mul(a, b) => get_value(a) * get_value(b),
            Calculation::Square(v) => get_value(v).square(),
            Calculation::Double(v) => get_value(v).double(),
            Calculation::Negate(v) => -get_value(v),
            Calculation::Horner(start_value, parts, factor) => {
                let factor = get_value(factor);
                let mut value = get_value(start_value);
                for part in parts.iter() {
                    value = value * factor + get_value(part);
                }
                value
            }
            Calculation::Store(v) => get_value(v),
        }
    }

    pub fn to_ffi(&self, arena: &mut Vec<crate::ValueSourceFFI>) -> crate::CalculationFFI {
        let dummy = crate::ValueSourceFFI {
            kind: crate::ValueSourceKind::Constant,
            param0: 0,
            param1: 0,
        };

        match self {
            Calculation::Add(a, b) => crate::CalculationFFI {
                kind: crate::CalculationKind::Add,
                a: a.to_ffi(),
                b: b.to_ffi(),
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Sub(a, b) => crate::CalculationFFI {
                kind: crate::CalculationKind::Sub,
                a: a.to_ffi(),
                b: b.to_ffi(),
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Mul(a, b) => crate::CalculationFFI {
                kind: crate::CalculationKind::Mul,
                a: a.to_ffi(),
                b: b.to_ffi(),
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Square(a) => crate::CalculationFFI {
                kind: crate::CalculationKind::Square,
                a: a.to_ffi(),
                b: dummy,
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Double(a) => crate::CalculationFFI {
                kind: crate::CalculationKind::Double,
                a: a.to_ffi(),
                b: dummy,
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Negate(a) => crate::CalculationFFI {
                kind: crate::CalculationKind::Negate,
                a: a.to_ffi(),
                b: dummy,
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Store(a) => crate::CalculationFFI {
                kind: crate::CalculationKind::Store,
                a: a.to_ffi(),
                b: dummy,
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Horner(start, parts, factor) => {
                let start_ffi = start.to_ffi();
                let factor_ffi = factor.to_ffi();
                let offset = arena.len();
                arena.extend(parts.iter().map(|p| p.to_ffi()));
                let ptr = arena[offset..].as_ptr();
                let len = parts.len();
                crate::CalculationFFI {
                    kind: crate::CalculationKind::Horner,
                    a: start_ffi,
                    b: dummy,
                    extra: factor_ffi,
                    horner_parts_ptr: ptr,
                    horner_parts_len: len,
                }
            }
        }
    }
}

/// Evaluator
#[derive(Clone, Default, Debug)]
pub struct Evaluator<F: PrimeField> {
    ///  Custom gates evalution
    pub custom_gates: GraphEvaluator<F>,
    ///  Lookups evalution
    pub lookups: Vec<GraphEvaluator<F>>,
}

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<F: PrimeField> {
    /// Constants
    pub constants: Vec<F>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
}

/// EvaluationData
#[derive(Default, Debug)]
pub struct EvaluationData<F: PrimeField> {
    /// Intermediates
    pub intermediates: Vec<F>,
    /// Rotations
    pub rotations: Vec<usize>,
}

/// CaluclationInfo
#[derive(Clone, Debug)]
pub struct CalculationInfo {
    /// Calculation
    pub calculation: Calculation,
    /// Target
    pub target: usize,
}

impl CalculationInfo {
    pub fn to_ffi(&self, arena: &mut Vec<crate::ValueSourceFFI>) -> crate::CalculationInfoFFI {
        crate::CalculationInfoFFI {
            calculation: self.calculation.to_ffi(arena),
            target: self.target,
        }
    }
}

// for fix size polys!
pub fn extract_inner_ptrs_of_poly<F, B>(polys: &[Polynomial<F, B>]) -> Vec<*const F> {
    polys.iter().map(|poly| poly.values.as_ptr()).collect()
}


pub fn extract_inner_ptrs_of_set_poly<F: PrimeField>(sets: &[CommittedSet<F>]) -> Vec<*const F> {
    sets.iter()
    .map(|set| {
        set.permutation_product_poly.values.as_ptr()}
    )
    .collect()
}

impl<F: WithSmallOrderMulGroup<3>> Evaluator<F> {
    /// Creates a new evaluation structure
    pub fn new(cs: &ConstraintSystem<F>) -> Self {
        let mut ev = Evaluator::default();

        // Custom gates
        let mut parts = Vec::new();
        for gate in cs.gates.iter() {
            parts.extend(
                gate.polynomials()
                    .iter()
                    .map(|poly| ev.custom_gates.add_expression(poly)),
            );
        }
        ev.custom_gates.add_calculation(Calculation::Horner(
            ValueSource::PreviousValue(),
            parts,
            ValueSource::Y(),
        ));

        // Lookups
        for lookup in cs.lookups.iter() {
            let mut graph = GraphEvaluator::default();

            let mut evaluate_lc = |expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Input coset
            let compressed_input_coset = evaluate_lc(&lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = graph.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Gamma(),
            ));
            let lc = graph.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Beta(),
            ));
            graph.add_calculation(Calculation::Mul(lc, right_gamma));

            ev.lookups.push(graph);
        }

        ev
    }

    /// Evaluate h poly
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn evaluate_h<CS: PolynomialCommitmentScheme<F>>(
        &self,
        pk: &ProvingKey<F, CS>,
        advice_polys: &[&[Polynomial<F, Coeff>]],
        instance_polys: &[&[Polynomial<F, Coeff>]],
        challenges: &[F],
        y: F,
        beta: F,
        gamma: F,
        theta: F,
        lookups: &[Vec<lookup::prover::Committed<F>>],
        permutations: &[permutation::prover::Committed<F>],
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        let domain = &pk.vk.domain;
        let size = domain.extended_len();
        let rot_scale = 1 << (domain.extended_k() - domain.k());
        let fixed = &pk.fixed_cosets[..];
        let extended_omega = domain.get_extended_omega();
        let isize = size as i32;
        let one = F::ONE;
        let l0 = &pk.l0;
        let l_last = &pk.l_last;
        let l_active_row = &pk.l_active_row;
        let p = &pk.vk.cs.permutation;

        let mut values = domain.empty_extended();

        // Core expression evaluations
        for (((advice, instance), lookups), permutation) in advice_polys
            .iter()
            .zip(instance_polys.iter())
            .zip(lookups.iter())
            .zip(permutations.iter())
        {   
            // GPU Implementation of "Core expression evaluations"
            let mut arena = Vec::new();
            let ffi_structs: Vec<crate::CalculationInfoFFI> = self.custom_gates.calculations
            .iter()
            .map(|c| c.to_ffi(&mut arena))
            .collect();
    
            let advice_poly_ptr = extract_inner_ptrs_of_poly(advice);
            let instance_poly_ptr = extract_inner_ptrs_of_poly(instance);
            let fixed_poly_ptr = extract_inner_ptrs_of_poly(fixed);
            
            let rotation_rot: Vec<i32> = self.custom_gates.rotations.iter().map(|rot| *rot).collect();

            let g_coset_value = pk.vk.domain.g_coset;
            let g_coset_inv_value: F = g_coset_value.square(); 

            let small_sizex = advice[0].values.len() as i32;

            let sets = &permutation.sets;
            let round1_flag = if sets.is_empty() { 1 } else { 0 };
            let round2_flag = if lookups.is_empty() { 1 } else { 0 };

            crate::custom_gates_evaluation_r(&ffi_structs, &fixed_poly_ptr, &advice_poly_ptr, &instance_poly_ptr,
            challenges, &beta, &gamma, &theta, &y, &mut values.values, &self.custom_gates.constants, &rotation_rot, &rot_scale, &isize,
            &l0.values, &l_last.values, &l_active_row.values, &g_coset_value, &g_coset_inv_value, &small_sizex, &round1_flag);

            // Permutations
            
            if !sets.is_empty() {
                let blinding_factors = pk.vk.cs.blinding_factors();
                let last_rotation = Rotation(-((blinding_factors + 1) as i32));
                let chunk_len = pk.vk.cs.degree() - 2;
                let delta_start = beta * &pk.vk.domain.g_coset;
             
                // Permutation constraints GPU SIDE
                //let permutation_product_cosets_ptr = extract_inner_ptrs_of_poly(&permutation_product_cosets);
                let permutation_product_poly = extract_inner_ptrs_of_set_poly(&sets);
                let pk_cosets_ptr = extract_inner_ptrs_of_poly(&pk.permutation.cosets);                
                let columns_ffi_structs: Vec<crate::ColumnFFI> = p.columns
                .iter()
                .map(|c| c.to_ffi())
                .collect();
                
                let chunk_len32: i32 = chunk_len as i32;
                let small_size = sets[0].permutation_product_poly.values.len() as i32;

                crate::permutations_evaluation_r(&columns_ffi_structs, &permutation_product_poly, &pk_cosets_ptr, &advice_poly_ptr, &instance_poly_ptr, &fixed_poly_ptr,
                    &mut values.values ,&l0.values, &l_last.values, &l_active_row.values,
                    &delta_start, &F::DELTA, &beta, &gamma, &y, &extended_omega, &chunk_len32, &last_rotation.0,
                    &rot_scale, &isize, &g_coset_value, &g_coset_inv_value, &small_size, &round2_flag);
            }

            // Lookups
            for (n, lookup) in lookups.iter().enumerate() {
                let round3_flag = if n == lookups.len() - 1 { 1 } else { 0 };
                
                let mut arena_lookups = Vec::with_capacity(1024);
                let ffi_structs_lookups: Vec<crate::CalculationInfoFFI> = self.lookups[n].calculations
                .iter()
                .map(|c| c.to_ffi(&mut arena_lookups))
                .collect();

                let rotation_rot_lookups: Vec<i32> = self.lookups[n].rotations.iter().map(|rot| *rot).collect();

                crate::lookups_evaluation_r(&ffi_structs_lookups,
                challenges, &beta, &gamma, &theta, &y, &mut values.values, &self.lookups[n].constants, &rotation_rot_lookups,
                 &rot_scale, &isize ,&lookup.product_poly.values, &lookup.permuted_input_poly.values, &lookup.permuted_table_poly.values,
                &g_coset_value, &g_coset_inv_value, &round3_flag);
            }
        }
        values
    }
}

impl<F: PrimeField> Default for GraphEvaluator<F> {
    fn default() -> Self {
        Self {
            // Fixed positions to allow easy access
            constants: vec![F::ZERO, F::ONE, F::from(2u64)],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
        }
    }
}

impl<F: PrimeField> GraphEvaluator<F> {
    /// Adds a rotation
    fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    fn add_constant(&mut self, constant: &F) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused  when that calculation
    /// is done multiple times.
    fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let existing_calculation = self
            .calculations
            .iter()
            .find(|c| c.calculation == calculation);
        match existing_calculation {
            Some(existing_calculation) => ValueSource::Intermediate(existing_calculation.target),
            None => {
                let target = self.num_intermediates;
                self.calculations.push(CalculationInfo {
                    calculation,
                    target,
                });
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    /// Generates an optimized evaluation for the expression
    fn add_expression(&mut self, expr: &Expression<F>) -> ValueSource {
        match expr {
            Expression::Constant(scalar) => self.add_constant(scalar),
            Expression::Selector(_selector) => unreachable!(),
            Expression::Fixed(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Fixed(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Advice(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Advice(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Instance(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Instance(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Challenge(challenge) => self.add_calculation(Calculation::Store(
                ValueSource::Challenge(challenge.index()),
            )),
            Expression::Negated(a) => match **a {
                Expression::Constant(scalar) => self.add_constant(&-scalar),
                _ => {
                    let result_a = self.add_expression(a);
                    match result_a {
                        ValueSource::Constant(0) => result_a,
                        _ => self.add_calculation(Calculation::Negate(result_a)),
                    }
                }
            },
            Expression::Sum(a, b) => {
                // Undo subtraction stored as a + (-b) in expressions
                match &**b {
                    Expression::Negated(b_int) => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b_int);
                        if result_a == ValueSource::Constant(0) {
                            self.add_calculation(Calculation::Negate(result_b))
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else {
                            self.add_calculation(Calculation::Sub(result_a, result_b))
                        }
                    }
                    _ => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else if result_a <= result_b {
                            self.add_calculation(Calculation::Add(result_a, result_b))
                        } else {
                            self.add_calculation(Calculation::Add(result_b, result_a))
                        }
                    }
                }
            }
            Expression::Product(a, b) => {
                let result_a = self.add_expression(a);
                let result_b = self.add_expression(b);
                if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                    ValueSource::Constant(0)
                } else if result_a == ValueSource::Constant(1) {
                    result_b
                } else if result_b == ValueSource::Constant(1) {
                    result_a
                } else if result_a == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_b))
                } else if result_b == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_a))
                } else if result_a == result_b {
                    self.add_calculation(Calculation::Square(result_a))
                } else if result_a <= result_b {
                    self.add_calculation(Calculation::Mul(result_a, result_b))
                } else {
                    self.add_calculation(Calculation::Mul(result_b, result_a))
                }
            }
            Expression::Scaled(a, f) => {
                if *f == F::ZERO {
                    ValueSource::Constant(0)
                } else if *f == F::ONE {
                    self.add_expression(a)
                } else {
                    let cst = self.add_constant(f);
                    let result_a = self.add_expression(a);
                    self.add_calculation(Calculation::Mul(result_a, cst))
                }
            }
        }
    }

    /// Creates a new evaluation structure
    pub fn instance(&self) -> EvaluationData<F> {
        EvaluationData {
            intermediates: vec![F::ZERO; self.num_intermediates],
            rotations: vec![0usize; self.rotations.len()],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<B: Basis>(
        &self,
        data: &mut EvaluationData<F>,
        fixed: &[Polynomial<F, B>],
        advice: &[Polynomial<F, B>],
        instance: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
        idx: usize,
        rot_scale: i32,
        isize: i32,
    ) -> F {
        // All rotation index values
        for (rot_idx, rot) in self.rotations.iter().enumerate() {
            data.rotations[rot_idx] = get_rotation_idx(idx, *rot, rot_scale, isize);
        }
        
        // All calculations, with cached intermediate results
        for calc in self.calculations.iter() {
            data.intermediates[calc.target] = calc.calculation.evaluate(
                &data.rotations,
                &self.constants,
                &data.intermediates,
                fixed,
                advice,
                instance,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            );
        }

        // Return the result of the last calculation (if any)
        if let Some(calc) = self.calculations.last() {
            data.intermediates[calc.target]
        } else {
            F::ZERO
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_gpu_test<B: Basis>(
        &self,
        rotation_input: &[i32],
        calculations: &[CalculationInfo],
        data: &mut EvaluationData<F>,
        fixed: &[Polynomial<F, B>],
        advice: &[Polynomial<F, B>],
        instance: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
        idx: usize,
        rot_scale: i32,
        isize: i32,
    ) -> F {
        // All rotation index values
        let mut rot_new: Vec<usize> = Vec::new();
        for (rot_idx, rot) in rotation_input.iter().enumerate() {
            //data.rotations[rot_idx] = get_rotation_idx(idx, *rot, rot_scale, isize);
            rot_new.push(get_rotation_idx(idx, *rot, rot_scale, isize));
        }
        
        // All calculations, with cached intermediate results
        //for calc in self.calculations.iter() {
        for calc in calculations.iter() {
            data.intermediates[calc.target] = calc.calculation.evaluate(
                //&data.rotations,
                &rot_new,
                &self.constants,
                &data.intermediates,
                fixed,
                advice,
                instance,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            );
        }

        // Return the result of the last calculation (if any)
        if let Some(calc) = calculations.last() {
            data.intermediates[calc.target]
        } else {
            F::ZERO
        }
    }
}

/// Simple evaluation of an expression
pub fn evaluate<F: Field, B: Basis>(
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
) -> Vec<F> {
    let mut values = vec![F::ZERO; size];
    let isize = size as i32;
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            *value = expression.evaluate(
                &|scalar| scalar,
                &|_| panic!("virtual selectors are removed during optimization"),
                &|query| {
                    fixed[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    advice[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    instance[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|challenge| challenges[challenge.index()],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            );
        }
    });
    values
}
