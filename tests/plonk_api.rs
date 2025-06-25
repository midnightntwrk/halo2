#[macro_use]
extern crate criterion;

use group::ff::Field;
use halo2_proofs::circuit::{Cell, Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::plonk::*;
use halo2_proofs::poly::Rotation;
use halo2curves::bn256;
//use rand_core::OsRng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use std::marker::PhantomData;

use criterion::{BenchmarkId, Criterion};
use halo2_proofs::poly::commitment::Guard;
use halo2_proofs::poly::kzg::params::ParamsVerifierKZG;
use halo2_proofs::poly::kzg::{params::ParamsKZG, KZGCommitmentScheme};
use halo2_proofs::transcript::{CircuitTranscript, Transcript};
use halo2_proofs::utils::rational::Rational;
//use halo2curves::bn256::bn256;

use blstrs::{Bls12, Scalar as Fr, G1Affine};
use halo2_proofs::{start_timer, end_timer};
#[derive(Clone)]
struct PlonkConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    sa: Column<Fixed>,
    sb: Column<Fixed>,
    sc: Column<Fixed>,
    sm: Column<Fixed>,
    p: Column<Instance>,
}

trait StandardCs<FF: Field> {
    fn raw_multiply<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        f: F,
    ) -> Result<(Cell, Cell, Cell), Error>
    where
        F: FnMut() -> Value<(Rational<FF>, Rational<FF>, Rational<FF>)>;

    fn raw_add<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        f: F,
    ) -> Result<(Cell, Cell, Cell), Error>
    where
        F: FnMut() -> Value<(Rational<FF>, Rational<FF>, Rational<FF>)>;

    fn copy(&self, layouter: &mut impl Layouter<FF>, a: Cell, b: Cell) -> Result<(), Error>;
}

#[derive(Clone)]
struct MyCircuit<F: Field> {
    inputs: Vec<Value<F>>,
}

struct StandardPlonk<F: Field> {
    config: PlonkConfig,
    _marker: PhantomData<F>,
}

impl<FF: Field> StandardPlonk<FF> {
    fn new(config: PlonkConfig) -> Self {
        StandardPlonk { config, _marker: PhantomData }
    }
}

impl<FF: Field> StandardCs<FF> for StandardPlonk<FF> {
    fn raw_multiply<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        mut f: F,
    ) -> Result<(Cell, Cell, Cell), Error>
    where
        F: FnMut() -> Value<(Rational<FF>, Rational<FF>, Rational<FF>)>,
    {
        layouter.assign_region(
            || "raw_multiply",
            |mut region| {
                let mut value = None;
                let lhs = region.assign_advice(
                    || "lhs",
                    self.config.a,
                    0,
                    || {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    },
                )?;
                let rhs = region.assign_advice(
                    || "rhs",
                    self.config.b,
                    0,
                    || value.unwrap().map(|v| v.1),
                )?;
                let out = region.assign_advice(
                    || "out",
                    self.config.c,
                    0,
                    || value.unwrap().map(|v| v.2),
                )?;
                // selectors for multiplication
                region.assign_fixed(|| "a_sel", self.config.sa, 0, || Value::known(FF::ZERO))?;
                region.assign_fixed(|| "b_sel", self.config.sb, 0, || Value::known(FF::ZERO))?;
                region.assign_fixed(|| "c_sel", self.config.sc, 0, || Value::known(FF::ONE))?;
                region.assign_fixed(|| "m_sel", self.config.sm, 0, || Value::known(FF::ONE))?;
                Ok((lhs.cell(), rhs.cell(), out.cell()))
            },
        )
    }

    fn raw_add<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        mut f: F,
    ) -> Result<(Cell, Cell, Cell), Error>
    where
        F: FnMut() -> Value<(Rational<FF>, Rational<FF>, Rational<FF>)>,
    {
        layouter.assign_region(
            || "raw_add",
            |mut region| {
                let mut value = None;
                let lhs = region.assign_advice(
                    || "lhs",
                    self.config.a,
                    0,
                    || {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    },
                )?;
                let rhs = region.assign_advice(
                    || "rhs",
                    self.config.b,
                    0,
                    || value.unwrap().map(|v| v.1),
                )?;
                let out = region.assign_advice(
                    || "out",
                    self.config.c,
                    0,
                    || value.unwrap().map(|v| v.2),
                )?;
                // selectors for addition
                region.assign_fixed(|| "a_sel", self.config.sa, 0, || Value::known(FF::ONE))?;
                region.assign_fixed(|| "b_sel", self.config.sb, 0, || Value::known(FF::ONE))?;
                region.assign_fixed(|| "c_sel", self.config.sc, 0, || Value::known(FF::ONE))?;
                region.assign_fixed(|| "m_sel", self.config.sm, 0, || Value::known(FF::ZERO))?;
                Ok((lhs.cell(), rhs.cell(), out.cell()))
            },
        )
    }

    fn copy(
        &self,
        layouter: &mut impl Layouter<FF>,
        left: Cell,
        right: Cell,
    ) -> Result<(), Error> {
        layouter.assign_region(|| "copy", |mut region| region.constrain_equal(left, right))
    }
}

impl<F: Field> Circuit<F> for MyCircuit<F> {
    type Config = PlonkConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        MyCircuit { inputs: vec![Value::unknown(); self.inputs.len()] }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> PlonkConfig {
        meta.set_minimum_degree(5);
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(c);

        let sa = meta.fixed_column();
        let sb = meta.fixed_column();
        let sc = meta.fixed_column();
        let sm = meta.fixed_column();

        let p = meta.instance_column();
        meta.enable_equality(p);

        meta.create_gate("Combined add-mult", |meta| {
            let a_q = meta.query_advice(a, Rotation::cur());
            let b_q = meta.query_advice(b, Rotation::cur());
            let c_q = meta.query_advice(c, Rotation::cur());
            let sa_q = meta.query_fixed(sa, Rotation::cur());
            let sb_q = meta.query_fixed(sb, Rotation::cur());
            let sc_q = meta.query_fixed(sc, Rotation::cur());
            let sm_q = meta.query_fixed(sm, Rotation::cur());
            vec![a_q.clone() * sa_q + b_q.clone() * sb_q + a_q * b_q * sm_q - (c_q * sc_q)]
        });

        PlonkConfig { a, b, c, sa, sb, sc, sm, p }
    }

    fn synthesize(
        &self,
        config: PlonkConfig,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let cs = StandardPlonk::new(config.clone());
        for (row, input) in self.inputs.iter().enumerate() {
            let a: Value<Rational<_>> = (*input).into();
            let mut a_squared = Value::unknown();
            let (a0, _, c0) = cs.raw_multiply(&mut layouter, || {
                a_squared = a.square();
                a.zip(a_squared).map(|(a, a2)| (a, a, a2))
            })?;
            let (a1, b1, _) = cs.raw_add(&mut layouter, || {
                let fin = a_squared + a;
                a.zip(a_squared)
                    .zip(fin)
                    .map(|((a, a2), fin)| (a, a2, fin))
            })?;
            cs.copy(&mut layouter, a0, a1)?;
            cs.copy(&mut layouter, b1, c0)?;

            // Constrain `c0` to the public input at row `row`
            layouter.constrain_instance(c0, config.p, row)?;
        }
        Ok(())
    }
}

// helper macro for public instance value
macro_rules! common {
    ($field:ident) => {{
        $field::ONE + $field::ONE
    }};
}

fn keygen(
    k: u32,
) -> (
    ParamsKZG<Bls12>,
    ProvingKey<Fr, KZGCommitmentScheme<Bls12>>,
) {
    let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(k, ChaCha20Rng::from_entropy());
    let empty_circuit: MyCircuit<Fr> = MyCircuit { inputs: vec![] };

    //let mut rng = ChaCha20Rng::from_entropy();
    //let circuit: MyCircuit<Fr> = MyCircuit {inputs: vec![
    //    Value::known(Fr::random(&mut rng)),
    //    Value::known(Fr::random(&mut rng)),
    //    Value::known(Fr::random(&mut rng)),
    //    Value::known(Fr::random(&mut rng)),
    //] };
    
    let vk = keygen_vk_with_k(&params, &empty_circuit, k).expect("keygen_vk should not fail");
    let pk = keygen_pk(vk, &empty_circuit).expect("keygen_pk should not fail");
    (params, pk)
}

fn prover(
    k: u32,
    params: &ParamsKZG<Bls12>,
    pk: &ProvingKey<Fr, KZGCommitmentScheme<Bls12>>,
) -> Vec<u8> {
    let mut rng = ChaCha20Rng::from_entropy();

    // Generate multiple inputs
    let num_inputs = 4;
    let inputs: Vec<Value<Fr>> = (0..num_inputs)
        .map(|_| Value::known(Fr::random(&mut rng)))
        .collect();
    let circuit = MyCircuit { inputs };

    let mut transcript = CircuitTranscript::init();

    // Prepare public instances (one per input)
    let public_instances: Vec<Fr> = (0..num_inputs).map(|_| common!(Fr)).collect();

    create_proof::<Fr, KZGCommitmentScheme<Bls12>, _, _>(
        params,
        pk,
        &[circuit],
        &[&[&public_instances[..]]],
        rng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    transcript.finalize()
}

fn verifier(
    params: &ParamsVerifierKZG<Bls12>,
    vk: &VerifyingKey<Fr, KZGCommitmentScheme<Bls12>>,
    proof: &[u8],
) {
    let num_inputs = 4;
    let public_instances: Vec<Fr> = (0..num_inputs).map(|_| common!(Fr)).collect();
    let mut transcript = CircuitTranscript::init_from_bytes(proof);
    assert!(prepare::<Fr, KZGCommitmentScheme<Bls12>, _>(
        vk,
        &[&[&public_instances[..]]],
        &mut transcript,
    )
    .unwrap()
    .verify(params)
    .is_ok());
}

#[test]
fn plonk_t_new() {
    let k_range = 12;

    start_timer!(KEYGEN);
    let (params, pk) = keygen(k_range);
    end_timer!(KEYGEN, "KEYGEN");

    start_timer!(PROVE);
    let proof = prover(k_range, &params, &pk);
    end_timer!(PROVE, "PROOF");

    start_timer!(VERIFICATION);
    verifier(&params.verifier_params(), &pk.get_vk(), &proof[..]);
    end_timer!(VERIFICATION, "VERIFICATION");
}
