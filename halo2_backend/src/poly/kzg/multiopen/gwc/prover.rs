use super::{
    construct_intermediate_sets_zcash, ChallengeX1, ChallengeX2, ChallengeX3, ChallengeX4,
};
use crate::arithmetic::{eval_polynomial, kate_division, powers, truncate, truncated_powers};
use crate::helpers::SerdeCurveAffine;
use crate::poly::commitment::Prover;
use crate::poly::commitment::{ParamsProver, MSM};
use crate::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use crate::poly::query::ProverQuery;
use crate::poly::{commitment::Blind, Coeff, Polynomial};
use crate::transcript::{EncodedChallenge, TranscriptWrite};

use crate::poly::kzg::msm::{DualMSM, MSMKZG};
use crate::poly::kzg::strategy::GuardKZG;
use ff::Field;
use group::Curve;
use halo2_middleware::zal::traits::MsmAccel;
use halo2curves::pairing::{Engine, MultiMillerLoop};
use halo2curves::CurveExt;
use rand_core::RngCore;
use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;

/// Concrete KZG prover with GWC variant
#[derive(Debug)]
pub struct ProverGWC<'params, E: Engine> {
    params: &'params ParamsKZG<E>,
}

impl<'params, E: Engine + Debug> ProverGWC<'params, E> {
    fn inner_product(
        polys: &[Polynomial<E::Fr, Coeff>],
        scalars: impl Iterator<Item = E::Fr>,
    ) -> Polynomial<E::Fr, Coeff> {
        polys
            .iter()
            .zip(scalars)
            .map(|(p, s)| p.clone() * s)
            .reduce(|acc, p| acc + &p)
            .unwrap()
    }
}

/// Create a multi-opening proof
impl<'params, E: Engine + Debug> Prover<'params, KZGCommitmentScheme<E>> for ProverGWC<'params, E>
where
    E: MultiMillerLoop,
    E::G1Affine: SerdeCurveAffine<ScalarExt = <E as Engine>::Fr, CurveExt = <E as Engine>::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::G2Affine: SerdeCurveAffine,
    E::Fr: Ord,
{
    const QUERY_INSTANCE: bool = false;

    fn new(params: &'params ParamsKZG<E>) -> Self {
        Self { params }
    }

    /// Create a multi-opening proof
    fn create_proof_with_engine<
        'com,
        Ch: EncodedChallenge<E::G1Affine>,
        T: TranscriptWrite<E::G1Affine, Ch>,
        R,
        I,
    >(
        &self,
        engine: &impl MsmAccel<E::G1Affine>,
        _: R,
        transcript: &mut T,
        queries: I,
    ) -> io::Result<()>
    where
        I: IntoIterator<Item = ProverQuery<'com, E::G1Affine>> + Clone,
        R: RngCore,
    {
        // Refer to the halo2 book for docs:
        // https://zcash.github.io/halo2/design/proving-system/multipoint-opening.html
        let x1: ChallengeX1<_> = transcript.squeeze_challenge_scalar();
        let x2: ChallengeX2<_> = transcript.squeeze_challenge_scalar();

        let (poly_map, point_sets) = construct_intermediate_sets_zcash(queries);

        let mut q_polys = vec![vec![]; point_sets.len()];

        for com_data in poly_map.iter() {
            q_polys[com_data.set_index].push(com_data.commitment.poly.clone());
        }

        let q_polys = q_polys
            .iter()
            .map(|polys| Self::inner_product(polys, truncated_powers(*x1)))
            .collect::<Vec<_>>();
        let f_poly = {
            let f_polys = point_sets
                .iter()
                .zip(q_polys.clone())
                .map(|(points, q_poly)| {
                    let mut poly = points.iter().fold(q_poly.clone().values, |poly, point| {
                        kate_division(&poly, *point)
                    });
                    poly.resize(self.params.n as usize, E::Fr::ZERO);
                    Polynomial {
                        values: poly,
                        _marker: PhantomData,
                    }
                })
                .collect::<Vec<_>>();
            Self::inner_product(&f_polys, powers(*x2))
        };
        let f_com = self
            .params
            .commit(engine, &f_poly, Blind::default())
            .to_affine();
        transcript.write_point(f_com)?;
        let x3: ChallengeX3<_> = transcript.squeeze_challenge_scalar();
        let x3 = truncate(*x3);
        for q_poly in q_polys.iter() {
            transcript.write_scalar(eval_polynomial(q_poly.as_ref(), x3))?;
        }

        let x4: ChallengeX4<_> = transcript.squeeze_challenge_scalar();

        let final_poly = {
            let mut polys = q_polys;
            polys.push(f_poly);
            Self::inner_product(&polys, truncated_powers(*x4))
        };
        let v = eval_polynomial(&final_poly, x3);

        let pi = {
            let pi_poly = Polynomial {
                values: kate_division(&(&final_poly - v).values, x3),
                _marker: PhantomData,
            };
            self.params
                .commit(engine, &pi_poly, Blind::default())
                .to_affine()
        };

        transcript.write_point(pi)?;

        let final_poly_com = self.params.commit(engine, &final_poly, Blind::default());

        let mut msm_accumulator = DualMSM::new();

        // Scale commitment
        let mut commitment = MSMKZG::<E>::new();
        commitment.append_term(E::Fr::ONE, final_poly_com);

        let mut pi_msm = MSMKZG::<E>::new();
        pi_msm.append_term(E::Fr::ONE, pi.into());

        // Scale zπ
        let mut scaled_pi = MSMKZG::<E>::new();
        scaled_pi.append_term(x3, pi.into());

        // (π, C − vG + zπ)
        msm_accumulator.left.add_msm(&pi_msm);

        msm_accumulator.right.add_msm(&commitment); // C
        let g0: E::G1 = self.params.g[0].into();
        msm_accumulator.right.append_term(v, -g0); // -vG
        msm_accumulator.right.add_msm(&scaled_pi); // zπ

        // TODO: What is this doing here? :thinking_face:? Literally just copying from the commit for no
        Ok::<_, std::io::Error>(GuardKZG { msm_accumulator })?;

        Ok(())
    }
}
