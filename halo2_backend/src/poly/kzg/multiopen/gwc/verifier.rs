use std::fmt::Debug;
use std::marker::PhantomData;

use super::{
    construct_intermediate_sets_zcash, ChallengeX1, ChallengeX2, ChallengeX3, ChallengeX4,
};
use crate::arithmetic::{eval_polynomial, lagrange_interpolate};
use crate::arithmetic::{truncate, truncated_powers};
use crate::helpers::SerdeCurveAffine;
use crate::poly::commitment::Verifier;
use crate::poly::commitment::MSM;
use crate::poly::kzg::commitment::KZGCommitmentScheme;
use crate::poly::kzg::msm::{DualMSM, MSMKZG};
use crate::poly::kzg::strategy::GuardKZG;
use crate::poly::query::{CommitmentReference, VerifierQuery};
use crate::poly::Error;
use crate::transcript::{EncodedChallenge, TranscriptRead};

use ff::{Field, PrimeField};
use group::prime::PrimeCurveAffine;
use halo2curves::pairing::{Engine, MultiMillerLoop};
use halo2curves::CurveExt;

#[derive(Debug)]
/// Concrete KZG verifier with GWC variant
pub struct VerifierGWC<E: Engine> {
    _marker: PhantomData<E>,
}

fn msm_inner_product<E>(msms: &[MSMKZG<E>], scalars: impl Iterator<Item = E::Fr>) -> MSMKZG<E>
where
    E: MultiMillerLoop + Debug,
    E::G1Affine: SerdeCurveAffine<ScalarExt = <E as Engine>::Fr, CurveExt = <E as Engine>::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::G2Affine: SerdeCurveAffine,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::Fr: Ord,
{
    let mut res = MSMKZG::<E>::new();
    let mut msms = msms.to_vec();
    for (msm, s) in msms.iter_mut().zip(scalars) {
        msm.scale(s);
        res.add_msm(msm);
    }
    res
}

fn scalars_inner_product<F: PrimeField>(v1: &[F], scalars: impl Iterator<Item = F>) -> F {
    v1.iter()
        .zip(scalars)
        .map(|(s1, s2)| *s1 * s2)
        .reduce(|acc, s| acc + s)
        .unwrap()
}

/// Inter produc with truncated powers of the given x.
fn evals_inner_product<F: PrimeField + Clone>(
    evals_set: &[Vec<F>],
    scalars: impl Iterator<Item = F>,
) -> Vec<F> {
    let mut res = vec![F::ZERO; evals_set[0].len()];
    for (poly_evals, s) in evals_set.iter().zip(scalars) {
        for i in 0..res.len() {
            res[i] += poly_evals[i] * s;
        }
    }
    res
}

impl<'params, E> Verifier<'params, KZGCommitmentScheme<E>> for VerifierGWC<E>
where
    E: MultiMillerLoop + Debug,
    E::G1Affine: SerdeCurveAffine<ScalarExt = <E as Engine>::Fr, CurveExt = <E as Engine>::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::G2Affine: SerdeCurveAffine,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::Fr: Ord,
{
    type Guard = GuardKZG<E>;
    type MSMAccumulator = DualMSM<E>;

    const QUERY_INSTANCE: bool = false;

    fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    fn verify_proof<
        'com,
        Ch: EncodedChallenge<E::G1Affine>,
        T: TranscriptRead<E::G1Affine, Ch>,
        I,
    >(
        &self,
        transcript: &mut T,
        queries: I,
        mut msm_accumulator: DualMSM<E>,
    ) -> Result<Self::Guard, Error>
    where
        I: IntoIterator<Item = VerifierQuery<'com, E::G1Affine, MSMKZG<E>>> + Clone,
    {
        // Refer to the halo2 book for docs:
        // https://zcash.github.io/halo2/design/proving-system/multipoint-opening.html

        let x1: ChallengeX1<_> = transcript.squeeze_challenge_scalar();
        let x2: ChallengeX2<_> = transcript.squeeze_challenge_scalar();

        let (commitment_map, point_sets) = construct_intermediate_sets_zcash(queries);

        let mut q_coms: Vec<_> = vec![vec![]; point_sets.len()];
        let mut q_eval_sets = vec![vec![]; point_sets.len()];

        for com_data in commitment_map.into_iter() {
            let com_data_msm = match com_data.commitment {
                CommitmentReference::Commitment(c) => {
                    let mut msm = MSMKZG::new();
                    msm.append_term(E::Fr::ONE, (*c).into());
                    msm
                }
                CommitmentReference::MSM(msm) => msm.clone(),
            };
            q_coms[com_data.set_index].push(com_data_msm);
            q_eval_sets[com_data.set_index].push(com_data.evals);
        }

        let q_coms = q_coms
            .iter()
            .map(|msms| msm_inner_product(msms, truncated_powers(*x1)))
            .collect::<Vec<_>>();
        let q_eval_sets = q_eval_sets
            .iter()
            .map(|evals| evals_inner_product(evals, truncated_powers(*x1)))
            .collect::<Vec<_>>();

        let f_com = transcript.read_point().map_err(|_| Error::SamplingError)?;
        // Sample a challenge x_3 for checking that f(X) was committed to
        // correctly.
        let x3: ChallengeX3<_> = transcript.squeeze_challenge_scalar();
        let x3 = truncate(*x3);

        let mut q_evals_on_x3 = Vec::with_capacity(q_eval_sets.len());
        for _ in 0..q_eval_sets.len() {
            q_evals_on_x3.push(transcript.read_scalar().map_err(|_| Error::SamplingError)?);
        }

        // We can compute the expected msm_eval at x_3 using the u provided
        // by the prover and from x_2
        let f_eval = point_sets
            .iter()
            .zip(q_eval_sets.iter())
            .zip(q_evals_on_x3.iter())
            .rev()
            .fold(E::Fr::ZERO, |acc_eval, ((points, evals), proof_eval)| {
                let r_poly = lagrange_interpolate(points, evals);
                let r_eval = eval_polynomial(&r_poly, x3);
                let eval = points.iter().fold(*proof_eval - r_eval, |eval, point| {
                    eval * (x3 - point).invert().unwrap()
                });
                acc_eval * *x2 + eval
            });

        let x4: ChallengeX4<_> = transcript.squeeze_challenge_scalar();

        let final_com = {
            let mut polys = q_coms;
            let mut f_com_as_msm = MSMKZG::new();
            f_com_as_msm.append_term(E::Fr::ONE, f_com.into());
            polys.push(f_com_as_msm);
            msm_inner_product(&polys, truncated_powers(*x4))
        };

        let v = {
            let mut evals = q_evals_on_x3;
            evals.push(f_eval);
            scalars_inner_product(&evals, truncated_powers(*x4))
        };

        let pi = transcript.read_point().map_err(|_| Error::SamplingError)?;

        let mut pi_msm = MSMKZG::<E>::new();
        pi_msm.append_term(E::Fr::ONE, pi.into());

        // Scale zπ
        let mut scaled_pi = MSMKZG::<E>::new();
        scaled_pi.append_term(x3, pi.into());

        // (π, C − vG + zπ)
        msm_accumulator.left.add_msm(&pi_msm); // π

        msm_accumulator.right.add_msm(&scaled_pi); // zπ
        msm_accumulator.right.add_msm(&final_com); // C
        let g0: E::G1 = E::G1Affine::generator().into();
        msm_accumulator.right.append_term(v, -g0); // -vG

        Ok(Self::Guard::new(msm_accumulator))
    }
}
