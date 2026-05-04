//! True radix-4 Cooley-Tukey kernels.
//!
//! ## Algorithm
//!
//! In-place DIT radix-4 FFT:
//! 1. Digit-reverse permutation (base 4).
//! 2. Stage loop with `len = 4, 16, 64, …` up to `N`:
//!    - For each chunk of `len` elements, for each `j in 0..len/4`:
//!      - Apply twiddle factors `W_len^j`, `W_len^{2j}`, `W_len^{3j}`.
//!      - Combine via the radix-4 butterfly:
//!        `t0=a0+a2, t1=a0-a2, t2=a1+a3, t3=a1-a3`.
//!        Forward: `y0=t0+t2, y1=t1-i·t3, y2=t0-t2, y3=t1+i·t3`.
//!        Inverse: `y0=t0+t2, y1=t1+i·t3, y2=t0-t2, y3=t1-i·t3`.
//!
//! The `±i` rotation is `rot_neg_i`/`rot_pos_i` from `WinogradComplex` — zero
//! multiplications, one swap and one negate.
//!
//! Both `Complex64` and `Complex32` monomorphize through the same
//! `radix4_inplace<C>` function; no algorithm body is duplicated.
//!
//! ## References
//!
//! Cooley, J.W. & Tukey, J.W. (1965). *Math. Comput.* 19(90), 297–301.

use super::kernel_api::radix_kernel_api;
use super::radix_permute::digit_reverse_permute_pow2_radix;
use super::radix_shape::{is_power_of_four, stage_twiddle};
use super::radix_stage::WinogradComplex;
use super::radix2_f16::Cf16;
use num_complex::{Complex32, Complex64};

radix_kernel_api! {
    check       = is_power_of_four,
    inplace64   = radix4_inplace_64,
    inplace32   = radix4_inplace_32,
    description = "power-of-four",
}

/// Generic in-place radix-4 DIT Cooley-Tukey FFT.
///
/// Monomorphizes for `C = Complex64` and `C = Complex32` with zero overhead.
/// The `±i` twiddle rotation in the butterfly combine uses `WinogradComplex::rot_neg_i`
/// / `rot_pos_i` — zero multiplications via a field-swap and negation.
///
/// ## Correctness
///
/// The radix-4 butterfly satisfies the DFT factorization:
/// `X[k] = E[k] + W_N^k · O[k]` for `k ∈ {0,1,2,3}` (Van Loan 1992, §3.1).
#[inline]
fn radix4_inplace<C>(data: &mut [C], inverse: bool, twiddles: Option<&[C]>)
where
    C: WinogradComplex + std::ops::Mul<Output = C> + std::ops::Neg<Output = C>,
{
    // All public entry points supply Some(twiddles); None is unreachable in the API.
    let twiddles = twiddles.unwrap();
    debug_assert!(is_power_of_four(data.len()));
    if data.len() <= 1 {
        return;
    }

    digit_reverse_permute_pow2_radix::<4, _>(data);

    let n = data.len();
    let mut len = 4usize;
    while len <= n {
        let quarter = len >> 2;
        let half    = len >> 1;
        let stage   = if len > 4 {
            Some(&twiddles[(half - 1)..(half - 1 + half)])
        } else {
            None
        };

        for chunk in data.chunks_exact_mut(len) {
            for j in 0..quarter {
                let i0 = j;
                let i1 = i0 + quarter;
                let i2 = i1 + quarter;
                let i3 = i2 + quarter;

                let a0 = chunk[i0];
                let mut a1 = chunk[i1];
                let mut a2 = chunk[i2];
                let mut a3 = chunk[i3];

                if let Some(st) = stage {
                    a1 = a1 * stage_twiddle(st, half, j);
                    a2 = a2 * stage_twiddle(st, half, 2 * j);
                    a3 = a3 * stage_twiddle(st, half, 3 * j);
                }

                let t0 = a0 + a2;
                let t1 = a0 - a2;
                let t2 = a1 + a3;
                let t3 = a1 - a3;

                // y1 = t1 − i·t3 (forward) or t1 + i·t3 (inverse)
                // y3 = t1 + i·t3 (forward) or t1 − i·t3 (inverse)
                // rot_neg_i: (re,im) → (im,−re)  (multiply by −i)
                // rot_pos_i: (re,im) → (−im, re) (multiply by +i)
                let (y1, y3) = if inverse {
                    (t1 + t3.rot_pos_i(), t1 - t3.rot_pos_i())
                } else {
                    (t1 + t3.rot_neg_i(), t1 - t3.rot_neg_i())
                };

                chunk[i0] = t0 + t2;
                chunk[i1] = y1;
                chunk[i2] = t0 - t2;
                chunk[i3] = y3;
            }
        }

        len <<= 2;
    }
}

// Thin monomorphizing shims — zero overhead after inlining.

#[inline(always)]
fn radix4_inplace_64(data: &mut [Complex64], inverse: bool, twiddles: Option<&[Complex64]>) {
    radix4_inplace(data, inverse, twiddles);
}

#[inline(always)]
fn radix4_inplace_32(data: &mut [Complex32], inverse: bool, twiddles: Option<&[Complex32]>) {
    radix4_inplace(data, inverse, twiddles);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};
    use super::super::test_utils::max_abs_err_64;

    #[test]
    fn radix4_forward_n16_matches_direct() {
        let n = 16usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.3).sin(), (k as f64 * 0.11).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix4 forward mismatch err={err:.2e}");
    }

    #[test]
    fn radix4_inverse_unnorm_n16_matches_direct() {
        let n = 16usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.27).cos(), (k as f64 * 0.17).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix4 inverse mismatch err={err:.2e}");
    }
}
