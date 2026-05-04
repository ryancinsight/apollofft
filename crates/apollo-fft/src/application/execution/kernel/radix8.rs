//! Radix-8 DIT FFT using Winograd DFT-8 inner butterflies.
//!
//! ## Algorithm
//!
//! For an N-point transform where N is a power of 8, we perform an iterative
//! DIT in stages of radix 8.  Each stage processes groups of length `len = 8^s`
//! (for stage s = 1, 2, …, log₈ N):
//!
//! 1. Apply the digit-reverse permutation (base-8) once up front.
//! 2. For each stage, partition the data into non-overlapping groups of `len`
//!    elements.  Within each group, apply inter-group twiddle factors
//!    `W_len^{k·j}` to the non-first sub-blocks, then call the Winograd DFT-8
//!    kernel on each set of 8 elements separated by stride `len/8`.
//!
//! The Winograd DFT-8 kernel requires only 4 real multiplications (the
//! `±√2/2` factors on the odd path) and 26 real additions, versus 64
//! multiplications for the generic DFT-matrix approach.
//!
//! ## References
//!
//! - Winograd, S. (1978). On computing the discrete Fourier transform.
//!   *Mathematics of Computation*, 32(141), 175-199.

use super::kernel_api::radix_kernel_api;
use super::radix_shape::is_power_of_eight;
use super::radix_stage;
use super::{winograd};
use super::radix2_f16::Cf16;
use num_complex::{Complex32, Complex64};

// ── core Winograd-radix-8 in-place kernel ─────────────────────────────────────

fn winograd_r8_inplace_64(data: &mut [Complex64], inverse: bool, twiddles: Option<&[Complex64]>) {
    debug_assert!(is_power_of_eight(data.len()));
    radix_stage::radix_winograd_inplace::<8, Complex64, _>(
        data,
        inverse,
        twiddles,
        &|buf: &mut [Complex64; 8], inv| winograd::dft8(buf, inv),
        None,
    );
}

fn winograd_r8_inplace_32(data: &mut [Complex32], inverse: bool, twiddles: Option<&[Complex32]>) {
    debug_assert!(is_power_of_eight(data.len()));
    radix_stage::radix_winograd_inplace::<8, Complex32, _>(
        data,
        inverse,
        twiddles,
        &|buf: &mut [Complex32; 8], inv| winograd::dft8(buf, inv),
        None,
    );
}

// ── public API ────────────────────────────────────────────────────────────────

radix_kernel_api! {
    check       = is_power_of_eight,
    inplace64   = winograd_r8_inplace_64,
    inplace32   = winograd_r8_inplace_32,
    description = "power-of-eight",
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};
    use super::super::test_utils::max_abs_err_64;

    #[test]
    fn radix8_forward_n8_matches_direct() {
        let n = 8usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.31).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-11, "radix8 n=8 forward err={err:.2e}");
    }

    #[test]
    fn radix8_forward_n64_matches_direct() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.13).sin(), (k as f64 * 0.09).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix8 n=64 forward err={err:.2e}");
    }

    #[test]
    fn radix8_forward_n512_matches_direct() {
        let n = 512usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.07).sin(), (k as f64 * 0.04).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-9, "radix8 n=512 forward err={err:.2e}");
    }

    #[test]
    fn radix8_inverse_unnorm_n64_matches_direct() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.07).cos(), (k as f64 * 0.11).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input)
            .into_iter().map(|x| x * n as f64).collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix8 n=64 inverse unnorm err={err:.2e}");
    }

    #[test]
    fn radix8_roundtrip_n64() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.19).sin(), -(k as f64 * 0.23).cos()))
            .collect();
        let mut buf = input.clone();
        forward_inplace_64(&mut buf);
        inverse_inplace_64(&mut buf);
        let err = max_abs_err_64(&buf, &input);
        assert!(err < 1e-10, "radix8 roundtrip err={err:.2e}");
    }
}
