//! True radix-16 Cooley-Tukey kernels using Winograd DFT-16 inner butterflies.
//!
//! ## Algorithm
//!
//! Each stage processes groups of 16 elements using the Winograd DFT-16
//! kernel, which recursively decomposes via 2×DFT-8 (each 2×DFT-4 with
//! ±√2/2 twiddles) to reduce the inner butterfly from 256 generic
//! multiplications to 8 real multiplications per group.
//!
//! ## References
//!
//! - Winograd, S. (1978). On computing the discrete Fourier transform.
//!   *Mathematics of Computation*, 32(141), 175–199.

use super::radix_shape::is_power_of_pow2_radix;
use super::radix_stage;
use super::kernel_api::radix_kernel_api;
use super::winograd;
use super::radix2_f16::Cf16;
use num_complex::{Complex32, Complex64};

#[inline]
fn is_power_of_sixteen(n: usize) -> bool {
    is_power_of_pow2_radix(n, 4)
}

radix_kernel_api! {
    check       = is_power_of_sixteen,
    inplace64   = radix16_inplace_64,
    inplace32   = radix16_inplace_32,
    description = "power-of-sixteen",
}

fn radix16_inplace_64(
    data: &mut [Complex64],
    inverse: bool,
    twiddles: Option<&[Complex64]>,
) {
    debug_assert!((data.len().trailing_zeros() as usize) % 4 == 0);
    radix_stage::radix_winograd_inplace::<16, Complex64, _>(
        data,
        inverse,
        twiddles,
        &|buf: &mut [Complex64; 16], inv| winograd::dft16(buf, inv),
        None,
    );
}

fn radix16_inplace_32(
    data: &mut [Complex32],
    inverse: bool,
    twiddles: Option<&[Complex32]>,
) {
    debug_assert!((data.len().trailing_zeros() as usize) % 4 == 0);
    radix_stage::radix_winograd_inplace::<16, Complex32, _>(
        data,
        inverse,
        twiddles,
        &|buf: &mut [Complex32; 16], inv| winograd::dft16(buf, inv),
        None,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};
    use super::super::test_utils::max_abs_err_64;

    #[test]
    fn radix16_forward_n16_matches_direct() {
        let n = 16usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.23).sin(), (k as f64 * 0.07).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        assert!(max_abs_err_64(&got, &expected) < 1e-10);
    }

    #[test]
    fn radix16_inverse_unnorm_n16_matches_direct() {
        let n = 16usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.15).cos(), (k as f64 * 0.11).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        assert!(max_abs_err_64(&got, &expected) < 1e-10);
    }
}
