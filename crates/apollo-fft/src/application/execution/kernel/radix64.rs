//! True radix-64 Cooley-Tukey kernels using Winograd DFT-64 inner butterflies.
//!
//! ## Algorithm
//!
//! Each stage processes groups of 64 elements using the Winograd DFT-64
//! kernel, which recursively decomposes as 2×DFT-32 to reduce the inner
//! butterfly from 4096 generic multiplications to 32 real multiplications
//! per group.
//!
//! ## References
//!
//! - Winograd, S. (1978). On computing the discrete Fourier transform.
//!   *Mathematics of Computation*, 32(141), 175–199.

use super::radix_shape::is_power_of_pow2_radix;
use super::radix_stage;
use super::tuning::RADIX_PARALLEL_CHUNK_THRESHOLD;
use super::kernel_api::radix_kernel_api;
use super::winograd;
use super::radix2_f16::Cf16;
use num_complex::{Complex32, Complex64};

#[inline]
fn is_power_of_sixty_four(n: usize) -> bool {
    is_power_of_pow2_radix(n, 6)
}

radix_kernel_api! {
    check       = is_power_of_sixty_four,
    inplace64   = radix64_inplace_64,
    inplace32   = radix64_inplace_32,
    description = "power-of-sixty-four",
}

fn radix64_inplace_64(
    data: &mut [Complex64],
    inverse: bool,
    twiddles: Option<&[Complex64]>,
) {
    debug_assert!((data.len().trailing_zeros() as usize) % 6 == 0);
    radix_stage::radix_winograd_inplace::<64, Complex64, _>(
        data,
        inverse,
        twiddles,
        &|buf: &mut [Complex64; 64], inv| winograd::dft64(buf, inv),
        Some(RADIX_PARALLEL_CHUNK_THRESHOLD),
    );
}

fn radix64_inplace_32(
    data: &mut [Complex32],
    inverse: bool,
    twiddles: Option<&[Complex32]>,
) {
    debug_assert!((data.len().trailing_zeros() as usize) % 6 == 0);
    radix_stage::radix_winograd_inplace::<64, Complex32, _>(
        data,
        inverse,
        twiddles,
        &|buf: &mut [Complex32; 64], inv| winograd::dft64(buf, inv),
        Some(RADIX_PARALLEL_CHUNK_THRESHOLD),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};
    use super::super::test_utils::max_abs_err_64;

    #[test]
    fn radix64_forward_n64_matches_direct() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.03).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        assert!(max_abs_err_64(&got, &expected) < 1e-10);
    }

    #[test]
    fn radix64_inverse_unnorm_n64_matches_direct() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.14).cos(), (k as f64 * 0.08).sin()))
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
