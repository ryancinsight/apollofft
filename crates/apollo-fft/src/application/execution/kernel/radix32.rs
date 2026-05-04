//! True radix-32 Cooley-Tukey kernels using Winograd DFT-32 inner butterflies.
//!
//! ## Algorithm
//!
//! Each stage processes groups of 32 elements using the Winograd DFT-32
//! kernel, which recursively decomposes as 2×DFT-16 (2×2×DFT-8 with exact
//! nested-sqrt twiddles) to reduce the inner butterfly from 1024 generic
//! multiplications to 16 real multiplications per group.
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
fn is_power_of_thirty_two(n: usize) -> bool {
    is_power_of_pow2_radix(n, 5)
}

radix_kernel_api! {
    check       = is_power_of_thirty_two,
    inplace64   = radix32_inplace_64,
    inplace32   = radix32_inplace_32,
    description = "power-of-thirty-two",
}

fn radix32_inplace_64(
    data: &mut [Complex64],
    inverse: bool,
    twiddles: Option<&[Complex64]>,
) {
    debug_assert!((data.len().trailing_zeros() as usize) % 5 == 0);
    radix_stage::radix_winograd_inplace::<32, Complex64, _>(
        data,
        inverse,
        twiddles,
        &|buf: &mut [Complex64; 32], inv| winograd::dft32(buf, inv),
        Some(RADIX_PARALLEL_CHUNK_THRESHOLD),
    );
}

fn radix32_inplace_32(
    data: &mut [Complex32],
    inverse: bool,
    twiddles: Option<&[Complex32]>,
) {
    debug_assert!((data.len().trailing_zeros() as usize) % 5 == 0);
    radix_stage::radix_winograd_inplace::<32, Complex32, _>(
        data,
        inverse,
        twiddles,
        &|buf: &mut [Complex32; 32], inv| winograd::dft32(buf, inv),
        Some(RADIX_PARALLEL_CHUNK_THRESHOLD),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};
    use super::super::test_utils::max_abs_err_64;

    #[test]
    fn radix32_forward_n32_matches_direct() {
        let n = 32usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.21).sin(), (k as f64 * 0.05).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        assert!(max_abs_err_64(&got, &expected) < 1e-10);
    }

    #[test]
    fn radix32_inverse_unnorm_n32_matches_direct() {
        let n = 32usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.09).cos(), (k as f64 * 0.12).sin()))
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
