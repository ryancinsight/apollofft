//! Application boundary for real power-of-two FFT kernels.
//!
//! The 1D plan owns orchestration and buffer lifetime. Concrete CPU kernels
//! remain infrastructure-owned behind this module so plan code does not bind to
//! hardware-specific modules.

use crate::infrastructure::cpu::simd::power_of_two::radix2;
use num_complex::Complex64;

/// Build post-processing twiddles for the real-input split FFT.
#[inline]
pub(crate) fn build_real_fwd_post_twiddles(n: usize) -> Vec<Complex64> {
    radix2::build_real_fwd_post_twiddles(n)
}

/// Execute the real-input forward power-of-two FFT.
#[inline]
pub(crate) fn forward_real_inplace(
    input: &[f64],
    output: &mut [Complex64],
    fft_twiddles: &[Complex64],
    post_twiddles: &[Complex64],
) {
    radix2::forward_real_inplace(input, output, fft_twiddles, post_twiddles);
}

/// Execute the normalized real-output inverse power-of-two FFT.
#[inline]
pub(crate) fn inverse_real_inplace(
    input: &[Complex64],
    output: &mut [f64],
    scratch: &mut [Complex64],
    inverse_twiddles: &[Complex64],
    post_twiddles: &[Complex64],
) {
    radix2::inverse_real_inplace(input, output, scratch, inverse_twiddles, post_twiddles);
}
